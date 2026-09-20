"""Independent piece-count models and three-family cross-play.

Each stage has a complete network, optimizer state and champion/rollback slot.
Inference groups a batch by stage before calling each network, so MCTS keeps
its batched interface. The governed PRO big worker calls run_generation after
its own feedback pass; no second GPU service is created.
"""
from __future__ import annotations

import gc
import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chess
import chess.engine
import numpy as np
import torch
import torch.nn.functional as F

from autoloop.common import ROOT, STATE, atomic_json, digest, event, read_json, write_bundle
from autoloop.evaluation import paired_game_scores, paired_sign_test, reliability
from autoloop.replay_buffer import ExperienceReplay
from engine.engine import UniChessEngine
from model.net import NetConfig, UniChessNet, count_params
from model.train_iteration_fast import loss_for, save_atomic
from search.mcts import MCTS, MCTSConfig


@dataclass(frozen=True)
class Stage:
    name: str
    minimum: int
    maximum: int


# Five contiguous stages. Positions with <=5 pieces can still use Syzygy.
STAGES = (
    Stage("opening", 25, 32),
    Stage("middlegame", 19, 24),
    Stage("transition", 13, 18),
    Stage("endgame", 7, 12),
    Stage("tablebase", 2, 6),
)
LAYERED_CONFIG = NetConfig(blocks=16, filters=256)  # 19,883,363 parameters
LAYERED_PARAMS = count_params(UniChessNet(LAYERED_CONFIG))["TOTAL"]
PAIRINGS = (("small", "layered"), ("layered", "big"), ("small", "big"))


def stage_for_piece_count(count: int) -> int:
    for index, stage in enumerate(STAGES):
        if stage.minimum <= count <= stage.maximum:
            return index
    raise ValueError(f"unsupported piece count: {count}")


def stage_for_board(board: chess.Board) -> int:
    return stage_for_piece_count(chess.popcount(board.occupied))


def stage_paths(models: Path | None = None) -> list[Path]:
    models = models or STATE / "models"
    return [models / f"layered-stage{i}-champion.pt" for i in range(len(STAGES))]


class LayeredPool:
    """Memory-mapped labelled data indexed by the five stages."""

    def __init__(self, paths, cache: Path, *, per_stage_per_shard: int = 100_000):
        self.paths = list(paths)
        self.maps = [np.memmap(path, dtype=__import__("data.record", fromlist=["RECORD_DTYPE"]).RECORD_DTYPE, mode="r")
                     for path in self.paths]
        self.offsets = np.cumsum([0] + [len(data) for data in self.maps])
        self.total = int(self.offsets[-1])
        cache = Path(cache)
        manifest = cache.with_suffix(".json")
        fingerprint = json.dumps([(str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in self.paths])
        if cache.exists() and manifest.exists() and manifest.read_text() == fingerprint:
            with np.load(cache) as data:
                self.indices = [data[f"stage{i}"] for i in range(len(STAGES))]
        else:
            rng = np.random.default_rng(20260910)
            groups = [[] for _ in STAGES]
            for shard, data in enumerate(self.maps):
                selected = [[] for _ in STAGES]
                for start in range(0, len(data), 262_144):
                    records = data[start:start + 262_144]
                    occupied = np.asarray(records["occ_white"] | records["occ_black"]).copy()
                    counts = np.unpackbits(occupied.view(np.uint8).reshape(-1, 8), axis=1).sum(1)
                    for index, stage in enumerate(STAGES):
                        positions = np.flatnonzero((counts >= stage.minimum) & (counts <= stage.maximum))
                        if len(positions):
                            selected[index].append(positions + start + self.offsets[shard])
                for index, chunks in enumerate(selected):
                    if chunks:
                        values = np.concatenate(chunks)
                        if len(values) > per_stage_per_shard:
                            values = rng.choice(values, per_stage_per_shard, replace=False)
                        groups[index].append(values.astype(np.int64))
            self.indices = [np.concatenate(chunks) if chunks else np.empty(0, np.int64) for chunks in groups]
            np.savez(cache, **{f"stage{i}": values for i, values in enumerate(self.indices)})
            manifest.write_text(fingerprint)
        missing = [STAGES[i].name for i, values in enumerate(self.indices) if not len(values)]
        if missing:
            raise ValueError(f"layered stage pools are empty: {missing}")

    def sample(self, rng: np.random.Generator, n: int, stage: int, *, device: str = "cuda"):
        choices = self.indices[stage]
        indices = rng.choice(choices, size=n, replace=True)
        shards = np.searchsorted(self.offsets, indices, side="right") - 1
        dtype = __import__("data.record", fromlist=["RECORD_DTYPE"]).RECORD_DTYPE
        records = np.empty(n, dtype=dtype)
        for shard in np.unique(shards):
            selected = shards == shard
            records[selected] = self.maps[shard][indices[selected] - self.offsets[shard]]
        from model.dataset import decode_batch, decode_targets
        x = decode_batch(records)
        policy, promo, wdl = decode_targets(records)
        tensors = [torch.from_numpy(t) for t in (x, policy, promo, wdl)]
        tensors[0] = tensors[0].contiguous(memory_format=torch.channels_last)
        return [tensor.to(device) for tensor in tensors]


class LayeredRouter:
    """Route a batch by piece count and restore its original order."""

    def __init__(self, evaluators: dict[int, object]):
        self.evaluators = dict(evaluators)

    def evaluate_batch(self, boards: list[chess.Board]):
        if not boards:
            return (np.empty((0, 4096), np.float32),
                    np.empty((0, 4), np.float32),
                    np.empty((0, 3), np.float32))
        policy = np.empty((len(boards), 4096), np.float32)
        promo = np.empty((len(boards), 4), np.float32)
        wdl = np.empty((len(boards), 3), np.float32)
        groups: dict[int, list[int]] = {}
        for index, board in enumerate(boards):
            groups.setdefault(stage_for_board(board), []).append(index)
        for stage, indices in groups.items():
            values = self.evaluators[stage]([boards[i] for i in indices])
            policy[indices], promo[indices], wdl[indices] = values
        return policy, promo, wdl


class LayeredModelSet(LayeredRouter):
    """Load one independent CUDA engine per stage."""

    def __init__(self, paths: list[Path], *, device: str = "cuda", half: bool = True,
                 syzygy_path: Path | None = None):
        if len(paths) != len(STAGES):
            raise ValueError("one checkpoint is required per layered stage")
        self.paths = list(paths)
        self.engines = {}
        for index, path in enumerate(self.paths):
            if not Path(path).exists():
                raise FileNotFoundError(path)
            self.engines[index] = UniChessEngine(path, device=device, half=half, syzygy_path=syzygy_path)
            self.engines[index].model.to(memory_format=torch.channels_last)
        super().__init__({index: engine.evaluate_batch for index, engine in self.engines.items()})

    @property
    def tablebase(self):
        return next((engine.tablebase for engine in self.engines.values() if engine.tablebase is not None), None)

    @property
    def versions(self):
        return {STAGES[index].name: digest(path) for index, path in enumerate(self.paths)}

    def close(self):
        for engine in self.engines.values():
            if engine.tablebase is not None:
                try:
                    engine.tablebase.close()
                except Exception:
                    pass
        self.engines.clear()


def _targets(rows):
    from autoloop.worker import targets
    return targets(rows)


def _outcome(board):
    result = board.outcome(claim_draw=False)
    if result:
        return result.result(), result.termination.name
    if board.is_repetition(3):
        return "1/2-1/2", "CLAIMED_THREEFOLD"
    if board.is_fifty_moves():
        return "1/2-1/2", "CLAIMED_FIFTY_MOVES"
    return None, None


def _start_board(rng, heldout=True):
    from autoloop.worker import start_board
    return start_board(rng, heldout=heldout)


def _should_stop() -> bool:
    try:
        from autoloop.worker import STOP
        return STOP.is_set()
    except ImportError:
        return bool(os.environ.get("UNICHESS_LAYERED_STOP"))


def _parallel_map(function, tasks, env_name: str, default: int = 8):
    """Run independent games concurrently while sharing read-only engines."""
    try:
        workers = max(1, min(16, int(os.environ.get(env_name, default))))
    except (TypeError, ValueError):
        workers = default
    tasks = list(tasks)
    if workers == 1 or len(tasks) <= 1:
        return [function(task) for task in tasks]
    with ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
        futures = [pool.submit(function, task) for task in tasks]
        return [future.result() for future in futures]

def _row_stage(row: dict) -> int:
    pieces = row.get("pieces")
    if pieces is None:
        pieces = chess.popcount(chess.Board(row["fen"]).occupied)
    return stage_for_piece_count(int(pieces))


def _trace_result_wdl(fen: str, result: str | None):
    if result not in ('1-0', '0-1', '1/2-1/2'):
        return None
    white = 1 if result == '1-0' else (-1 if result == '0-1' else 0)
    z = white if chess.Board(fen).turn else -white
    return [int(z == 1), int(z == 0), int(z == -1)]


def _label_trace(rows: list[dict], result: str | None, source: str, termination: str | None = None, **fields) -> list[dict]:
    labelled = []
    for original in rows:
        label_result = result if result in ('1-0', '0-1', '1/2-1/2') else (
            '1/2-1/2' if termination == 'PLY_LIMIT' else None)
        target = _trace_result_wdl(original['fen'], label_result)
        if target is None or not original.get('policy'):
            continue
        row = dict(original)
        row['teacher_policy'] = list(row['policy'])
        row['teacher_wdl'] = target
        row['result_wdl'] = target
        row['source'] = source
        row['truncated'] = result is None
        row['importance'] = float(row.get('importance', 0.75 if result is None else 1.0)) * (1.0 + 0.35 * bool(row.get('promotion_threat')) + 0.15 * bool(row.get('repetition')))
        row['termination'] = termination
        row.update(fields)
        labelled.append(row)
    return labelled


def _stage_validation(checkpoint: Path, pool: LayeredPool, stage: int, *, batches: int = 4,
                      batch_size: int = 128) -> dict[str, float]:
    engine = UniChessEngine(checkpoint, device="cuda", half=True)
    engine.model.to(memory_format=torch.channels_last)
    values = []
    with torch.no_grad():
        for batch_no in range(batches):
            rng = np.random.default_rng(77291 + stage * 100 + batch_no)
            x, p, promo, w = pool.sample(rng, batch_size, stage)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                policy_logits, promo_logits, wdl_logits = engine.model(x)
            policy = policy_logits.float().softmax(1)
            predicted = wdl_logits.float().softmax(1)
            loss = (-(p * F.log_softmax(policy_logits.float(), 1)).sum(1).mean()
                    - (w * F.log_softmax(wdl_logits.float(), 1)).sum(1).mean())
            values.append((float(loss), float(((predicted - w) ** 2).sum(1).mean()),
                           float((predicted[:, 0] - predicted[:, 2] - (w[:, 0] - w[:, 2])).abs().mean()),
                           float((policy.argmax(1) == p.argmax(1)).float().mean())))
    if engine.tablebase is not None:
        engine.tablebase.close()
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return dict(zip(("loss", "wdl_brier", "expected_value_mae", "policy_top1"),
                    np.mean(values, axis=0).tolist()))


def _train_stage(stage: int, champion: Path | None, rows: list[dict], generation: int,
                 updates: int, train_pool: LayeredPool, seed_path: Path | None = None) -> Path | None:
    models = STATE / "models"
    checkpoint = models / f"layered-stage{stage}-recovery.pt"
    candidate = models / f"layered-stage{stage}-candidate.pt"
    cfg = LAYERED_CONFIG
    seed_source = seed_path or champion
    if seed_source is not None and seed_source.exists():
        seed = torch.load(seed_source, map_location="cpu", weights_only=False)
    else:
        torch.manual_seed(20260910 + stage)
        seed = {"model": UniChessNet(cfg).state_dict(), "cfg": cfg.__dict__}
    model = UniChessNet(NetConfig(**seed["cfg"])).cuda().to(memory_format=torch.channels_last)
    model.load_state_dict(seed["model"])
    parent = digest(champion) if champion is not None and champion.exists() else "bootstrap"
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4, fused=True)
    step = 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved.get("parent") == parent and saved.get("generation") == generation and saved.get("stage") == stage:
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            step = int(saved["step"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.cuda().contiguous(memory_format=torch.channels_last) if value.ndim == 4 else value.cuda()
        del saved
    del seed
    replay_rows = [row for row in rows if _row_stage(row) == stage]
    compiled = _targets(replay_rows) if replay_rows else None
    if compiled:
        compiled = [tensor.pin_memory() for tensor in compiled]
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
    batch = 1024
    started = time.time()
    initial_step = step
    while step < updates and not _should_stop():
        rng = np.random.default_rng(20260910 + generation * 10000 + stage * 1000 + step)
        optimizer.zero_grad(set_to_none=True)
        if compiled is not None:
            indices = torch.from_numpy(rng.integers(len(replay_rows), size=batch // 4))
            replay_batch = [tensor.index_select(0, indices).pin_memory() for tensor in compiled]
            from autoloop.worker import replay_loss
            replay_value, replay_metrics = replay_loss(model, [tensor.to("cuda", non_blocking=True) for tensor in replay_batch])
            (0.25 * replay_value).backward()
            anchor_value, _, _ = loss_for(model, train_pool.sample(rng, batch * 3 // 4, stage))
            (0.75 * anchor_value).backward()
        else:
            anchor_value, _, _ = loss_for(model, train_pool.sample(rng, batch, stage))
            anchor_value.backward()
            replay_value = anchor_value.detach() * 0.0
            replay_metrics = {}
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        if not bool(torch.isfinite(norm) and torch.isfinite(anchor_value) and torch.isfinite(replay_value)):
            raise FloatingPointError(f"non-finite layered stage {stage} update")
        optimizer.step()
        step += 1
        if step == initial_step + 1 or step % 10 == 0 or step == updates:
            fields = dict(stage=stage, stage_name=STAGES[stage].name, generation=generation,
                          step=step, updates=updates, replay_loss=float(replay_value),
                          anchor_loss=float(anchor_value), gradient_norm=float(norm),
                          lr=optimizer.param_groups[0]["lr"],
                          samples_per_second=(step - initial_step) * batch / max(0.01, time.time() - started),
                          params=LAYERED_PARAMS)
            fields.update({key: float(value) for key, value in replay_metrics.items()})
            event("layered", "train", **fields)
        if step % 50 == 0 or step == updates:
            save_atomic({"model": model.state_dict(), "cfg": cfg.__dict__, "optimizer": optimizer.state_dict(),
                         "step": step, "generation": generation, "stage": stage, "parent": parent}, checkpoint)
    if _should_stop():
        save_atomic({"model": model.state_dict(), "cfg": cfg.__dict__, "optimizer": optimizer.state_dict(),
                     "step": step, "generation": generation, "stage": stage, "parent": parent}, checkpoint)
        return None
    save_atomic({"model": model.state_dict(), "cfg": cfg.__dict__, "step": step,
                 "generation": generation, "stage": stage, "parent": parent}, candidate)
    del model, optimizer, compiled
    gc.collect()
    torch.cuda.empty_cache()
    return candidate


def stage_gate(candidate: Path, champion: Path | None, pool: LayeredPool, stage: int) -> dict:
    candidate_validation = _stage_validation(candidate, pool, stage)
    if champion is None or not champion.exists():
        return {"accepted": True, "reason": "bootstrap", "candidate_validation": candidate_validation,
                "champion_validation": None, "validation_pass": True}
    champion_validation = _stage_validation(champion, pool, stage)
    validation_pass = (candidate_validation["loss"] <= champion_validation["loss"] * 1.05
                       and candidate_validation["wdl_brier"] <= champion_validation["wdl_brier"] * 1.10
                       and candidate_validation["expected_value_mae"] <= champion_validation["expected_value_mae"] * 1.10)
    improved = (candidate_validation["loss"] < champion_validation["loss"] * 0.998
                or candidate_validation["wdl_brier"] < champion_validation["wdl_brier"] * 0.995)
    return {"accepted": bool(validation_pass and improved),
            "reason": "heldout_gate" if validation_pass else "validation_regression",
            "candidate_validation": candidate_validation, "champion_validation": champion_validation,
            "validation_pass": validation_pass, "improved": improved}


def _play_game(players: dict[str, object], white: str, black: str, start: str,
               seed: int, simulations: int = 300, max_plies: int = 320) -> dict:
    rng = np.random.default_rng(seed)
    board = chess.Board(start)
    searches = {name: MCTS(player.evaluate_batch if hasattr(player, "evaluate_batch") else player,
                           MCTSConfig(simulations=simulations, batch_size=64, claim_draw=True),
                           tablebase=getattr(player, "tablebase", None), rng=rng)
                for name, player in players.items()}
    moves, history, trace = [], [], []
    trees = {name: None for name in players}
    started = time.time()
    termination = "PLY_LIMIT"
    while len(moves) < max_plies and not _should_stop():
        result, reason = _outcome(board)
        if result:
            termination = reason
            break
        name = white if board.turn else black
        before = time.perf_counter()
        move, root = searches[name].best_move(board, root=trees[name])
        if move not in board.legal_moves:
            termination = "ILLEGAL_MOVE"
            break
        visits = [[candidate.uci(), float(prob)] for candidate, prob in searches[name].visit_policy(root)]
        trace.append(dict(start=start, history=list(history), fen=board.fen(), policy=visits,
                          q=searches[name].root_value(root), actor=name, actor_role=name,
                          root_visits=int(root.N.sum()), search_seconds=time.perf_counter() - before,
                          root_entropy=float(-sum(prob * math.log(max(prob, 1e-12)) for _, prob in visits)),
                          pieces=chess.popcount(board.occupied), chosen=move.uci(),
                          priors=[[candidate.uci(), float(prob)] for candidate, prob in zip(root.moves, root.P)],
                          action_q=[[candidate.uci(), float(w / n) if n else None]
                                    for candidate, w, n in zip(root.moves, root.W, root.N)]))
        for player_name in trees:
            trees[player_name] = searches[player_name].advance_root(
                root if player_name == name else trees[player_name], move)
        moves.append(move.uci())
        board.push(move)
        history.append(move.uci())
    result, reason = _outcome(board)
    if result:
        termination = reason
    elif termination != "ILLEGAL_MOVE":
        result = None
    trace = _label_trace(trace, result, 'model_crossplay', termination=termination)
    for row in trace:
        row['opponent'] = black if row['actor_role'] == white else white
    return {"start": start, "moves": moves, "result": result, "termination": termination,
            "white": white, "black": black, "seed": seed, "plies": len(moves),
            "seconds": time.time() - started, "illegal": termination == "ILLEGAL_MOVE", "trace": trace}


def _crossplay(players: dict[str, object], generation: int, *, pairs: int = 4,
               simulations: int = 300, max_plies: int = 320, prefix: str = "joint") -> dict:
    games = []
    training_rows = []
    tasks = []
    for pairing_index, (white, black) in enumerate(PAIRINGS):
        for pair in range(pairs):
            start = _start_board(np.random.default_rng(910000 + generation * 1000 + pairing_index * 100 + pair)).fen()
            for candidate_white in (True, False):
                actual_white, actual_black = (white, black) if candidate_white else (black, white)
                tasks.append((actual_white, actual_black, start,
                              910000 + generation * 1000 + pairing_index * 1000 + pair * 2 + int(not candidate_white),
                              pair, f"{white}_vs_{black}", candidate_white))
    def play(task):
        actual_white, actual_black, start, seed, pair, pairing, candidate_white = task
        result = _play_game(players, actual_white, actual_black, start,
                            seed, simulations, max_plies)
        result.update(pair=pair, pairing=pairing, candidate_white=candidate_white)
        return result

    for game in _parallel_map(play, tasks, 'UNICHESS_LAYERED_CROSSPLAY_WORKERS'):
        training_rows.extend(game.get('trace', []))
        game.pop('trace', None)
        games.append(game)
    stats = reliability(games)
    by_pairing = {}
    for white, black in PAIRINGS:
        pairing = f"{white}_vs_{black}"
        subset = [game for game in games if game["pairing"] == pairing]
        scores = paired_game_scores(subset)
        by_pairing[pairing] = dict(paired_sign_test(scores), **reliability(subset))
    if training_rows:
        write_bundle(STATE / 'population' / f'crossplay-{generation}-{time.time_ns()}.json.gz', training_rows)
    event('layered', 'crossplay_training', generation=generation, games=len(games),
          complete_games=sum(game.get('result') is not None for game in games), rows=len(training_rows),
          source='model_crossplay')
    report = {"generation": generation, "pairings": len(PAIRINGS), "games": len(games),
              "pair_count": pairs, "simulations": simulations, "by_pairing": by_pairing,
              "training_rows": len(training_rows), **stats}
    write_bundle(STATE / "games" / f"{prefix}-{generation}-{time.time_ns()}.json.gz", games)
    event("layered", "joint_crossplay", **report)
    return report



def _stockfish_game(player, name: str, start: str, model_white: bool, seed: int,
                    stockfish, *, simulations: int, nodes: int, max_plies: int):
    board = chess.Board(start)
    model_color = chess.WHITE if model_white else chess.BLACK
    search = MCTS(player.evaluate_batch if hasattr(player, 'evaluate_batch') else player,
                  MCTSConfig(simulations=simulations, batch_size=64, claim_draw=True),
                  tablebase=getattr(player, 'tablebase', None),
                  rng=np.random.default_rng(seed))
    tree = None
    moves, history, trace = [], [], []
    started = time.time()
    termination = 'PLY_LIMIT'
    while len(moves) < max_plies and not _should_stop():
        result, reason = _outcome(board)
        if result:
            termination = reason
            break
        if board.turn == model_color:
            before = time.perf_counter()
            move, root = search.best_move(board, root=tree)
            visits = [[candidate.uci(), float(prob)] for candidate, prob in search.visit_policy(root)]
            trace.append(dict(start=start, history=list(history), fen=board.fen(), policy=visits,
                              q=search.root_value(root), actor=name, actor_role=name,
                              root_visits=int(root.N.sum()), search_seconds=time.perf_counter() - before,
                              root_entropy=float(-sum(prob * math.log(max(prob, 1e-12)) for _, prob in visits)),
                              pieces=chess.popcount(board.occupied), chosen=move.uci(),
                              priors=[[candidate.uci(), float(prob)] for candidate, prob in zip(root.moves, root.P)],
                              action_q=[[candidate.uci(), float(w / n) if n else None]
                                        for candidate, w, n in zip(root.moves, root.W, root.N)]))
            next_tree = search.advance_root(root, move)
        else:
            move = stockfish.play(board, chess.engine.Limit(nodes=nodes)).move
            next_tree = search.advance_root(tree, move)
        if move not in board.legal_moves:
            termination = 'ILLEGAL_MOVE'
            break
        moves.append(move.uci())
        board.push(move)
        history.append(move.uci())
        tree = next_tree
    result, reason = _outcome(board)
    if result:
        termination = reason
    elif termination != 'ILLEGAL_MOVE':
        result = None
    trace = _label_trace(trace, result, 'model_stockfish', termination=termination, opponent='stockfish')
    return ({'start': start, 'moves': moves, 'result': result, 'termination': termination,
             'white': name if model_white else 'stockfish',
             'black': 'stockfish' if model_white else name, 'seed': seed,
             'plies': len(moves), 'seconds': time.time() - started,
             'illegal': termination == 'ILLEGAL_MOVE', 'family': name}, trace)


def _stockfish_crossplay(players: dict[str, object], generation: int, *, pairs: int = 1,
                         simulations: int = 160, nodes: int = 100000, max_plies: int = 320) -> dict:
    games, training_rows = [], []
    try:
        with chess.engine.SimpleEngine.popen_uci(str(ROOT / 'tools/stockfish')) as sf:
            sf.configure({'Threads': 2, 'Hash': 256})
            for family, player in players.items():
                for pair in range(pairs):
                    start = _start_board(np.random.default_rng(950000 + generation * 100 + pair)).fen()
                    for model_white in (True, False):
                        game, rows = _stockfish_game(
                            player, family, start, model_white,
                            950000 + generation * 1000 + pair * 2 + int(not model_white),
                            sf, simulations=simulations, nodes=nodes, max_plies=max_plies)
                        games.append(game)
                        training_rows.extend(rows)
    except Exception as exc:
        event('layered', 'stockfish_crossplay', generation=generation, enabled=True,
              games=len(games), rows=len(training_rows), error=repr(exc))
        return {'generation': generation, 'enabled': True, 'games': len(games),
                'training_rows': len(training_rows), 'error': repr(exc)}
    if training_rows:
        write_bundle(STATE / 'population' / f'stockfish-{generation}-{time.time_ns()}.json.gz', training_rows)
    families = {}
    for family in players:
        subset = [game for game in games if game['family'] == family]
        families[family] = dict(paired_sign_test(paired_game_scores(subset)), **reliability(subset))
    report = {'generation': generation, 'enabled': True, 'games': len(games),
              'training_rows': len(training_rows), 'families': families,
              **reliability(games)}
    write_bundle(STATE / 'games' / f'stockfish-{generation}-{time.time_ns()}.json.gz', games)
    event('layered', 'stockfish_crossplay', **report)
    return report


def _progress_eval(current: dict[str, object], reference: dict[str, object], generation: int,
                   *, pairs: int = 4, simulations: int = 200, max_plies: int = 320) -> dict:
    """Compare current models with frozen first-generation references.

    Starts and colors are fixed for every invocation.  This is a longitudinal
    signal, not a promotion gate: a candidate must still pass the live
    champion gate before it can be routed to actors.
    """
    families = {}
    games = []
    for family in ("small", "layered", "big"):
        family_games = []
        for pair in range(pairs):
            start = _start_board(np.random.default_rng(930000 + pair)).fen()
            for current_white in (True, False):
                white = "current" if current_white else "reference"
                black = "reference" if current_white else "current"
                game = _play_game({"current": current[family], "reference": reference[family]},
                                  white, black, start, 930000 + pair * 2 + int(not current_white),
                                  simulations, max_plies)
                game.update(pair=pair, family=family, candidate_white=current_white)
                family_games.append(game)
        scores = paired_game_scores(family_games)
        stats = dict(paired_sign_test(scores), **reliability(family_games))
        stats["strong_positive_signal"] = bool(stats["reliable"] and stats["pairs"] == pairs
                                               and stats["score"] > 0.5
                                               and stats["paired_sign_p"] < 0.05)
        families[family] = stats
        games.extend(family_games)
    report = {"generation": generation, "pairs_per_family": pairs,
              "simulations": simulations, "families": families, **reliability(games)}
    write_bundle(STATE / "games" / f"progress-{generation}-{time.time_ns()}.json.gz", games)
    event("layered", "true_progress", **report)
    return report


def run_generation(generation: int, rows: list[dict], *, small_checkpoint: Path,
                   big_checkpoint: Path, smoke: bool = False) -> dict:
    """Train all stages, then evaluate small/layered/big in mirrored games."""
    # Keep a separate stage-balanced reservoir for the five independent models.
    # The big worker has already applied its feedback replacements to `rows`.
    layered_store = ExperienceReplay('layered')
    external_rows = layered_store.refresh()
    layered_rows = layered_store.refresh(source_rows=rows + external_rows)
    rows = layered_rows or rows
    models = STATE / "models"
    models.mkdir(parents=True, exist_ok=True)
    state_path = STATE / "layered-state.json"
    state = read_json(state_path, {"generation": generation, "stage": 0, "phase": "ready"})
    paths = stage_paths(models)
    data_paths = sorted((ROOT / "data/shards_evals").glob("*.bin"))
    train_pool = LayeredPool(data_paths[:-4], ROOT / "runs/iteration46_20260909/layered-train-pools.npz")
    valid_pool = LayeredPool(data_paths[-4:], ROOT / "runs/iteration46_20260909/layered-valid-pools.npz")
    stage_reports = []
    bootstrap_steps = int(os.environ.get("UNICHESS_LAYERED_BOOTSTRAP_STEPS", "2000"))
    update_cap = int(os.environ.get("UNICHESS_LAYERED_UPDATE_CAP", "300"))
    stage_start = int(state.get("stage", 0)) if state.get("generation") == generation else 0
    for stage in range(stage_start, len(STAGES)):
        champion = paths[stage] if paths[stage].exists() else None
        learner_path = models / f"layered-stage{stage}-learner.pt"
        seed_path = champion
        if champion is not None and learner_path.exists():
            try:
                learner_meta = torch.load(learner_path, map_location="cpu", weights_only=False)
                if learner_meta.get("parent") == digest(champion) and learner_meta.get("cfg") == LAYERED_CONFIG.__dict__:
                    seed_path = learner_path
            except Exception:
                pass
        stage_rows = [row for row in rows if _row_stage(row) == stage]
        if state.get("phase") == "training" and state.get("generation") == generation and state.get("stage") == stage:
            updates = int(state.get("active_updates", bootstrap_steps if champion is None else update_cap))
        else:
            updates = (int(os.environ.get("UNICHESS_LAYERED_SMOKE_UPDATES", "2")) if smoke else
                       (bootstrap_steps if champion is None else min(update_cap, max(8, math.ceil(max(len(stage_rows), 256) * 4 / 256)))))
            state.update(generation=generation, phase="training", stage=stage, active_updates=updates)
            atomic_json(state_path, state)
        atomic_json(STATE / "layered-status.json", {"time": time.time(), "phase": "training", "generation": generation,
                                                    "stage": stage, "stage_name": STAGES[stage].name,
                                                    "stages": len(STAGES), "rows": len(stage_rows), "params": LAYERED_PARAMS})
        candidate = _train_stage(stage, champion, stage_rows, generation, updates, train_pool, seed_path=seed_path)
        if candidate is None:
            return {"generation": generation, "phase": "paused", "stage": stage,
                    "stage_reports": stage_reports}
        report = stage_gate(candidate, champion, valid_pool, stage)
        report.update(stage=stage, stage_name=STAGES[stage].name, generation=generation)
        stage_reports.append(report)
        event("layered", "stage_gate", **report)
        atomic_json(STATE / f"layered-stage{stage}-last-gate.json", report)
        if report["accepted"]:
            if champion is not None:
                shutil.copy2(champion, models / f"layered-stage{stage}-previous.pt")
            os.replace(candidate, paths[stage])
            event("layered", "stage_promoted", generation=generation, stage=stage,
                  stage_name=STAGES[stage].name, champion=digest(paths[stage]),
                  parent=digest(champion) if champion else None)
        else:
            shutil.copy2(candidate, models / f"layered-stage{stage}-learner.pt")
            candidate.unlink(missing_ok=True)
        (models / f"layered-stage{stage}-learner.pt").unlink(missing_ok=True) if report["accepted"] else None
        state.update(phase="ready", stage=stage + 1)
        atomic_json(state_path, state)
    if not all(path.exists() for path in paths):
        report = {"generation": generation, "phase": "waiting_for_stage_models", "stage_reports": stage_reports}
        atomic_json(STATE / "layered-status.json", {"time": time.time(), "phase": report["phase"],
                                                    "generation": generation, "stages": len(STAGES),
                                                    "ready_stages": sum(path.exists() for path in paths),
                                                    "params": LAYERED_PARAMS})
        return report
    if not small_checkpoint.exists():
        report = {"generation": generation, "phase": "waiting_for_small_snapshot", "stage_reports": stage_reports,
                  "ready_stages": len(paths), "required_families": 3}
        event("layered", "joint_crossplay_wait", **report)
        atomic_json(STATE / "layered-status.json", {"time": time.time(), "phase": report["phase"],
                                                    "generation": generation, "stages": len(STAGES),
                                                    "ready_stages": len(paths), "params": LAYERED_PARAMS})
        return report
    reference_small = models / "reference-small.pt"
    reference_big = models / "reference-big.pt"
    reference_layered = [models / f"reference-layered-stage{i}.pt" for i in range(len(STAGES))]
    references_ready = reference_small.exists() and reference_big.exists() and all(path.exists() for path in reference_layered)
    baseline_initialized = False
    if not references_ready:
        shutil.copy2(small_checkpoint, reference_small)
        shutil.copy2(big_checkpoint, reference_big)
        for source, target in zip(paths, reference_layered):
            shutil.copy2(source, target)
        baseline_initialized = True
    syzygy = ROOT / "data/raw/syzygy345"
    layered = LayeredModelSet(paths, syzygy_path=syzygy if syzygy.is_dir() else None)
    small = UniChessEngine(small_checkpoint, device="cuda", half=True)
    big = UniChessEngine(big_checkpoint, device="cuda", half=True)
    for engine in (small, big):
        engine.model.to(memory_format=torch.channels_last)
    players = {"small": small, "layered": layered, "big": big}
    crossplay = _crossplay(players, generation, pairs=1 if smoke else 4,
                           simulations=32 if smoke else int(os.environ.get("UNICHESS_LAYERED_CROSSPLAY_SIMS", "300")),
                           max_plies=16 if smoke else 320)
    if smoke:
        stockfish = {'generation': generation, 'enabled': False, 'reason': 'smoke'}
    else:
        every = max(1, int(os.environ.get('UNICHESS_STOCKFISH_EVERY', '3')))
        stockfish = (_stockfish_crossplay(
            players, generation,
            pairs=max(1, int(os.environ.get('UNICHESS_STOCKFISH_PAIRS', '1'))),
            simulations=int(os.environ.get('UNICHESS_STOCKFISH_SIMS', '160')),
            nodes=int(os.environ.get('UNICHESS_STOCKFISH_NODES', '100000')),
            max_plies=int(os.environ.get('UNICHESS_STOCKFISH_PLIES', '320')))
            if generation % every == 0 else
            {'generation': generation, 'enabled': False, 'reason': 'cadence', 'every': every})
    progress = {"baseline_initialized": baseline_initialized}
    progress_interval = max(1, int(os.environ.get("UNICHESS_LAYERED_PROGRESS_EVERY", "5")))
    if baseline_initialized:
        progress["status"] = "baseline_initialized"
    elif not smoke and generation % progress_interval == 0:
        reference_layered_set = LayeredModelSet(reference_layered, syzygy_path=syzygy if syzygy.is_dir() else None)
        reference_small_engine = UniChessEngine(reference_small, device="cuda", half=True)
        reference_big_engine = UniChessEngine(reference_big, device="cuda", half=True)
        reference_players = {"small": reference_small_engine, "layered": reference_layered_set, "big": reference_big_engine}
        progress = _progress_eval(players, reference_players, generation,
                                  pairs=int(os.environ.get("UNICHESS_LAYERED_PROGRESS_PAIRS", "4")),
                                  simulations=int(os.environ.get("UNICHESS_LAYERED_PROGRESS_SIMS", "200")))
        progress["baseline_initialized"] = False
        event("layered", "progress_snapshot", generation=generation,
              strong_families=sum(item.get("strong_positive_signal", False)
                                  for item in progress["families"].values()))
        for reference_engine in (reference_small_engine, reference_big_engine):
            if reference_engine.tablebase is not None:
                reference_engine.tablebase.close()
        reference_layered_set.close()
    atomic_json(STATE / "layered-progress.json", progress)
    report = {"generation": generation, "phase": "complete", "params": LAYERED_PARAMS,
              "stage_reports": stage_reports, "crossplay": crossplay,
              "stockfish": stockfish, "progress": progress}
    event("layered", "generation_complete", generation=generation,
          promoted_stages=sum(item["accepted"] for item in stage_reports),
          completion_rate=crossplay["completion_rate"], unknown_games=crossplay["unknown_games"])
    atomic_json(STATE / "layered-last-generation.json", report)
    atomic_json(STATE / "layered-status.json", {"time": time.time(), "phase": "complete", "generation": generation,
                                                "stages": len(STAGES), "ready_stages": len(paths),
                                                "params": LAYERED_PARAMS, "crossplay": crossplay, "stockfish": stockfish, "progress": progress})
    state.update(generation=generation + 1, phase="complete", stage=0)
    atomic_json(state_path, state)
    if small.tablebase is not None:
        small.tablebase.close()
    if big.tablebase is not None:
        big.tablebase.close()
    layered.close()
    del small, big, layered, train_pool, valid_pool
    gc.collect()
    torch.cuda.empty_cache()
    return report

