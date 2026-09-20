"""Standard AlphaZero self-play/training loop.

One role owns one local replay window and one champion.  The loop is deliberately
plain: self-play from the initial position with MCTS, uniform recent-position
replay, SGD training, and a fixed arena against the current champion.  There is
no Stockfish teacher, cross-family feedback, priority replay, progress heuristic,
or cross-machine replay exchange.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gc
import multiprocessing as mp
import os
import queue
import shutil
import signal
import threading
import time
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn.functional as F

from autoloop.common import (
    ROOT, STATE, atomic_json, digest, event, lock, maintain_storage,
    read_bundle, read_json, setup, write_bundle,
)
from core.encoding import encode, orient_move
from core.moves import move_to_index, move_to_promo_index
from engine.engine import UniChessEngine
from model.net import NetConfig, UniChessNet
from search.mcts import MCTS, MCTSConfig

STOP = threading.Event()
STAGE_RANGES = ((25, 32), (19, 24), (13, 18), (7, 12), (2, 6))
STAGE_NAMES = ("opening", "middlegame", "transition", "endgame", "tablebase")
AZ_REPLAY = STATE / "az_replay"


def handle_stop(*_):
    STOP.set()


def _int_env(name: str, default: int, lower: int, upper: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(lower, min(upper, value))


def _float_env(name: str, default: float, lower: float, upper: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(lower, min(upper, value))


def stage_for_board(board: chess.Board) -> int:
    pieces = chess.popcount(board.occupied)
    for index, (minimum, maximum) in enumerate(STAGE_RANGES):
        if minimum <= pieces <= maximum:
            return index
    raise ValueError(f"unsupported piece count: {pieces}")


def result_wdl(board: chess.Board, result: str) -> list[int]:
    white = 1 if result == "1-0" else (-1 if result == "0-1" else 0)
    value = white if board.turn else -white
    return [int(value == 1), int(value == 0), int(value == -1)]


def board_outcome(board: chess.Board) -> tuple[str | None, str | None]:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None, None
    return outcome.result(), outcome.termination.name


class RecentReplay:
    """The AlphaZero replay window: newest games only, uniform position sampling."""

    def __init__(self, path: Path = AZ_REPLAY, capacity_games: int = 5000):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.capacity_games = max(1, int(capacity_games))
        self.meta = self.path / "metadata.json"

    def _files(self) -> list[Path]:
        return sorted(self.path.glob("game-*.json.gz"),
                      key=lambda item: item.stat().st_mtime, reverse=True)

    def append(self, generation: int, rows: list[dict]) -> None:
        if not rows:
            return
        write_bundle(self.path / f"game-{generation:08d}-{time.time_ns()}.json.gz", rows)
        files = self._files()
        for stale in files[self.capacity_games:]:
            stale.unlink(missing_ok=True)
        atomic_json(self.meta, {
            "capacity_games": self.capacity_games,
            "games": min(len(files), self.capacity_games),
            "positions": sum(len(read_bundle(path)) for path in files[:self.capacity_games]),
            "updated_at": time.time(),
        })

    def rows(self) -> list[dict]:
        rows: list[dict] = []
        for path in self._files()[:self.capacity_games]:
            try:
                rows.extend(read_bundle(path))
            except (OSError, EOFError, ValueError, TypeError):
                continue
        return rows


def _policy_target(board: chess.Board, visits) -> tuple[np.ndarray, np.ndarray]:
    policy = np.zeros(4096, dtype=np.float32)
    promotion = np.full(4, -100, dtype=np.int64)
    promotion_mass = np.zeros(4, dtype=np.float32)
    for uci, probability in visits:
        move = chess.Move.from_uci(str(uci))
        if move not in board.legal_moves:
            raise ValueError(f"illegal AlphaZero visit target: {uci}")
        oriented = orient_move(move, board.turn)
        policy[move_to_index(oriented)] += float(probability)
        promo_index = move_to_promo_index(oriented)
        if promo_index is not None:
            promotion_mass[promo_index] += float(probability)
    total = float(policy.sum())
    if total <= 0:
        legal = list(board.legal_moves)
        for move in legal:
            oriented = orient_move(move, board.turn)
            policy[move_to_index(oriented)] = 1.0 / len(legal)
    else:
        policy /= total
    if promotion_mass.sum() > 0:
        promotion = promotion_mass / promotion_mass.sum()
    return policy, promotion


def compile_rows(rows: list[dict]) -> list[torch.Tensor]:
    states, policies, promotions, values = [], [], [], []
    for row in rows:
        board = chess.Board(row["fen"])
        policy, promotion = _policy_target(board, row["policy"])
        states.append(encode(board).astype(np.float32, copy=False))
        policies.append(policy)
        promotions.append(promotion)
        values.append(np.asarray(row["wdl"], dtype=np.float32))
    if not states:
        raise ValueError("AlphaZero replay window is empty")
    tensors = [
        torch.from_numpy(np.asarray(states, dtype=np.float32)),
        torch.from_numpy(np.asarray(policies, dtype=np.float32)),
        torch.from_numpy(np.asarray(promotions, dtype=np.int64)),
        torch.from_numpy(np.asarray(values, dtype=np.float32)),
    ]
    tensors[0] = tensors[0].contiguous(memory_format=torch.channels_last)
    return tensors


def alpha_zero_loss(model, batch):
    x, policy_target, promotion_target, value_target = batch
    with torch.autocast("cuda", dtype=torch.bfloat16):
        policy_logits, promotion_logits, value_logits = model(
            x.contiguous(memory_format=torch.channels_last)
        )
    policy_loss = -(policy_target * F.log_softmax(policy_logits.float(), 1)).sum(1).mean()
    value_loss = -(value_target * F.log_softmax(value_logits.float(), 1)).sum(1).mean()
    mask = promotion_target >= 0
    if mask.any():
        promotion_loss = F.cross_entropy(
            promotion_logits.float()[mask],
            promotion_target[mask],
        )
    else:
        promotion_loss = policy_loss.detach() * 0.0
    total = policy_loss + value_loss + 0.1 * promotion_loss
    with torch.no_grad():
        predicted = value_logits.float().softmax(1)
        brier = ((predicted - value_target) ** 2).sum(1).mean()
    return total, {
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "promotion_loss": promotion_loss.detach(),
        "brier": brier.detach(),
    }


# The actor-side evaluator is only a proxy.  CUDA remains in the parent process.
_ACTOR_REQUESTS = None
_ACTOR_RESPONSES = None


def _actor_init(requests, responses):
    global _ACTOR_REQUESTS, _ACTOR_RESPONSES
    _ACTOR_REQUESTS = requests
    _ACTOR_RESPONSES = responses
    torch.set_num_threads(1)
    signal.signal(signal.SIGTERM, handle_stop)


class _ActorEvaluator:
    def __init__(self, index: int):
        self.index = index

    def __call__(self, boards):
        _ACTOR_REQUESTS.put((self.index, boards))
        ok, payload = _ACTOR_RESPONSES[self.index].get(timeout=180)
        if not ok:
            raise RuntimeError(payload)
        return payload


class _ProcessBatcher:
    def __init__(self, evaluator, requests, responses, batch_limit: int, wait_ms: float):
        self.evaluator = evaluator
        self.requests = requests
        self.responses = responses
        self.batch_limit = max(64, int(batch_limit))
        self.wait = max(0.001, float(wait_ms) / 1000.0)
        self.closed = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.closed:
            try:
                first = self.requests.get(timeout=0.1)
            except queue.Empty:
                continue
            request_list = [first]
            positions = len(first[1])
            deadline = time.monotonic() + self.wait
            while positions < self.batch_limit and time.monotonic() < deadline:
                try:
                    item = self.requests.get(timeout=0.001)
                except queue.Empty:
                    continue
                request_list.append(item)
                positions += len(item[1])
            try:
                boards = [board for _, batch in request_list for board in batch]
                outputs = self.evaluator.evaluate_batch(boards)
                offset = 0
                for index, batch in request_list:
                    size = len(batch)
                    self.responses[index].put(
                        (True, tuple(value[offset:offset + size] for value in outputs))
                    )
                    offset += size
            except Exception as exc:
                for index, _ in request_list:
                    self.responses[index].put((False, repr(exc)))

    def close(self):
        self.closed = True
        self.thread.join(timeout=5)


def play_game(evaluator, seed: int, model_version: str, simulations: int,
              max_plies: int, temperature_moves: int = 30) -> tuple[list[dict], dict]:
    evaluator = getattr(evaluator, "evaluate_batch", evaluator)
    rng = np.random.default_rng(seed)
    board = chess.Board()
    search = MCTS(
        evaluator,
        MCTSConfig(
            simulations=simulations,
            batch_size=_int_env("UNICHESS_AZ_MCTS_BATCH", 128, 16, 512),
            claim_draw=True,
        ),
        rng=rng,
    )
    tree = None
    trajectory = []
    moves = []
    started = time.time()
    while len(moves) < max_plies and not STOP.is_set():
        result, reason = board_outcome(board)
        if result is not None:
            break
        move, root = search.best_move(
            board,
            temperature=1.0 if len(moves) < temperature_moves else 0.0,
            add_noise=True,
            root=tree,
        )
        visits = [[candidate.uci(), float(probability)]
                  for candidate, probability in search.visit_policy(root)]
        trajectory.append({
            "fen": board.fen(),
            "policy": visits,
            "turn": "white" if board.turn else "black",
            "stage": stage_for_board(board),
            "model": model_version,
        })
        tree = search.advance_root(root, move)
        moves.append(move.uci())
        board.push(move)
    result, reason = board_outcome(board)
    if result is None and not STOP.is_set() and len(moves) >= max_plies:
        result, reason = "1/2-1/2", "PLY_LIMIT_DRAW"
    if result is None:
        reason = "INTERRUPTED"
        return [], {
            "seed": seed, "model": model_version, "moves": moves,
            "result": None, "termination": reason, "plies": len(moves),
            "seconds": time.time() - started,
        }
    for row in trajectory:
        row["wdl"] = result_wdl(chess.Board(row["fen"]), result)
    return trajectory, {
        "seed": seed, "model": model_version, "moves": moves,
        "result": result, "termination": reason, "plies": len(moves),
        "seconds": time.time() - started, "positions": len(trajectory),
    }


def _actor_loop(index, tasks, results, requests, responses, model_version,
                simulations, max_plies):
    _actor_init(requests, responses)
    while True:
        try:
            task = tasks.get(timeout=1)
        except queue.Empty:
            continue
        if task is None:
            return
        seed = int(task)
        try:
            rows, summary = play_game(
                _ActorEvaluator(index), seed, model_version, simulations, max_plies
            )
            results.put((index, rows, summary))
        except Exception as exc:
            results.put((index, [], {
                "seed": seed, "model": model_version, "result": None,
                "termination": "ERROR", "error": repr(exc),
            }))


def collect_selfplay(evaluator, generation: int, model_version: str, role: str,
                     replay_path: Path, games: int, simulations: int, max_plies: int,
                     actor_count: int, batch_limit: int, wait_ms: float) -> int:
    if games <= 0:
        return 0
    replay_path = Path(replay_path)
    replay_path.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    requests = context.Queue(maxsize=max(16, actor_count * 2))
    responses = [context.Queue(maxsize=2) for _ in range(actor_count)]
    tasks = context.Queue(maxsize=max(1, actor_count * 2))
    results = context.Queue()
    batcher = _ProcessBatcher(evaluator, requests, responses, batch_limit, wait_ms)
    actors = [
        context.Process(
            target=_actor_loop,
            args=(index, tasks, results, requests, responses, model_version,
                  simulations, max_plies),
            daemon=True,
        )
        for index in range(actor_count)
    ]
    for actor in actors:
        actor.start()
    for index in range(games):
        tasks.put(700000 + generation * 1000 + index)
    completed = 0
    try:
        while completed < games and not STOP.is_set():
            try:
                _, rows, summary = results.get(timeout=0.5)
            except queue.Empty:
                continue
            completed += 1
            event(role, "selfplay_game", generation=generation, **summary)
            if rows:
                write_bundle(
                    replay_path / f"game-{generation:08d}-{time.time_ns()}.json.gz",
                    rows,
                )
    finally:
        for _ in actors:
            try:
                tasks.put_nowait(None)
            except queue.Full:
                break
        if STOP.is_set():
            for actor in actors:
                if actor.is_alive():
                    actor.terminate()
        for actor in actors:
            actor.join(timeout=5)
            if actor.is_alive():
                actor.kill()
        batcher.close()
        requests.close()
        tasks.close()
        results.close()
        for response in responses:
            response.close()
    return completed


class StageRouter:
    def __init__(self, engines: dict[int, UniChessEngine]):
        self.engines = dict(engines)

    def evaluate_batch(self, boards):
        policy = np.empty((len(boards), 4096), dtype=np.float32)
        promotion = np.empty((len(boards), 4), dtype=np.float32)
        values = np.empty((len(boards), 3), dtype=np.float32)
        groups: dict[int, list[int]] = {}
        for index, board in enumerate(boards):
            groups.setdefault(stage_for_board(board), []).append(index)
        for stage, indices in groups.items():
            outputs = self.engines[stage].evaluate_batch([boards[i] for i in indices])
            policy[indices], promotion[indices], values[indices] = outputs
        return policy, promotion, values

    def close(self):
        for engine in self.engines.values():
            if getattr(engine, "tablebase", None) is not None:
                try:
                    engine.tablebase.close()
                except Exception:
                    pass
        self.engines.clear()


def load_engine(path: Path):
    torch.backends.cudnn.benchmark = True
    engine = UniChessEngine(path, device="cuda", half=True)
    engine.model.to(memory_format=torch.channels_last)
    return engine


def close_engine(engine):
    if engine is None:
        return
    close = getattr(engine, "close", None)
    if callable(close):
        close()
    elif getattr(engine, "tablebase", None) is not None:
        try:
            engine.tablebase.close()
        except Exception:
            pass
    del engine
    gc.collect()
    torch.cuda.empty_cache()


def stage_paths(prefix: str) -> list[Path]:
    return [STATE / "models" / f"{prefix}-stage{index}-champion.pt"
            for index in range(len(STAGE_RANGES))]


def load_router(paths: list[Path]) -> StageRouter:
    return StageRouter({index: load_engine(path) for index, path in enumerate(paths)})


def _train_model(champion: Path, candidate: Path, recovery: Path,
                 rows: list[dict], generation: int, role: str,
                 updates: int, batch_size: int) -> Path | None:
    if not rows or STOP.is_set():
        return None
    seed = torch.load(champion, map_location="cpu", weights_only=False)
    cfg = NetConfig(**seed["cfg"])
    model = UniChessNet(cfg).cuda().to(memory_format=torch.channels_last)
    model.load_state_dict(seed["model"])
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=_float_env("UNICHESS_AZ_LR", 0.01, 1e-5, 0.5),
        momentum=0.9,
        weight_decay=1e-4,
    )
    compiled = compile_rows(rows)
    for tensor in compiled:
        if tensor.device.type == "cpu":
            tensor.pin_memory()
    rng = np.random.default_rng(20260914 + generation)
    started = time.time()
    model.train()
    step = 0
    try:
        while step < updates and not STOP.is_set():
            indices = torch.from_numpy(
                rng.integers(len(rows), size=min(batch_size, len(rows)))
            )
            batch = [
                tensor.index_select(0, indices).to("cuda", non_blocking=True)
                for tensor in compiled
            ]
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = alpha_zero_loss(model, batch)
            loss.backward()
            optimizer.step()
            step += 1
            if step == 1 or step % 10 == 0 or step == updates:
                event(role, "train", generation=generation, step=step,
                      updates=updates, loss=float(loss.detach()),
                      samples=len(rows), samples_per_second=step * batch_size /
                      max(0.01, time.time() - started),
                      **{key: float(value) for key, value in metrics.items()})
            if step % 50 == 0 or step == updates or STOP.is_set():
                save_atomic_checkpoint = {
                    "model": model.state_dict(), "cfg": cfg.__dict__,
                    "optimizer": optimizer.state_dict(), "step": step,
                    "generation": generation, "parent": digest(champion),
                }
                torch.save(save_atomic_checkpoint, recovery)
        if STOP.is_set():
            return None
        torch.save({
            "model": model.state_dict(), "cfg": cfg.__dict__,
            "step": step, "generation": generation,
            "parent": digest(champion),
        }, candidate)
        return candidate
    finally:
        del model, optimizer, compiled
        gc.collect()
        torch.cuda.empty_cache()


def play_arena(candidate, champion, role: str, generation: int,
               games: int, simulations: int, max_plies: int) -> dict:
    wins = draws = losses = completed = 0
    started = time.time()
    for game_index in range(games):
        if STOP.is_set():
            break
        board = chess.Board()
        candidate_white = game_index % 2 == 0
        players = {
            "candidate": candidate,
            "champion": champion,
        }
        trees = {"candidate": None, "champion": None}
        searches = {
            name: MCTS(
                player.evaluate_batch if hasattr(player, "evaluate_batch") else player,
                MCTSConfig(
                    simulations=simulations,
                    batch_size=_int_env("UNICHESS_AZ_MCTS_BATCH", 128, 16, 512),
                    claim_draw=True,
                ),
            )
            for name, player in players.items()
        }
        moves = 0
        while moves < max_plies and not STOP.is_set():
            result, _ = board_outcome(board)
            if result is not None:
                break
            side = "candidate" if (
                (board.turn == chess.WHITE and candidate_white) or
                (board.turn == chess.BLACK and not candidate_white)
            ) else "champion"
            move, root = searches[side].best_move(
                board, temperature=0.0, add_noise=False, root=trees[side]
            )
            for name in trees:
                source = root if name == side else trees[name]
                trees[name] = searches[name].advance_root(source, move)
            board.push(move)
            moves += 1
        result, _ = board_outcome(board)
        if result is None and moves >= max_plies:
            result = "1/2-1/2"
        if result is None:
            continue
        completed += 1
        candidate_score = (
            1.0 if (result == "1-0") == candidate_white and result != "1/2-1/2"
            else 0.0 if result != "1/2-1/2" else 0.5
        )
        if candidate_score == 1.0:
            wins += 1
        elif candidate_score == 0.5:
            draws += 1
        else:
            losses += 1
    score = (wins + 0.5 * draws) / completed if completed else 0.0
    report = {
        "generation": generation, "games": games, "completed_games": completed,
        "wins": wins, "draws": draws, "losses": losses, "score": score,
        "seconds": time.time() - started,
        "accepted": bool(completed == games and score > 0.55 and not STOP.is_set()),
    }
    event(role, "arena", **report)
    return report


def _ensure_champion(role: str) -> Path:
    models = STATE / "models"
    models.mkdir(parents=True, exist_ok=True)
    target = models / f"{role}-az-champion.pt"
    if target.exists():
        return target
    seeds = {
        "small": [
            models / "small-champion.pt",
            ROOT / "runs/stage1/ckpt_00187578.pt",
        ],
        "big": [
            models / "big-champion.pt",
            ROOT / "runs/iteration46_20260909/best.pt",
        ],
    }
    for source in seeds.get(role, []):
        if source.exists():
            shutil.copy2(source, target)
            event(role, "seed", champion=digest(target), source=str(source))
            return target
    raise FileNotFoundError(f"no seed checkpoint for {role}: {target}")


def run_single(role: str, once: bool, smoke: bool) -> None:
    setup()
    lease = lock(f"alphazero-{role}")
    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)
    champion = _ensure_champion(role)
    replay = RecentReplay(
        AZ_REPLAY,
        _int_env("UNICHESS_AZ_REPLAY_GAMES", 5000, 1, 50000),
    )
    state_path = STATE / f"{role}-az-state.json"
    state = read_json(state_path, {"generation": 0})
    while not STOP.is_set():
        generation = int(state.get("generation", 0))
        storage = maintain_storage()
        if not storage["ok"]:
            event(role, "paused_storage", storage=storage)
            if once:
                break
            STOP.wait(30)
            continue
        atomic_json(STATE / f"{role}-az-status.json", {
            "time": time.time(), "phase": "selfplay",
            "generation": generation, "champion": digest(champion),
        })
        evaluator = load_engine(champion)
        try:
            count = _int_env(
                "UNICHESS_AZ_SELFPLAY_GAMES",
                2 if smoke else (32 if role == "small" else 16),
                1, 512,
            )
            actors = _int_env(
                "UNICHESS_AZ_ACTORS",
                1 if smoke else (8 if role == "small" else 16),
                1, 32,
            )
            collect_selfplay(
                evaluator, generation, digest(champion), role, AZ_REPLAY, count,
                _int_env("UNICHESS_AZ_SELFPLAY_SIMS", 32 if smoke else 800, 8, 3200),
                _int_env("UNICHESS_AZ_MAX_PLIES", 32 if smoke else 512, 16, 1024),
                min(actors, count),
                _int_env("UNICHESS_AZ_INFERENCE_BATCH", 256, 64, 2048),
                _float_env("UNICHESS_AZ_INFERENCE_WAIT_MS", 8.0, 1.0, 20.0),
            )
        finally:
            close_engine(evaluator)
        rows = replay.rows()
        if not rows:
            if once:
                break
            STOP.wait(5)
            continue
        atomic_json(STATE / f"{role}-az-status.json", {
            "time": time.time(), "phase": "training",
            "generation": generation, "champion": digest(champion),
            "games": len(replay._files()), "positions": len(rows),
        })
        candidate = STATE / "models" / f"{role}-az-candidate.pt"
        recovery = STATE / "models" / f"{role}-az-recovery.pt"
        trained = _train_model(
            champion, candidate, recovery, rows, generation, role,
            _int_env("UNICHESS_AZ_UPDATES", 2 if smoke else 1000, 1, 100000),
            _int_env("UNICHESS_AZ_BATCH", 128 if smoke else 1024, 16, 4096),
        )
        if trained is None:
            break
        atomic_json(STATE / f"{role}-az-status.json", {
            "time": time.time(), "phase": "arena",
            "generation": generation, "champion": digest(champion),
        })
        old = load_engine(champion)
        new = load_engine(trained)
        try:
            report = play_arena(
                new, old, role, generation,
                _int_env("UNICHESS_AZ_ARENA_GAMES", 2 if smoke else 400, 2, 2000),
                _int_env("UNICHESS_AZ_ARENA_SIMS", 32 if smoke else 800, 8, 3200),
                _int_env("UNICHESS_AZ_ARENA_PLIES", 64 if smoke else 512, 16, 1024),
            )
        finally:
            close_engine(new)
            close_engine(old)
        if report["accepted"]:
            shutil.copy2(champion, STATE / "models" / f"{role}-az-previous.pt")
            os.replace(trained, champion)
            event(role, "promoted", generation=generation, champion=digest(champion))
        state.update(generation=generation + 1)
        atomic_json(state_path, state)
        if once:
            break


def run_layered(once: bool, smoke: bool) -> None:
    setup()
    lease = lock("alphazero-layered")
    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)
    models = STATE / "models"
    paths = stage_paths("layered-az")
    seeds = [models / f"layered-stage{index}-champion.pt"
             for index in range(len(STAGE_RANGES))]
    for target, source in zip(paths, seeds):
        if not target.exists() and source.exists():
            shutil.copy2(source, target)
    if not all(path.exists() for path in paths):
        missing = [str(path) for path in paths if not path.exists()]
        raise FileNotFoundError("layered checkpoints missing: " + ", ".join(missing))
    replay = RecentReplay(
        STATE / "layered-az-replay",
        _int_env("UNICHESS_AZ_REPLAY_GAMES", 5000, 1, 50000),
    )
    state_path = STATE / "layered-az-state.json"
    state = read_json(state_path, {"generation": 0})
    while not STOP.is_set():
        generation = int(state.get("generation", 0))
        current = load_router(paths)
        try:
            collect_selfplay(
                current, generation, digest(paths[0]), "layered",
                STATE / "layered-az-replay",
                _int_env("UNICHESS_AZ_SELFPLAY_GAMES", 2 if smoke else 16, 1, 512),
                _int_env("UNICHESS_AZ_SELFPLAY_SIMS", 32 if smoke else 800, 8, 3200),
                _int_env("UNICHESS_AZ_MAX_PLIES", 32 if smoke else 512, 16, 1024),
                _int_env("UNICHESS_AZ_ACTORS", 1 if smoke else 16, 1, 32),
                _int_env("UNICHESS_AZ_INFERENCE_BATCH", 256, 64, 2048),
                _float_env("UNICHESS_AZ_INFERENCE_WAIT_MS", 8.0, 1.0, 20.0),
            )
        finally:
            current.close()
        rows = replay.rows()
        for stage, champion in enumerate(paths):
            if STOP.is_set():
                break
            stage_rows = [row for row in rows if row.get("stage") == stage]
            if not stage_rows:
                continue
            candidate = models / f"layered-az-stage{stage}-candidate.pt"
            recovery = models / f"layered-az-stage{stage}-recovery.pt"
            trained = _train_model(
                champion, candidate, recovery, stage_rows, generation, "layered",
                _int_env("UNICHESS_AZ_UPDATES", 2 if smoke else 1000, 1, 100000),
                _int_env("UNICHESS_AZ_BATCH", 128 if smoke else 1024, 16, 4096),
            )
            if trained is None:
                break
            candidate_paths = list(paths)
            candidate_paths[stage] = trained
            old_router = load_router(paths)
            new_router = load_router(candidate_paths)
            try:
                report = play_arena(
                    new_router, old_router, "layered", generation,
                    _int_env("UNICHESS_AZ_ARENA_GAMES", 2 if smoke else 400, 2, 2000),
                    _int_env("UNICHESS_AZ_ARENA_SIMS", 32 if smoke else 800, 8, 3200),
                    _int_env("UNICHESS_AZ_ARENA_PLIES", 64 if smoke else 512, 16, 1024),
                )
            finally:
                new_router.close()
                old_router.close()
            if report["accepted"]:
                shutil.copy2(champion, models / f"layered-az-stage{stage}-previous.pt")
                os.replace(trained, champion)
                event("layered", "stage_promoted", generation=generation, stage=stage)
            else:
                trained.unlink(missing_ok=True)
        state.update(generation=generation + 1)
        atomic_json(state_path, state)
        if once:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("small", "big", "layered"), required=True)
    parser.add_argument("--mode", choices=("loop", "combined", "actor", "learner"), default="loop")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.once = True
    if args.role == "layered":
        run_layered(args.once, args.smoke)
    else:
        run_single(args.role, args.once, args.smoke)


if __name__ == "__main__":
    main()

