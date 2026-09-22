"""A3：用 Stockfish 给局面打策略软标签（MultiPV=5）+ WDL 价值标签。

为什么需要它：evals 库虽然有 3.9 亿个局面的评估，但只给单条 PV，
用作策略标签就是 one-hot，信号太弱。MultiPV=5 能把「前 5 个候选走法各有多好」
变成一个软分布，这是 DeepMind searchless chess 的做法。

代价是每个局面贵 5 倍，所以分工：
    evals 库  -> 海量 value 标签
    本机 SF   -> 较少但高质量的 policy 标签

多进程，每个 worker 独占一个 Stockfish 进程。可断点续跑：
输出按分片写，重启时跳过已完成的分片。
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.encoding import orient_move
from core.moves import move_to_index, move_to_promo_index
from data.record import RECORD_DTYPE, board_to_record

# Lichess 的 centipawn -> 胜率换算（logistic），用于把 SF 评分变成 WDL
_CP_K = 0.00368208


def cp_to_win_prob(cp: float) -> float:
    """厘兵评分 -> 行棋方胜率 [0,1]。"""
    return 1.0 / (1.0 + math.exp(-_CP_K * cp))


def score_to_wdl(cp: float | None, mate: int | None) -> tuple[float, float, float]:
    """(cp, mate) -> (胜, 和, 负) 三分布。

    和棋概率随分差增大而衰减：均势时和棋概率最高，这符合国象高水平的实际分布。
    单标量 value head 在国象会塌成全 0，所以必须用三头。
    """
    if mate is not None:
        return (1.0, 0.0, 0.0) if mate > 0 else (0.0, 0.0, 1.0)
    w_raw = cp_to_win_prob(cp)
    # 和棋权重：|cp| 越小越大，峰值 0.5 左右
    draw = 0.55 * math.exp(-((cp / 220.0) ** 2))
    win = (1.0 - draw) * w_raw
    loss = (1.0 - draw) * (1.0 - w_raw)
    total = win + draw + loss
    return (win / total, draw / total, loss / total)


def policy_from_multipv(board: chess.Board, infos: list, temperature: float = 90.0
                        ) -> tuple[list[tuple[int, float]], int | None]:
    """MultiPV 结果 -> 策略软分布 + 最佳走法的升变索引。

    温度单位是厘兵：temperature=90 表示差 90 厘兵的走法权重约为 1/e。
    """
    cands: list[tuple[chess.Move, float]] = []
    for info in infos:
        pv = info.get("pv")
        if not pv:
            continue
        score = info["score"].pov(board.turn)
        mate = score.mate()
        cp = 30000.0 if (mate is not None and mate > 0) else \
             -30000.0 if mate is not None else float(score.score())
        cands.append((pv[0], cp))
    if not cands:
        return [], None

    best_cp = max(c for _, c in cands)
    weights = [math.exp((cp - best_cp) / temperature) for _, cp in cands]
    total = sum(weights)

    policy = []
    for (mv, _), w in zip(cands, weights):
        om = orient_move(mv, board.turn)
        policy.append((move_to_index(om), w / total))
    best_move = cands[0][0]
    promo = move_to_promo_index(orient_move(best_move, board.turn))
    return policy, promo


def _worker(args) -> tuple[int, int]:
    shard_id, fens, out_path, sf_path, nodes, multipv, threads, hash_mb = args
    if Path(out_path).exists():
        return shard_id, -1  # 已完成，跳过

    engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    engine.configure({"Threads": threads, "Hash": hash_mb})
    limit = chess.engine.Limit(nodes=nodes)

    recs = np.zeros(len(fens), dtype=RECORD_DTYPE)
    n = 0
    try:
        for fen in fens:
            try:
                board = chess.Board(fen)
                if board.is_game_over(claim_draw=False) or not board.is_valid():
                    continue
                infos = engine.analyse(board, limit, multipv=multipv)
                policy, promo = policy_from_multipv(board, infos)
                if not policy:
                    continue
                score = infos[0]["score"].pov(board.turn)
                wdl = score_to_wdl(
                    None if score.mate() is not None else float(score.score()),
                    score.mate())
                recs[n] = board_to_record(board, policy=policy, wdl=wdl, promo=promo)
                n += 1
            except Exception:
                continue
    finally:
        engine.quit()

    tmp = str(out_path) + ".tmp"
    recs[:n].tofile(tmp)
    os.replace(tmp, out_path)
    return shard_id, n


def annotate(fens, out_dir: Path, sf_path: str, *, workers: int = 14,
             shard_size: int = 20000, nodes: int = 100_000, multipv: int = 5,
             threads: int = 1, hash_mb: int = 256) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks, buf, sid = [], [], 0
    for fen in fens:
        buf.append(fen)
        if len(buf) >= shard_size:
            tasks.append((sid, buf, out_dir / f"anno_{sid:06d}.bin",
                          sf_path, nodes, multipv, threads, hash_mb))
            buf, sid = [], sid + 1
    if buf:
        tasks.append((sid, buf, out_dir / f"anno_{sid:06d}.bin",
                      sf_path, nodes, multipv, threads, hash_mb))

    todo = [t for t in tasks if not t[2].exists()]
    print(f"共 {len(tasks)} 个分片，待处理 {len(todo)}，{workers} 进程，"
          f"每局面 {nodes:,} nodes, MultiPV={multipv}")

    done_n, t0 = 0, time.time()
    with mp.Pool(workers) as pool:
        for i, (shard_id, n) in enumerate(pool.imap_unordered(_worker, todo), 1):
            if n >= 0:
                done_n += n
            el = time.time() - t0
            rate = done_n / el if el > 0 else 0
            print(f"[{i}/{len(todo)}] shard {shard_id}: {n} 条 | "
                  f"累计 {done_n:,} | {rate:.1f} 局面/秒 | "
                  f"已用 {el/60:.1f} 分", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Stockfish MultiPV 标注")
    ap.add_argument("--fens", required=True, help="每行一个 FEN 的文本文件")
    ap.add_argument("--out", required=True, help="输出分片目录")
    ap.add_argument("--stockfish", default="tools/stockfish")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--nodes", type=int, default=100_000)
    ap.add_argument("--multipv", type=int, default=5)
    ap.add_argument("--shard-size", type=int, default=20000)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个局面，0 = 全部")
    a = ap.parse_args()

    def gen():
        with open(a.fens) as f:
            for i, line in enumerate(f):
                if a.limit and i >= a.limit:
                    break
                line = line.strip()
                if line:
                    yield line

    annotate(gen(), Path(a.out), a.stockfish, workers=a.workers,
             shard_size=a.shard_size, nodes=a.nodes, multipv=a.multipv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
