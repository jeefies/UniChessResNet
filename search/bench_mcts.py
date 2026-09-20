"""MCTS 吞吐基准：nodes/s，并把 GPU 推理和 Python 树操作分开计时。

这是方案要求的 Stage 2 基线，有两个用途：
  1. 判断纯 Python 的 MCTS 到底够不够用
  2. 给 Stage 3 的 C++ 移植一个对照基准——只有知道 Python 树操作占多少时间，
     才能估出移植能拿到多少收益。如果推理占 90%，移植就没意义。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.engine import UniChessEngine
from search.mcts import MCTS, MCTSConfig

POSITIONS = [
    ("起始局面", chess.STARTING_FEN),
    ("开局后",   "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4"),
    ("中局",     "r2q1rk1/pp2bppp/2n1bn2/2pp4/3P4/2P1PN2/PP1NBPPP/R1BQ1RK1 w - - 0 10"),
    ("残局",     "8/5pk1/6p1/7p/7P/5PP1/5K2/3r4 w - - 0 40"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--syzygy", default="data/raw/syzygy345")
    ap.add_argument("--sims", type=int, default=1600)
    ap.add_argument("--batches", type=int, default=0,
                    help="要测的 MCTS batch_size，逗号分隔；缺省测一组")
    a = ap.parse_args()

    eng = UniChessEngine(a.ckpt, device=a.device, syzygy_path=a.syzygy)

    # 包一层评估器，单独统计 GPU 推理耗时和被评估的局面数
    stats = {"eval_s": 0.0, "positions": 0, "calls": 0}

    def timed_eval(boards):
        t0 = time.perf_counter()
        out = eng.evaluate_batch(boards)
        stats["eval_s"] += time.perf_counter() - t0
        stats["positions"] += len(boards)
        stats["calls"] += 1
        return out

    print(f"网络 {eng.cfg.name}，设备 {a.device}，每步 {a.sims} 次模拟\n")
    print(f"{'批大小':>6} {'局面':>10} {'总耗时':>8} {'nodes/s':>9} "
          f"{'推理占比':>8} {'平均批':>7}")

    for bs in (64, 128, 256, 512):
        stats.update(eval_s=0.0, positions=0, calls=0)
        mcts = MCTS(timed_eval, MCTSConfig(simulations=a.sims, batch_size=bs),
                    tablebase=eng.tablebase)
        # 预热
        mcts.search(chess.Board(), simulations=256)
        stats.update(eval_s=0.0, positions=0, calls=0)

        t0 = time.perf_counter()
        total_nodes = 0
        for _, fen in POSITIONS:
            root = mcts.search(chess.Board(fen))
            total_nodes += int(root.N.sum())
        dt = time.perf_counter() - t0

        nps = total_nodes / dt
        frac = stats["eval_s"] / dt
        avg_batch = stats["positions"] / max(stats["calls"], 1)
        print(f"{bs:>6} {total_nodes:>10,} {dt:>7.2f}s {nps:>9,.0f} "
              f"{100*frac:>7.1f}% {avg_batch:>7.0f}")

    print()
    print("推理占比越高，说明瓶颈越偏 GPU，移植 C++ 的收益越小；")
    print("占比低则说明时间花在 Python 树操作上，移植收益大。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
