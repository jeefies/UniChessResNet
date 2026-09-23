"""测 CPU 上的推理与 MCTS 速度，评估「评测完全不占 GPU」是否可行。

PRO 6000 那台机器有 208 核，如果 CPU 够快，评测就可以完全避开 GPU，
不必去争论「对别人的任务影响多大」这种测不准的问题。
"""
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unichess_r.engine.engine import UniChessEngine
from unichess_r.search.mcts import MCTS, MCTSConfig

ckpt = sys.argv[1]
threads = int(sys.argv[2]) if len(sys.argv) > 2 else 16
torch.set_num_threads(threads)
print(f"torch 线程数: {torch.get_num_threads()}")

eng = UniChessEngine(ckpt, device="cpu", syzygy_path="data/raw/syzygy345")

# 随机局面
boards = []
rng = np.random.default_rng(0)
for _ in range(512):
    b = chess.Board()
    for _ in range(rng.integers(4, 30)):
        ms = list(b.legal_moves)
        if not ms:
            break
        b.push(ms[rng.integers(len(ms))])
    boards.append(b)

print("\n=== 批量推理 ===")
for n in (64, 128, 256):
    sub = boards[:n]
    eng.evaluate_batch(sub)                      # 预热
    t0 = time.perf_counter()
    for _ in range(5):
        eng.evaluate_batch(sub)
    dt = (time.perf_counter() - t0) / 5
    print(f"  批 {n:4}: {dt*1000:7.1f} ms  ->  {n/dt:>8,.0f} 局面/秒")

print("\n=== MCTS ===")
mcts = MCTS(eng.evaluate_batch, MCTSConfig(simulations=800, batch_size=64),
            tablebase=eng.tablebase)
mcts.search(chess.Board(), simulations=128)      # 预热
for sims in (400, 800):
    t0 = time.perf_counter()
    root = mcts.search(chess.Board(), simulations=sims)
    dt = time.perf_counter() - t0
    print(f"  {sims:4} 次模拟: {dt:5.2f}s  ->  {sims/dt:>7,.0f} nodes/s  "
          f"(每步耗时 {dt:.2f}s)")
