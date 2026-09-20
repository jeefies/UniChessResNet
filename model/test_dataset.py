"""验收：dataset 的 numpy 快路径 == core.encoding 的 python-chess 参考实现。

训练时为了速度用纯位运算解码（decode_batch），推理时用 python-chess 走 encode()。
两套实现必须逐元素相等，否则「训练时看到的棋盘」和「下棋时看到的棋盘」不是一回事，
网络会以一种极难察觉的方式变弱。
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.encoding import encode
from data.record import RECORD_DTYPE, board_to_record
from model.dataset import decode_batch, decode_targets, piece_count_bucket

EDGE_FENS = [
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R b Kq - 37 9",
    "8/8/8/KPpP3k/8/8/8/8 w - c6 0 1",
    "8/8/8/8/kpPp4/8/8/K7 b - c3 0 1",
    "8/PPPPPPPP/8/8/8/8/pppppppp/K6k w - - 99 60",
    "8/PPPPPPPP/8/8/8/8/pppppppp/K6k b - - 12 60",
    "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
]


def main(target: int = 30_000, seed: int = 20260908) -> int:
    rng = random.Random(seed)
    boards: list[tuple[chess.Board, int]] = []

    for fen in EDGE_FENS:
        b = chess.Board(fen)
        for _ in range(30):
            boards.append((b.copy(stack=False), rng.randint(0, 2)))
            legal = list(b.legal_moves)
            if not legal:
                break
            b.push(rng.choice(legal))

    while len(boards) < target:
        b = chess.Board()
        while not b.is_game_over(claim_draw=False) and b.fullmove_number < 160:
            boards.append((b.copy(stack=False), rng.randint(0, 2)))
            if len(boards) >= target:
                break
            b.push(rng.choice(list(b.legal_moves)))

    print(f"比对 {len(boards):,} 个局面 ...")

    # 参考实现
    ref = np.stack([encode(b, repetitions=r) for b, r in boards])

    # 快路径
    recs = np.zeros(len(boards), dtype=RECORD_DTYPE)
    for i, (b, r) in enumerate(boards):
        recs[i] = board_to_record(
            b, policy=[(796, 0.5), (1000, 0.5)], wdl=(0.4, 0.4, 0.2), rep=r)
    fast = decode_batch(recs)

    assert fast.shape == ref.shape, f"形状不一致 {fast.shape} vs {ref.shape}"
    if not np.array_equal(fast, ref):
        diff = np.argwhere(fast != ref)
        i, pl, r_, c_ = diff[0]
        raise AssertionError(
            f"共 {len(diff)} 处不一致。首例：样本 {i} 平面 {pl} 格 ({r_},{c_}) "
            f"fast={fast[i,pl,r_,c_]} ref={ref[i,pl,r_,c_]}\n"
            f"  FEN: {boards[i][0].fen()}\n"
            f"  涉及平面: {sorted(set(int(d[1]) for d in diff))}")
    print("✓ decode_batch 与 encode 逐元素完全一致")

    # 标签解码
    pol, promo, wdl = decode_targets(recs)
    assert np.allclose(pol.sum(axis=1), 1.0, atol=1e-4), "策略分布未归一化"
    assert np.allclose(wdl.sum(axis=1), 1.0, atol=1e-4), "WDL 未归一化"
    assert (promo == -100).all(), "未设升变的样本应为 ignore_index"
    print("✓ 标签解码正确（策略/WDL 均归一化，升变缺省为 ignore_index）")

    # 分桶
    for nb in (4, 8):
        bk = piece_count_bucket(recs, nb)
        assert bk.min() >= 0 and bk.max() < nb, f"桶索引越界 {bk.min()}..{bk.max()}"
    print("✓ 子力数分桶索引合法")
    print(f"\n通过：{len(boards):,} 局面，快路径与参考实现零差异。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
