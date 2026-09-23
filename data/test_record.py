"""A4 验收：96 字节记录的往返一致性。

比对四件事（按严格程度递增）：
  1. 棋子摆放 board_fen 完全一致
  2. 行棋方 / 易位权 / 吃过路兵 / 五十步计数 完全一致
  3. **合法走法集合完全一致** —— 这是最强的检查，位棋盘错一位就会暴露
  4. **encode() 出的 19x8x8 张量逐元素相等** —— 这是训练真正吃进去的东西
"""
from __future__ import annotations

import random
import sys

import chess
import numpy as np

sys.path.insert(0, ".")
from unichess_r.core.encoding import encode
from data.record import (board_to_record, record_policy, record_to_board,
                         record_wdl)

EDGE_FENS = [
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R b Kq - 13 7",
    "8/8/8/KPpP3k/8/8/8/8 w - c6 0 1",
    "8/PPPPPPPP/8/8/8/8/pppppppp/K6k w - - 99 60",
    "4k3/8/8/8/8/8/8/4K3 w - - 50 1",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
]


def check(board: chess.Board) -> None:
    policy = [(796, 0.6), (1000, 0.2), (2000, 0.1), (3000, 0.05), (4000, 0.05)]
    wdl = (0.55, 0.30, 0.15)
    rec = board_to_record(board, policy=policy, wdl=wdl, promo=2, rep=1)
    back = record_to_board(rec)

    assert back.board_fen() == board.board_fen(), \
        f"摆放不一致\n  原: {board.board_fen()}\n  回: {back.board_fen()}"
    assert back.turn == board.turn, "行棋方不一致"
    assert back.castling_rights == board.castling_rights, \
        f"易位权不一致 {back.castling_rights:x} vs {board.castling_rights:x} @ {board.fen()}"
    assert back.ep_square == board.ep_square, f"吃过路兵格不一致 @ {board.fen()}"
    assert back.halfmove_clock == board.halfmove_clock, "五十步计数不一致"

    a = sorted(m.uci() for m in board.legal_moves)
    b = sorted(m.uci() for m in back.legal_moves)
    assert a == b, f"合法走法集合不一致 @ {board.fen()}\n  少: {set(a)-set(b)}\n  多: {set(b)-set(a)}"

    ta = encode(board, repetitions=1)
    tb = encode(back, repetitions=1)
    assert np.array_equal(ta, tb), f"编码张量不一致 @ {board.fen()}"

    # 标签定点化误差应小于 1/65535
    got_p = record_policy(rec)
    assert len(got_p) == 5
    for (m0, p0), (m1, p1) in zip(policy, got_p):
        assert m0 == m1 and abs(p0 - p1) < 2e-5, f"策略标签失真 {p0} vs {p1}"
    for v0, v1 in zip(wdl, record_wdl(rec)):
        assert abs(v0 - v1) < 2e-5, f"WDL 标签失真 {v0} vs {v1}"


def main(target: int = 50_000, seed: int = 20260908) -> int:
    rng = random.Random(seed)
    n = 0
    for fen in EDGE_FENS:
        board = chess.Board(fen)
        for _ in range(40):
            check(board); n += 1
            legal = list(board.legal_moves)
            if not legal: break
            board.push(rng.choice(legal))

    while n < target:
        board = chess.Board()
        while not board.is_game_over(claim_draw=False) and board.fullmove_number < 160:
            check(board); n += 1
            if n >= target: break
            board.push(rng.choice(list(board.legal_moves)))
            if n % 10000 == 0:
                print(f"  ... {n:,} 局面")

    print(f"通过：{n:,} 局面记录往返，摆放/状态/合法走法集合/编码张量 全部一致。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
