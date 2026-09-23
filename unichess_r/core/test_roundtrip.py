"""A2 硬性验收：局面/走法编解码往返一致性。

覆盖三件事：
  1. 每个合法走法 -> 索引 -> 还原，必须与原走法完全相等（含升变、易位、吃过路兵）
  2. 同一局面内不存在两个不同的合法走法映射到同一 (索引, 升变索引)——否则策略头无法区分
  3. 编码张量的形状/取值范围合法，且黑方局面经 orient 后与其镜像的白方局面编码完全一致

编码错了后面全白练，所以这个测试必须 100% 通过。
"""
from __future__ import annotations

import random
import sys

import chess
import numpy as np

sys.path.insert(0, ".")
from unichess_r.core.encoding import (INPUT_SHAPE, encode, orient, orient_move,
                           unorient_move)
from unichess_r.core.moves import (index_to_move, move_to_index, move_to_promo_index)

# 升变/易位/吃过路兵密集的局面，弥补随机对局覆盖不到的角落
EDGE_FENS = [
    "8/PPPPPPPP/8/8/8/8/pppppppp/K6k w - - 0 1",        # 白方八路升变
    "8/PPPPPPPP/8/8/8/8/pppppppp/K6k b - - 0 1",        # 黑方八路升变
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",             # 双方双向易位
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    "8/8/8/KPpP3k/8/8/8/8 w - c6 0 1",                  # 吃过路兵
    "8/8/8/8/kpPp4/8/8/K7 b - c3 0 1",
    "4k3/8/8/8/8/8/6PP/4K2R w K - 0 1",
    "8/2P5/8/8/8/8/8/K1k5 w - - 0 1",                   # 单兵升变
]


def check_position(board: chess.Board) -> None:
    turn = board.turn
    seen: dict[tuple[int, int | None], chess.Move] = {}

    for move in board.legal_moves:
        om = orient_move(move, turn)
        idx = move_to_index(om)
        promo = move_to_promo_index(om)

        # 2) 碰撞检测
        key = (idx, promo)
        if key in seen:
            raise AssertionError(
                f"索引碰撞 @ {board.fen()}: {seen[key].uci()} 与 {move.uci()} "
                f"同为 (idx={idx}, promo={promo})")
        seen[key] = move

        # 1) 往返
        back = unorient_move(index_to_move(idx, promo), turn)
        if back != move:
            raise AssertionError(
                f"往返失配 @ {board.fen()}: {move.uci()} -> idx={idx},promo={promo} "
                f"-> {back.uci()}")

    # 3) 编码检查
    t = encode(board, repetitions=0)
    assert t.shape == INPUT_SHAPE, f"形状错误 {t.shape}"
    assert t.dtype == np.float32, f"dtype 错误 {t.dtype}"
    assert np.isfinite(t).all(), "存在 NaN/Inf"
    assert (t >= 0.0).all() and (t <= 1.0).all(), "取值超出 [0,1]"

    # 黑方局面的编码，应当等于「其镜像局面轮白方走」的编码
    if turn == chess.BLACK:
        t_mirror = encode(orient(board), repetitions=0)
        assert np.array_equal(t, t_mirror), f"视角不对称 @ {board.fen()}"


def main(target: int = 100_000, seed: int = 20260908) -> int:
    rng = random.Random(seed)
    checked = 0
    moves_checked = 0

    # 先过一遍边角局面，并从每个局面出发随机走若干步
    for fen in EDGE_FENS:
        board = chess.Board(fen)
        for _ in range(60):
            check_position(board)
            checked += 1
            moves_checked += board.legal_moves.count()
            legal = list(board.legal_moves)
            if not legal or board.is_game_over():
                break
            board.push(rng.choice(legal))

    # 随机自对弈铺量
    games = 0
    while checked < target:
        board = chess.Board()
        games += 1
        while not board.is_game_over(claim_draw=False) and board.fullmove_number < 160:
            check_position(board)
            checked += 1
            moves_checked += board.legal_moves.count()
            if checked >= target:
                break
            board.push(rng.choice(list(board.legal_moves)))
            if checked % 20000 == 0:
                print(f"  ... {checked:,} 局面")

    print(f"通过：{checked:,} 局面 / {moves_checked:,} 次走法往返 / {games} 盘随机棋")
    print("往返一致性 100%，无索引碰撞，编码取值合法，黑白视角对称。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
