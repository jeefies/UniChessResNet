"""走法 <-> 索引 的双向映射。

策略头输出 (64, 8, 8) 的张量，展平后索引为 `from_square * 64 + to_square`：
    plane = from_square, 空间位置 (row, col) = to_square
共 4096 维。升变棋子另由一个 4 维的头给出（后/车/象/马），
因为 (from, to) 已经唯一确定了升变走法的落点，只差升变成什么。

所有索引都在「当前行棋方视角」下——调用方需保证棋盘已经 mirror 过
（见 core.encoding.orient）。
"""
from __future__ import annotations

import chess

POLICY_SIZE = 64 * 64  # 4096
PROMO_SIZE = 4

# 升变棋子 <-> 索引。顺序固定，不要改动，否则已训练的权重会错位。
PROMO_PIECES = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
PROMO_TO_IDX = {p: i for i, p in enumerate(PROMO_PIECES)}


def move_to_index(move: chess.Move) -> int:
    """走法 -> [0, 4096) 的策略索引。忽略升变棋子（由 promo 头单独处理）。"""
    return move.from_square * 64 + move.to_square


def move_to_promo_index(move: chess.Move) -> int | None:
    """升变走法 -> [0, 4) 的升变索引；非升变走法返回 None。"""
    if move.promotion is None:
        return None
    return PROMO_TO_IDX[move.promotion]


def index_to_move(index: int, promo_index: int | None = None) -> chess.Move:
    """策略索引 -> 走法。promo_index 为 None 时不带升变。

    注意：本函数不校验合法性，调用方应当在合法走法集合内使用。
    """
    from_square, to_square = divmod(index, 64)
    promotion = None if promo_index is None else PROMO_PIECES[promo_index]
    return chess.Move(from_square, to_square, promotion=promotion)


def legal_move_indices(board: chess.Board) -> list[int]:
    """当前局面所有合法走法的策略索引（去重后）。用于 mask。"""
    return sorted({move_to_index(m) for m in board.legal_moves})


def legal_mask(board: chess.Board):
    """返回长度 4096 的 bool 掩码，合法走法处为 True。"""
    import numpy as np

    mask = np.zeros(POLICY_SIZE, dtype=bool)
    for m in board.legal_moves:
        mask[move_to_index(m)] = True
    return mask
