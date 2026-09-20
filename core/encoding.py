"""局面 -> 19x8x8 神经网络输入张量。

约定：**永远从当前行棋方的视角编码**。若轮到黑方，先把棋盘上下翻转并交换
颜色（chess.Board.mirror），于是网络只需要学「我方 vs 对方」，不必区分黑白，
等效于把数据量翻倍。因此不需要「行棋方」平面。

平面布局（共 19 层）：
     0- 5  我方 兵马象车后王
     6-11  对方 兵马象车后王
    12     我方 王翼易位权      （整层填充 0/1）
    13     我方 后翼易位权
    14     对方 王翼易位权
    15     对方 后翼易位权
    16     吃过路兵目标格        （单格置 1）
    17     五十步计数 / 100      （整层填充）
    18     重复次数 / 2          （整层填充）

重复次数无法从单个局面推出，需由调用方传入（训练数据的定长记录里存了这一字节）。
"""
from __future__ import annotations

import chess
import numpy as np

NUM_PLANES = 19
BOARD_SIZE = 8
INPUT_SHAPE = (NUM_PLANES, BOARD_SIZE, BOARD_SIZE)

# 棋子类型 -> 平面偏移。顺序固定，改动会让已训练权重错位。
_PIECE_ORDER = (chess.PAWN, chess.KNIGHT, chess.BISHOP,
                chess.ROOK, chess.QUEEN, chess.KING)
_PIECE_PLANE = {p: i for i, p in enumerate(_PIECE_ORDER)}


def orient(board: chess.Board) -> chess.Board:
    """把棋盘转成「当前行棋方在下方」的白方视角。

    轮到白方时返回原棋盘本身（不复制，调用方勿修改）；轮到黑方时返回镜像副本。
    """
    return board if board.turn == chess.WHITE else board.mirror()


def orient_square(square: int, turn: bool) -> int:
    """把格子坐标转到已 orient 的棋盘坐标系下。"""
    return square if turn == chess.WHITE else chess.square_mirror(square)


def orient_move(move: chess.Move, turn: bool) -> chess.Move:
    """把走法转到已 orient 的棋盘坐标系下。"""
    if turn == chess.WHITE:
        return move
    return chess.Move(chess.square_mirror(move.from_square),
                      chess.square_mirror(move.to_square),
                      promotion=move.promotion)


def unorient_move(move: chess.Move, turn: bool) -> chess.Move:
    """orient_move 的逆运算（镜像是对合的，所以实现相同）。"""
    return orient_move(move, turn)


def encode(board: chess.Board, repetitions: int | None = None) -> np.ndarray:
    """局面 -> float32 的 (19, 8, 8) 张量。

    board 可以是任意一方行棋的局面，内部会自动 orient。
    repetitions 为 None 时尝试从 board 的历史推断（需要 board 带着走子历史）。
    """
    if repetitions is None:
        repetitions = 2 if board.is_repetition(3) else (1 if board.is_repetition(2) else 0)

    turn = board.turn
    b = orient(board)
    planes = np.zeros(INPUT_SHAPE, dtype=np.float32)

    # 0-11: 棋子。b 已经是白方视角，白 = 我方。
    for square, piece in b.piece_map().items():
        plane = _PIECE_PLANE[piece.piece_type]
        if piece.color == chess.BLACK:
            plane += 6
        row, col = divmod(square, 8)
        planes[plane, row, col] = 1.0

    # 12-15: 易位权
    if b.has_kingside_castling_rights(chess.WHITE):
        planes[12] = 1.0
    if b.has_queenside_castling_rights(chess.WHITE):
        planes[13] = 1.0
    if b.has_kingside_castling_rights(chess.BLACK):
        planes[14] = 1.0
    if b.has_queenside_castling_rights(chess.BLACK):
        planes[15] = 1.0

    # 16: 吃过路兵目标格
    if b.ep_square is not None:
        row, col = divmod(b.ep_square, 8)
        planes[16, row, col] = 1.0

    # 17: 五十步计数
    planes[17] = min(b.halfmove_clock, 100) / 100.0

    # 18: 重复次数
    planes[18] = min(repetitions, 2) / 2.0

    return planes
