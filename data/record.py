"""训练样本的 96 字节定长二进制记录。

不存 FEN（解析太慢），不存展开的浮点平面（2 亿局面会撑爆磁盘）。
存位棋盘 + 标签，训练时内存映射读取，在 GPU 上现场解码成 19x8x8 平面。

字段（合计正好 96 字节）：
    pawns/knights/bishops/rooks/queens/kings  6 x u64 = 48   棋子类型占位
    occ_white / occ_black                     2 x u64 = 16   颜色占位
    castling  u8   低 4 位 = K Q k q
    ep        u8   吃过路兵目标格 0-63，255 = 无
    halfmove  u8   五十步计数，255 封顶（75 步规则 150 半步强制和棋，故无损）
    side      u8   0 = 白方行棋，1 = 黑方行棋
    rep       u8   重复次数 0/1/2
    promo     u8   最佳走法的升变索引 0-3，255 = 非升变
    policy_move  5 x u16 = 10   Stockfish MultiPV 前 5 的走法索引（from*64+to）
    policy_prob  5 x u16 = 10   对应概率，按 65535 定点化
    wdl          3 x u16 =  6   胜/和/负概率，按 65535 定点化
                                 ---
                                  96
"""
from __future__ import annotations

import chess
import numpy as np

RECORD_DTYPE = np.dtype([
    ("pawns",       "<u8"), ("knights", "<u8"), ("bishops", "<u8"),
    ("rooks",       "<u8"), ("queens",  "<u8"), ("kings",   "<u8"),
    ("occ_white",   "<u8"), ("occ_black", "<u8"),
    ("castling",    "u1"), ("ep",   "u1"), ("halfmove", "u1"),
    ("side",        "u1"), ("rep",  "u1"), ("promo",    "u1"),
    ("policy_move", "<u2", (5,)),
    ("policy_prob", "<u2", (5,)),
    ("wdl",         "<u2", (3,)),
])
assert RECORD_DTYPE.itemsize == 96, f"记录应为 96 字节，实为 {RECORD_DTYPE.itemsize}"

NO_EP = 255
NO_PROMO = 255
_SCALE = 65535


def board_to_record(board: chess.Board, *, policy: list[tuple[int, float]] | None = None,
                    wdl: tuple[float, float, float] = (0.0, 1.0, 0.0),
                    promo: int | None = None, rep: int = 0) -> np.void:
    """把局面 + 标签打成一条记录。policy 为 [(走法索引, 概率), ...] 最多 5 条。"""
    rec = np.zeros(1, dtype=RECORD_DTYPE)[0]
    rec["pawns"] = board.pawns
    rec["knights"] = board.knights
    rec["bishops"] = board.bishops
    rec["rooks"] = board.rooks
    rec["queens"] = board.queens
    rec["kings"] = board.kings
    rec["occ_white"] = board.occupied_co[chess.WHITE]
    rec["occ_black"] = board.occupied_co[chess.BLACK]

    castling = 0
    if board.has_kingside_castling_rights(chess.WHITE):  castling |= 1
    if board.has_queenside_castling_rights(chess.WHITE): castling |= 2
    if board.has_kingside_castling_rights(chess.BLACK):  castling |= 4
    if board.has_queenside_castling_rights(chess.BLACK): castling |= 8
    rec["castling"] = castling

    rec["ep"] = NO_EP if board.ep_square is None else board.ep_square
    rec["halfmove"] = min(board.halfmove_clock, 255)
    rec["side"] = 0 if board.turn == chess.WHITE else 1
    rec["rep"] = min(rep, 2)
    rec["promo"] = NO_PROMO if promo is None else promo

    if policy:
        for i, (mv, pr) in enumerate(policy[:5]):
            rec["policy_move"][i] = mv
            rec["policy_prob"][i] = int(round(max(0.0, min(1.0, pr)) * _SCALE))
    for i, v in enumerate(wdl):
        rec["wdl"][i] = int(round(max(0.0, min(1.0, v)) * _SCALE))
    return rec


def record_to_board(rec: np.void) -> chess.Board:
    """记录 -> chess.Board。直接写位棋盘，比 FEN 解析快一个数量级。"""
    board = chess.Board(None)  # 空棋盘，不摆子
    board.pawns = int(rec["pawns"])
    board.knights = int(rec["knights"])
    board.bishops = int(rec["bishops"])
    board.rooks = int(rec["rooks"])
    board.queens = int(rec["queens"])
    board.kings = int(rec["kings"])
    board.occupied_co[chess.WHITE] = int(rec["occ_white"])
    board.occupied_co[chess.BLACK] = int(rec["occ_black"])
    board.occupied = int(rec["occ_white"]) | int(rec["occ_black"])
    board.promoted = 0

    c = int(rec["castling"])
    mask = 0
    if c & 1: mask |= chess.BB_H1
    if c & 2: mask |= chess.BB_A1
    if c & 4: mask |= chess.BB_H8
    if c & 8: mask |= chess.BB_A8
    board.castling_rights = mask

    ep = int(rec["ep"])
    board.ep_square = None if ep == NO_EP else ep
    board.halfmove_clock = int(rec["halfmove"])
    board.turn = chess.WHITE if int(rec["side"]) == 0 else chess.BLACK
    board.fullmove_number = 1
    return board


def record_policy(rec: np.void) -> list[tuple[int, float]]:
    """取出策略软标签，过滤掉概率为 0 的空位。"""
    out = []
    for mv, pr in zip(rec["policy_move"], rec["policy_prob"]):
        if pr:
            out.append((int(mv), float(pr) / _SCALE))
    return out


def record_wdl(rec: np.void) -> tuple[float, float, float]:
    return tuple(float(v) / _SCALE for v in rec["wdl"])


def open_shard(path, mode: str = "r") -> np.memmap:
    """内存映射打开一个分片文件。"""
    return np.memmap(path, dtype=RECORD_DTYPE, mode=mode)
