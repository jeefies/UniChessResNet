"""易位索引契约：训练标签落在哪个索引，推理端就必须从哪个索引读。

这个测试存在的理由是一个真实事故。data/build_evals.py 原先用
`chess.Move.from_uci` 直接吃 Lichess 评估库的 UCI，而那个库把易位写成
「王吃己车」的 e1h1 / e1a1。要命的是 `mv in board.legal_moves` 对这种形式
返回 **True**（python-chess 的 is_pseudo_legal 会先内部归一化再判定），
于是它一路通过校验，而 move_to_index 拿到的 to_square 还是 h1，标签就写到了
索引 4*64+7 上。推理端（search/mcts.py:priors_from_policy）却是对
board.legal_moves 里的 e1g1 算索引 4*64+6 去查表的——那份易位概率一次也
读不到。已建好的 12G 分片里 2587 个易位标签有 2585 个是错的，占全部策略
质量的 1.99%。等于网络从没学过易位，引擎也从没拿到过易位的先验。

所以这里验两件事：
  1. 构建 -> 记录 -> 解码 之后，易位标签必须落在推理端读的那个索引上
  2. 解码期的修正**不能误伤**：王不在 e1 时，e1 上的车走到 h1 同样是索引
     4*64+7，那是合法的普通着法，绝不能被搬走
"""
from __future__ import annotations

import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.encoding import orient_move
from core.moves import move_to_index
from data.record import board_to_record
from model.dataset import decode_targets


def _label_index(board: chess.Board, move: chess.Move) -> int:
    """按 build_evals.py 的路径造一条记录，返回解码后拿到概率的那个索引。"""
    om = orient_move(move, board.turn)
    rec = np.zeros(1, dtype=board_to_record(board).dtype)
    rec[0] = board_to_record(board, policy=[(move_to_index(om), 1.0)],
                             wdl=(0.0, 1.0, 0.0))
    policy, _, _ = decode_targets(rec)
    nz = np.nonzero(policy[0])[0]
    assert len(nz) == 1, f"期望恰好一个非零索引，得到 {nz}"
    return int(nz[0])


def _engine_index(board: chess.Board, move: chess.Move) -> int:
    """推理端查表用的索引（search/mcts.py:priors_from_policy 的算法）。"""
    return move_to_index(orient_move(move, board.turn))


CASTLING_CASES = [
    # (说明, FEN, 易位走法的 UCI —— 两种写法都要能对上)
    ("白王翼", "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", "e1g1", "e1h1"),
    ("白后翼", "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", "e1c1", "e1a1"),
    ("黑王翼", "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1", "e8g8", "e8h8"),
    ("黑后翼", "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1", "e8c8", "e8a8"),
]

# 王不在 e1/e8，但 e 线底排有车可以横走到 h/a —— 索引与易位撞车，必须不动
NON_CASTLING_CASES = [
    ("白车 e1->h1（王在 a1）", "4k3/8/8/8/8/8/8/K3R3 w - - 0 1", "e1h1"),
    ("白车 e1->a1（王在 g1）", "4k3/8/8/8/8/8/8/4R1K1 w - - 0 1", "e1a1"),
    ("黑车 e8->h8（王在 a8）", "k3r3/8/8/8/8/8/8/4K3 b - - 0 1", "e8h8"),
    ("黑车 e8->a8（王在 g8）", "4r1k1/8/8/8/8/8/8/4K3 b - - 0 1", "e8a8"),
]


def main() -> int:
    ok = True
    print("=== 易位：两种写法都要落到推理端读的索引上 ===")
    for name, fen, king_uci, rook_uci in CASTLING_CASES:
        board = chess.Board(fen)
        canon = board.parse_uci(king_uci)
        assert board.is_castling(canon), f"{name}: {king_uci} 不是易位"
        want = _engine_index(board, canon)

        got_k = _label_index(board, canon)
        # 「王吃己车」写法：Move.from_uci 不归一化，正是事故的来源
        got_r = _label_index(board, chess.Move.from_uci(rook_uci))
        good = got_k == want and got_r == want
        ok &= good
        print(f"  {name:8} 推理端读 {want:5}  王走两格标签 {got_k:5}  "
              f"王吃己车标签 {got_r:5}  {'OK' if good else '失败'}")

    print()
    print("=== 非易位：王不在 e1 时，e 线车横走到 h/a 不能被搬走 ===")
    for name, fen, uci in NON_CASTLING_CASES:
        board = chess.Board(fen)
        mv = board.parse_uci(uci)
        assert not board.is_castling(mv), f"{name}: 居然被判成易位"
        want = _engine_index(board, mv)
        got = _label_index(board, mv)
        good = got == want
        ok &= good
        print(f"  {name:22} 推理端读 {want:5}  标签 {got:5}  "
              f"{'OK' if good else '失败（被误搬）'}")

    print()
    print("易位索引契约:", "通过" if ok else "未通过")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
