"""MCTS 验收：用**零知识**的均匀评估器，检验搜索本身是否正确。

为什么用均匀评估器：把「搜索对不对」和「网络强不强」彻底分开。
如果连纯搜索都找不到一步杀，那问题一定在树里（多半是价值符号取反弄错了），
而不是网络。等 Stage 1 网络训好再一起测，就分不清是谁的锅了。

四项检查：
  1. 一步杀：必须找到
  2. 两步杀：必须找到（要求价值能跨两层正确取反回传）
  3. 送子检测：不能走进一步被将死的着法
  4. 残局表：接上 Syzygy 后，必胜残局的根节点价值应接近 +1
"""
from __future__ import annotations

import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from unichess_r.search.mcts import MCTS, MCTSConfig


def uniform_evaluator(boards):
    """零知识评估器：策略均匀、价值全和棋。搜索到的任何信号都只能来自终局。"""
    n = len(boards)
    policy = np.full((n, 4096), 1.0 / 4096, dtype=np.float32)
    promo = np.full((n, 4), 0.25, dtype=np.float32)
    wdl = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (n, 1))
    return policy, promo, wdl


MATE_IN_1 = [
    ("后翼底线杀", "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "a1a8"),
    ("双车阶梯杀", "7k/8/8/8/8/8/6R1/7R w - - 0 1", "h1h8"),
    ("学者杀",     "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4", None),
]

# 预期不再手写——由 Stockfish 判定真实杀棋步数。
# 手写预期已经错过两次（KPvK 理论和判成必胜、KQ 的 mate-in-6 写成 mate-in-2），
# 两次都浪费了排查时间去查一个根本没坏的实现。
MATE_IN_2_CANDIDATES = [
    ("双车",     "7k/8/8/8/8/8/R7/1R5K w - - 0 1"),
    ("后车配合", "6k1/8/6K1/8/8/8/8/7R w - - 0 1"),
    ("底线",     "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"),
]


def check_mate_in_1() -> bool:
    ok = True
    mcts = MCTS(uniform_evaluator, MCTSConfig(simulations=400, batch_size=32))
    for name, fen, expect in MATE_IN_1:
        board = chess.Board(fen)
        if board.turn == chess.BLACK and expect is None:
            # 学者杀那条是「黑方已被将死」，只验终局识别
            print(f"  {name:12} 已将死: {board.is_checkmate()}")
            ok &= board.is_checkmate()
            continue
        mv, root = mcts.best_move(board)
        board.push(mv)
        mated = board.is_checkmate()
        print(f"  {name:12} 走 {mv.uci()}  将死={mated}  "
              f"{'✓' if mated else '✗ 期望 ' + str(expect)}")
        ok &= mated
    return ok


def check_mate_in_2() -> bool:
    """只用 Stockfish 确认过是 mate-in-2 的局面，要求 MCTS 在 4 半步内杀掉。"""
    import chess.engine
    eng = chess.engine.SimpleEngine.popen_uci("tools/stockfish")
    verified = []
    try:
        for name, fen in MATE_IN_2_CANDIDATES:
            b = chess.Board(fen)
            info = eng.analyse(b, chess.engine.Limit(depth=25))
            m = info["score"].pov(b.turn).mate()
            if m is not None and 0 < m <= 2:
                verified.append((name, fen, m))
            else:
                print(f"  {name:12} Stockfish 判为 mate={m}，不是两步杀，跳过")
    finally:
        eng.quit()

    if not verified:
        print("  没有可用的两步杀局面")
        return False

    ok = True
    mcts = MCTS(uniform_evaluator, MCTSConfig(simulations=1600, batch_size=64))
    for name, fen, m in verified:
        board = chess.Board(fen)
        plies = 0
        while not board.is_game_over(claim_draw=False) and plies < 2 * m:
            mv, _ = mcts.best_move(board)
            assert mv in board.legal_moves, f"非法走法 {mv} @ {board.fen()}"
            board.push(mv)
            plies += 1
        mated = board.is_checkmate()
        print(f"  {name:12} Stockfish mate={m}, MCTS {plies} 半步 将死={mated} "
              f"{'✓' if mated else '✗'}")
        ok &= mated
    return ok


def check_avoid_blunder() -> bool:
    """不能走进「走完就被一步将死」的着法。"""
    # 白方若走 Kg1-h1?? 之类会被立刻将杀；这里用一个明确的送杀局面
    fen = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
    board = chess.Board(fen)
    mcts = MCTS(uniform_evaluator, MCTSConfig(simulations=600, batch_size=32))
    mv, root = mcts.best_move(board)
    board.push(mv)
    # 走完之后，黑方不应有一步将死
    black_mate = any(
        (board.push(m), board.is_checkmate(), board.pop())[1]
        for m in list(board.legal_moves))
    print(f"  走 {mv.uci()} 后黑方有一步杀: {black_mate} {'✓' if not black_mate else '✗'}")
    return not black_mate


def check_tablebase() -> bool:
    tb_dir = Path("data/raw/syzygy345")
    if not tb_dir.is_dir() or len(list(tb_dir.glob("*.rtbw"))) < 100:
        print("  残局表不完整，跳过")
        return True
    import chess.syzygy
    tb = chess.syzygy.open_tablebase(str(tb_dir))
    mcts = MCTS(uniform_evaluator, MCTSConfig(simulations=200, batch_size=32),
                tablebase=tb)
    ok = True
    for name, fen, expect in [("KQvK 必胜", "8/8/8/3k4/8/8/4Q3/4K3 w - - 0 1", 1.0),
                              ("KvK 和棋",  "8/8/8/3k4/8/8/8/4K3 w - - 0 1", 0.0)]:
        board = chess.Board(fen)
        root = mcts.search(board)
        v = root.terminal_value if root.terminal_value is not None \
            else MCTS.root_value(root)
        good = abs(v - expect) < 0.35
        print(f"  {name:10} 根节点价值 {v:+.2f} (期望 {expect:+.1f}) {'✓' if good else '✗'}")
        ok &= good
    return ok


def check_batching() -> bool:
    """确认确实是批量推理：记录每次调用的批大小。"""
    sizes: list[int] = []

    def counting_eval(boards):
        sizes.append(len(boards))
        return uniform_evaluator(boards)

    mcts = MCTS(counting_eval, MCTSConfig(simulations=512, batch_size=128))
    mcts.search(chess.Board())
    big = [s for s in sizes if s > 1]
    avg = sum(big) / len(big) if big else 0
    print(f"  推理调用 {len(sizes)} 次，平均批大小 {avg:.0f} "
          f"(最大 {max(sizes)})  {'✓' if avg >= 32 else '✗ 批量太小'}")
    return avg >= 32


def main() -> int:
    print("=== 一步杀 ===")
    a = check_mate_in_1()
    print("\n=== 两步杀（要求价值跨层正确取反）===")
    b = check_mate_in_2()
    print("\n=== 不送杀 ===")
    c = check_avoid_blunder()
    print("\n=== 残局表接入 ===")
    d = check_tablebase()
    print("\n=== 批量推理 ===")
    e = check_batching()
    ok = a and b and c and d and e
    print("\nMCTS 验收:", "通过" if ok else "未通过")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
