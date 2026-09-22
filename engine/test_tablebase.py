"""残局表验收：引擎在少子残局必须能把棋「收掉」。

无搜索网络最典型的失败是「知道自己赢，但走不出将杀，拖成 50 步和棋」，
所以挂 Syzygy 不是锦上添花，是必需品。

判定标准不用人工预期（容易写错），而是**以表自身的 WDL 为裁判**：
  表说必胜 -> 引擎必须将死对手
  表说和棋 -> 引擎至少不能输（对手随机走时赢了也算合格）
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import chess
import chess.syzygy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.engine import UniChessEngine

CASES = [
    # 注意：必须是合法局面。若「未行棋方正被将军」，python-chess 会生成
    # 「吃王」走法，探到不存在的表（如 KQv），排查起来很费时间。
    ("KQvK",   "8/8/8/3k4/8/8/4Q3/4K3 w - - 0 1"),
    ("KRvK",   "8/8/8/3k4/8/8/4R3/4K3 w - - 0 1"),
    ("KBNvK",  "8/8/8/4k3/8/8/8/3BKN2 w - - 0 1"),
    ("KPvK-1", "8/8/8/4K3/4P3/8/8/4k3 w - - 0 1"),
    ("KPvK-2", "8/8/8/8/4k3/8/4P3/4K3 w - - 0 1"),
    ("KRPvK",  "8/8/8/3k4/8/8/4PR2/4K3 w - - 0 1"),
    ("KQvKR",  "8/8/8/3k4/7r/8/4Q3/4K3 w - - 0 1"),
]


def main(ckpt: str, tb_path: str, trials: int = 3) -> int:
    tb = chess.syzygy.open_tablebase(tb_path)
    eng = UniChessEngine(ckpt, device="cpu", syzygy_path=tb_path)
    print(f"{'用例':10} {'表判定':>8}  {'实战结果':>10}  {'半步':>5}  判定")
    all_ok = True

    for name, fen in CASES:
        root = chess.Board(fen)
        assert root.is_valid(), f"{name} 的 FEN 是非法局面: {fen}"
        try:
            wdl = tb.probe_wdl(root)
        except Exception as e:
            print(f"{name:10} 探测失败: {type(e).__name__}: {e}")
            all_ok = False
            continue
        expect = "必胜" if wdl > 0 else ("和棋" if wdl == 0 else "必负")

        worst = None
        for t in range(trials):
            rng = random.Random(1000 + t)
            b = chess.Board(fen)
            plies = 0
            while not b.is_game_over(claim_draw=False) and plies < 300:
                mv = eng.play(b) if b.turn == chess.WHITE \
                     else rng.choice(list(b.legal_moves))
                assert mv in b.legal_moves, f"非法走法 {mv} @ {b.fen()}"
                b.push(mv)
                plies += 1
            got = ("将死" if b.is_checkmate() else
                   "被将死" if b.is_game_over() and not b.is_checkmate() and False else
                   "和/未决")
            if b.is_checkmate() and b.turn == chess.WHITE:
                got = "被将死"          # 轮到白走却已被将死
            if worst is None or (got != "将死"):
                worst = (got, plies)

        got, plies = worst
        if wdl > 0:
            ok = got == "将死"          # 必胜局面必须真的将死
        elif wdl == 0:
            ok = got != "被将死"        # 和棋局面不能输
        else:
            ok = True
        all_ok &= ok
        print(f"{name:10} {expect:>8}  {got:>10}  {plies:>5}  {'✓' if ok else '✗'}")

    print()
    print("残局表验收:", "通过" if all_ok else "未通过")
    return 0 if all_ok else 1


def _default_ckpt() -> str:
    """挑一个实际存在的权重。

    原先写死的是 `runs/smoke/ckpt_00000200.pt`，那个冒烟产物早被清掉了，
    于是这个测试只会抛 FileNotFoundError——看起来像环境坏了，其实是路径过期。
    这里改成到 runs/ 下找最新的 ckpt：残局收官靠的是 Syzygy 而不是网络，
    用哪个权重都测得出来，唯独「哪个权重存在」不该写死在源码里。
    """
    root = Path(__file__).resolve().parent.parent
    cands = sorted((root / "runs").glob("*/ckpt_*.pt"))
    if not cands:
        raise SystemExit("runs/ 下没有任何 ckpt_*.pt，先训一个或指定路径："
                         "python engine/test_tablebase.py <ckpt> [syzygy 目录]")
    return str(cands[-1])


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else _default_ckpt()
    tb = sys.argv[2] if len(sys.argv) > 2 else "data/raw/syzygy345"
    raise SystemExit(main(ckpt, tb))
