"""用 Stockfish 的 UCI_Elo 档位标定我们引擎的绝对强度。

Stockfish 的 UCI_LimitStrength + UCI_Elo 是官方校准过的，
比拿 nodes 数自己换算靠谱。做法是在若干档位上各打一批，
找到胜率跨过 50% 的位置，那里就是我们的 Elo。

每个开局双方各执一次白（arena.run_match 保证），并给出误差棒。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.arena import (EngineSpec, elo_with_error, fmt_elo, los,
                        make_openings, run_match)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="我方引擎命令")
    ap.add_argument("--name", default="UniChess")
    ap.add_argument("--stockfish", default="tools/stockfish")
    ap.add_argument("--elos", default="1350,1600,1900,2200",
                    help="要测的 Stockfish UCI_Elo 档位")
    ap.add_argument("--pairs", type=int, default=30, help="每档的开局组数")
    ap.add_argument("--movetime", type=float, default=0.5)
    ap.add_argument("--sf-nodes", type=int, default=0,
                    help=">0 时改用固定节点数而不是限时")
    ap.add_argument("--openings", default="eval/openings.txt")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    op = Path(a.openings)
    if not op.exists():
        print(f"生成 {max(a.pairs, 40)} 个均势开局 ...")
        fens = make_openings(a.stockfish, max(a.pairs, 40))
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text("\n".join(fens) + "\n")
    fens = [x for x in op.read_text().splitlines() if x.strip()][:a.pairs]
    print(f"每档 {len(fens)} 个开局 x 2 局 = {2*len(fens)} 局\n")

    mine = EngineSpec(a.engine.split(), a.name, movetime=a.movetime)
    results = []
    for elo in [int(x) for x in a.elos.split(",")]:
        opp = EngineSpec(
            [a.stockfish], f"SF@{elo}",
            options={"UCI_LimitStrength": True, "UCI_Elo": elo,
                     "Threads": 1, "Hash": 64},
            nodes=a.sf_nodes or None,
            movetime=None if a.sf_nodes else a.movetime)
        print(f"--- vs Stockfish UCI_Elo={elo} ---")
        r = run_match(mine, opp, fens, workers=a.workers, elo0=-50, elo1=50)
        r["opponent_elo"] = elo
        # 我方绝对 Elo ≈ 对手档位 + 相对差
        if r["elo"] not in (float("inf"), float("-inf")):
            r["abs_elo"] = elo + r["elo"]
        results.append(r)
        print(f"  {r['w']}胜 {r['d']}和 {r['l']}负  得分率 {r['score']:.3f}  "
              f"相对 Elo {fmt_elo(r['elo'])} "
              f"[{fmt_elo(r['elo_lo'])}, {fmt_elo(r['elo_hi'])}]")
        if "abs_elo" in r:
            print(f"  -> 推算绝对 Elo ≈ {r['abs_elo']:.0f}")
        print()

    print("=" * 62)
    print(f"{'对手档位':>10} {'得分率':>8} {'相对Elo':>10} {'推算绝对Elo':>12}")
    for r in results:
        abs_s = f"{r['abs_elo']:.0f}" if "abs_elo" in r else "—"
        print(f"{r['opponent_elo']:>10} {r['score']:>8.3f} "
              f"{fmt_elo(r['elo']):>10} {abs_s:>12}")

    est = [r["abs_elo"] for r in results if "abs_elo" in r]
    if est:
        print(f"\n各档推算的均值: {sum(est)/len(est):.0f} Elo")
        print("（跨档一致性越好，估计越可信；差异大说明档位选得太偏）")
    if a.out:
        Path(a.out).write_text(json.dumps(results, ensure_ascii=False, indent=1,
                                          default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
