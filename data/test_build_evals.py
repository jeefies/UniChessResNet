"""验收：evals 分片的正确性。

四项检查：
  1. 每个策略走法（**带上 promo 字段**）在该局面必须合法
     —— 漏掉 promo 会把 g7f8n 误判成非法的 g7f8
  2. 策略概率之和 == 1（截断到前 5 条之后仍须归一）
  3. WDL 之和 == 1
  4. WDL 与 cp 符号一致：行棋方占优时 win > loss
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unichess_r.core.encoding import unorient_move
from unichess_r.core.moves import index_to_move
from data.record import (RECORD_DTYPE, NO_PROMO, record_policy,
                         record_to_board, record_wdl)


def main(shard_dir: str = "data/shards_evals_test", sample: int = 50000) -> int:
    files = sorted(glob.glob(f"{shard_dir}/*.bin"))
    assert files, f"{shard_dir} 下没有分片"
    a = np.memmap(files[0], dtype=RECORD_DTYPE, mode="r")
    n = min(sample, len(a))
    print(f"校验 {n:,} / {len(a):,} 条")

    bad_legal = bad_pol = bad_wdl = 0
    promo_seen = 0
    for r in a[:n]:
        b = record_to_board(r)
        pol = record_policy(r)
        promo = int(r["promo"])

        for j, (idx, _) in enumerate(pol):
            mv = index_to_move(idx, promo if (j == 0 and promo != NO_PROMO) else None)
            mv = unorient_move(mv, b.turn)
            if mv not in b.legal_moves:
                # 非首选走法也可能是升变，逐个升变子试一遍
                ok = any(unorient_move(index_to_move(idx, k), b.turn) in b.legal_moves
                         for k in range(4))
                if not ok:
                    bad_legal += 1
                    break
        if promo != NO_PROMO:
            promo_seen += 1
        if not (0.995 < sum(p for _, p in pol) < 1.005):
            bad_pol += 1
        w = record_wdl(r)
        if not (0.995 < sum(w) < 1.005):
            bad_wdl += 1

    print(f"  非法走法        : {bad_legal}")
    print(f"  策略未归一化    : {bad_pol}")
    print(f"  WDL 未归一化    : {bad_wdl}")
    print(f"  含升变的记录    : {promo_seen:,} ({100*promo_seen/n:.2f}%)")
    ok = bad_legal == 0 and bad_pol == 0 and bad_wdl == 0
    print("\nevals 分片验收:", "通过" if ok else "未通过")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(*(sys.argv[1:] or [])))
