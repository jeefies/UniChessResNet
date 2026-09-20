"""验收 + 基准：整批解码的 BatchShardDataset。

先证明它和逐条解码的 ShardDataset **逐元素完全一致**，再测速度。
顺序不能反——一个更快但结果不同的数据管线是灾难，
它会静默地把网络训歪，而且要等到测 Elo 才发现。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.dataset import (BatchShardDataset, ShardDataset, make_loader)


def check_equivalence(shard_dir: str, n: int = 3000) -> bool:
    """同样的下标，两条路必须给出完全相同的张量。"""
    old = ShardDataset(shard_dir)
    new = BatchShardDataset(shard_dir)
    assert len(old) == len(new), f"长度不一致 {len(old)} vs {len(new)}"

    rng = np.random.default_rng(20260908)
    idx = rng.choice(len(new), size=n, replace=False)

    # 逐条
    xs, ps, prs, ws = [], [], [], []
    for i in idx:
        x, p, pr, w = old[int(i)]
        xs.append(x); ps.append(p); prs.append(pr); ws.append(w)
    x_old = torch.stack(xs)
    p_old = torch.stack(ps)
    pr_old = torch.stack(prs)
    w_old = torch.stack(ws)

    # 整批
    x_new, p_new, pr_new, w_new = new[idx]

    checks = [
        ("输入平面", x_old, x_new),
        ("策略目标", p_old, p_new),
        ("升变目标", pr_old, pr_new),
        ("WDL 目标", w_old, w_new),
    ]
    ok = True
    for name, a, b in checks:
        same = torch.equal(a, b)
        if not same:
            diff = (a != b).sum().item()
            print(f"  {name}: ✗ 有 {diff} 处不同")
            ok = False
        else:
            print(f"  {name}: ✓ 逐元素一致  {tuple(a.shape)}")
    return ok


def bench(shard_dir: str, batch: int = 512, steps: int = 40,
          workers: int = 4) -> None:
    print(f"\n  批大小 {batch}, {steps} 批, {workers} workers")

    # 旧：逐条 __getitem__ + 默认 collate
    from torch.utils.data import DataLoader
    old = ShardDataset(shard_dir)
    dl_old = DataLoader(old, batch_size=batch, shuffle=True, drop_last=True,
                        num_workers=workers, persistent_workers=workers > 0)
    it = iter(dl_old)
    next(it)                       # 预热，排除 worker 启动开销
    t0 = time.time()
    for _ in range(steps):
        next(it)
    dt_old = time.time() - t0
    del it, dl_old

    # 新：整批解码
    _, dl_new = make_loader(shard_dir, batch, num_workers=workers)
    it = iter(dl_new)
    next(it)
    t0 = time.time()
    for _ in range(steps):
        next(it)
    dt_new = time.time() - t0
    del it, dl_new

    sps_old = steps * batch / dt_old
    sps_new = steps * batch / dt_new
    print(f"  逐条解码: {dt_old:6.2f}s  {sps_old:>9,.0f} 样本/秒")
    print(f"  整批解码: {dt_new:6.2f}s  {sps_new:>9,.0f} 样本/秒")
    print(f"  提速 {sps_new / sps_old:.1f}x")


def main() -> int:
    shard_dir = sys.argv[1] if len(sys.argv) > 1 else "data/shards_evals"
    if not list(Path(shard_dir).glob("*.bin")):
        print(f"{shard_dir} 下没有分片")
        return 1

    print("=== 等价性（必须先过这关）===")
    ok = check_equivalence(shard_dir)
    if not ok:
        print("\n结果不一致，不做基准测试。")
        return 1

    print("\n=== 吞吐基准 ===")
    bench(shard_dir)
    print("\nDataLoader 验收: 通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
