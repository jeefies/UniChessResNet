"""从 96 字节定长分片读训练样本。

分片用内存映射打开，解码在 __getitem__ 里做（CPU worker 并行），
所以磁盘上只放紧凑记录，不放展开的浮点平面。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from unichess_r.core.encoding import NUM_PLANES
from data.record import RECORD_DTYPE, NO_PROMO

POLICY_SIZE = 4096
_SCALE = 65535.0

# 平面偏移，必须与 core/encoding.py 的布局一致
_P_OWN, _P_OPP = 0, 6
_P_CASTLE, _P_EP, _P_HALF, _P_REP = 12, 16, 17, 18


def decode_batch(recs: np.ndarray) -> np.ndarray:
    """一批记录 -> (N, 19, 8, 8) float32。纯 numpy 位运算，无 python-chess 开销。"""
    n = len(recs)
    planes = np.zeros((n, NUM_PLANES, 64), dtype=np.float32)

    occ_w = recs["occ_white"].astype(np.uint64)
    occ_b = recs["occ_black"].astype(np.uint64)
    side = recs["side"].astype(bool)          # True = 黑方行棋
    # 我方/对方占位：黑方行棋时对调
    own = np.where(side, occ_b, occ_w)
    opp = np.where(side, occ_w, occ_b)

    bits = np.arange(64, dtype=np.uint64)
    for i, field in enumerate(("pawns", "knights", "bishops", "rooks", "queens", "kings")):
        bb = recs[field].astype(np.uint64)
        present = ((bb[:, None] >> bits[None, :]) & np.uint64(1)).astype(bool)
        is_own = ((own[:, None] >> bits[None, :]) & np.uint64(1)).astype(bool)
        is_opp = ((opp[:, None] >> bits[None, :]) & np.uint64(1)).astype(bool)
        planes[:, _P_OWN + i] = (present & is_own).astype(np.float32)
        planes[:, _P_OPP + i] = (present & is_opp).astype(np.float32)

    c = recs["castling"].astype(np.uint8)
    own_k = np.where(side, (c >> 2) & 1, c & 1)
    own_q = np.where(side, (c >> 3) & 1, (c >> 1) & 1)
    opp_k = np.where(side, c & 1, (c >> 2) & 1)
    opp_q = np.where(side, (c >> 1) & 1, (c >> 3) & 1)
    for off, v in enumerate((own_k, own_q, opp_k, opp_q)):
        planes[:, _P_CASTLE + off] = v.astype(np.float32)[:, None]

    ep = recs["ep"].astype(np.int16)
    has_ep = ep != 255
    rows = np.nonzero(has_ep)[0]
    if len(rows):
        planes[rows, _P_EP, ep[rows]] = 1.0

    planes[:, _P_HALF] = (np.minimum(recs["halfmove"], 100) / 100.0
                          ).astype(np.float32)[:, None]
    planes[:, _P_REP] = (np.minimum(recs["rep"], 2) / 2.0).astype(np.float32)[:, None]

    planes = planes.reshape(n, NUM_PLANES, 8, 8)
    # 黑方行棋的样本需要上下翻转（颜色已在上面对调）
    flip = np.nonzero(side)[0]
    if len(flip):
        planes[flip] = planes[flip][:, :, ::-1, :]
    return planes


# 易位索引的两种写法。镜像后（永远是我方在下方）王一定在 e1 = 4 号格。
_E1 = 4
_CASTLE_FIX = (
    (_E1 * 64 + 7, _E1 * 64 + 6),   # e1h1（王吃己车，Chess960 写法）-> e1g1
    (_E1 * 64 + 0, _E1 * 64 + 2),   # e1a1                          -> e1c1
)


def _repair_castling(policy: np.ndarray, recs: np.ndarray) -> int:
    """把易位标签从「王吃己车」写法搬到「王走两格」写法，返回搬动的条数。

    data/shards_evals 是用修复前的 build_evals.py 建的：那份代码用
    `chess.Move.from_uci` 直接吃 Lichess 评估库的 UCI，而该库把易位写成
    e1h1 / e1a1。`mv in board.legal_moves` 对这种形式返回 True
    （python-chess 内部会先归一化再判定），于是它通过了校验，索引却按
    to_square=h1 算成了 4*64+7。

    **推理端读的是 4*64+6**（对 board.legal_moves 里的 e1g1 算索引）。
    所以这些概率在实战中一次也读不到——等于网络从来没学过易位，引擎也
    从来拿不到易位的先验。实测：12G 分片里 2587 个易位标签有 2585 个是
    错误形式，占全部策略质量的 1.99%。

    重建 12G 分片代价太高，所以在解码时修。判据是「我方王是否在 e1」：
    索引 4*64+7 本身并不一定是易位——王在别处时，e1 上的车走到 h1 也是
    这个索引，那是合法的普通着法，绝不能搬。而王在 e1 时，王不可能用一步
    走到 h1，所以此时该索引只可能是易位。

    修好 build_evals.py 之后新建的分片不会再命中这里（那时索引已经是
    4*64+6），这个函数会自然变成空操作，留着是为了老分片还能用。
    """
    kings = recs["kings"].astype(np.uint64)
    white = recs["occ_white"].astype(np.uint64)
    black = recs["occ_black"].astype(np.uint64)
    # side: 0=白。镜像后我方在下方，所以白方看 E1(4)、黑方看 E8(60)。
    is_white = recs["side"] == 0
    on_e1 = np.where(is_white,
                     (kings & white & np.uint64(1 << 4)) != 0,
                     (kings & black & np.uint64(1 << 60)) != 0)
    rows = np.flatnonzero(on_e1)
    moved = 0
    for src, dst in _CASTLE_FIX:
        r = rows[policy[rows, src] > 0]
        if r.size:
            policy[r, dst] += policy[r, src]
            policy[r, src] = 0.0
            moved += int(r.size)
    return moved


def decode_targets(recs: np.ndarray):
    """记录 -> (policy_target[N,4096], promo_target[N], wdl_target[N,3])。"""
    n = len(recs)
    policy = np.zeros((n, POLICY_SIZE), dtype=np.float32)
    mv = recs["policy_move"].astype(np.int32)
    pb = recs["policy_prob"].astype(np.float32) / _SCALE
    rows = np.repeat(np.arange(n), 5)
    np.add.at(policy, (rows, mv.ravel()), pb.ravel() * (pb.ravel() > 0))
    _repair_castling(policy, recs)      # 老分片的易位索引修正，见该函数
    s = policy.sum(axis=1, keepdims=True)
    np.divide(policy, s, out=policy, where=s > 0)

    promo = recs["promo"].astype(np.int64)
    promo = np.where(promo == NO_PROMO, -100, promo)  # -100 = CrossEntropy 的 ignore_index

    wdl = recs["wdl"].astype(np.float32) / _SCALE
    ws = wdl.sum(axis=1, keepdims=True)
    np.divide(wdl, ws, out=wdl, where=ws > 0)
    return policy, promo, wdl


def piece_count_bucket(recs: np.ndarray, num_buckets: int) -> np.ndarray:
    """按子力数分桶（Stockfish NNUE 的做法：免费路由器，不破坏批量）。"""
    occ = recs["occ_white"].astype(np.uint64) | recs["occ_black"].astype(np.uint64)
    cnt = np.zeros(len(recs), dtype=np.int64)
    x = occ.copy()
    while x.any():
        cnt += (x & np.uint64(1)).astype(np.int64)
        x >>= np.uint64(1)
    # 32 子均分到 num_buckets 桶
    return np.clip((cnt - 1) * num_buckets // 32, 0, num_buckets - 1)


class ShardDataset(Dataset):
    """把一个目录下的所有 .bin 分片当成一个连续数据集。"""

    def __init__(self, shard_dir: str | Path, num_buckets: int = 1):
        self.paths = sorted(Path(shard_dir).glob("*.bin"))
        if not self.paths:
            raise FileNotFoundError(f"{shard_dir} 下没有 .bin 分片")
        self.num_buckets = num_buckets
        self._maps: list[np.memmap | None] = [None] * len(self.paths)
        counts = [p.stat().st_size // RECORD_DTYPE.itemsize for p in self.paths]
        self.offsets = np.cumsum([0] + counts)
        self.total = int(self.offsets[-1])

    def __len__(self) -> int:
        return self.total

    def _map(self, i: int) -> np.memmap:
        if self._maps[i] is None:  # 每个 worker 进程各自映射，不能跨进程共享
            self._maps[i] = np.memmap(self.paths[i], dtype=RECORD_DTYPE, mode="r")
        return self._maps[i]

    def __getitem__(self, idx: int):
        s = int(np.searchsorted(self.offsets, idx, side="right") - 1)
        rec = self._map(s)[idx - self.offsets[s]:idx - self.offsets[s] + 1]
        x = decode_batch(rec)[0]
        p, pr, w = decode_targets(rec)
        out = [torch.from_numpy(x), torch.from_numpy(p[0]),
               torch.tensor(pr[0]), torch.from_numpy(w[0])]
        if self.num_buckets > 1:
            out.append(torch.tensor(piece_count_bucket(rec, self.num_buckets)[0]))
        return tuple(out)

class BatchShardDataset(Dataset):
    """按**整批**取样的数据集。

    为什么需要它：decode_batch 是向量化的位运算，一次解 512 条和解 1 条的
    耗时差不多。原来的 ShardDataset 每次 __getitem__ 只解一条，
    等于把向量化的收益全扔了，DataLoader 成了瓶颈而 GPU 喂不饱。

    用法（注意 batch_size=None，批的划分交给 sampler）：
        from torch.utils.data import BatchSampler, RandomSampler
        ds = BatchShardDataset(dir)
        sampler = BatchSampler(RandomSampler(ds), batch_size=1024, drop_last=True)
        loader = DataLoader(ds, sampler=sampler, batch_size=None, num_workers=4)
    """

    def __init__(self, shard_dir: str | Path, num_buckets: int = 1):
        self.paths = sorted(Path(shard_dir).glob("*.bin"))
        if not self.paths:
            raise FileNotFoundError(f"{shard_dir} 下没有 .bin 分片")
        self.num_buckets = num_buckets
        self._maps: list[np.memmap | None] = [None] * len(self.paths)
        counts = [p.stat().st_size // RECORD_DTYPE.itemsize for p in self.paths]
        self.offsets = np.cumsum([0] + counts)
        self.total = int(self.offsets[-1])

    def __len__(self) -> int:
        return self.total

    def _map(self, i: int) -> np.memmap:
        if self._maps[i] is None:   # 每个 worker 进程各自映射，不能跨进程共享
            self._maps[i] = np.memmap(self.paths[i], dtype=RECORD_DTYPE, mode="r")
        return self._maps[i]

    def _gather(self, indices: np.ndarray) -> np.ndarray:
        """把跨分片的一批全局下标，取成一个连续的记录数组。"""
        shard = np.searchsorted(self.offsets, indices, side="right") - 1
        out = np.empty(len(indices), dtype=RECORD_DTYPE)
        for s in np.unique(shard):
            sel = shard == s
            local = indices[sel] - self.offsets[s]
            out[sel] = self._map(int(s))[local]
        return out

    def __getitem__(self, indices):
        if isinstance(indices, (int, np.integer)):
            indices = [int(indices)]
        idx = np.asarray(indices, dtype=np.int64)
        recs = self._gather(idx)

        x = torch.from_numpy(decode_batch(recs))
        p, pr, w = decode_targets(recs)
        out = [x, torch.from_numpy(p), torch.from_numpy(pr), torch.from_numpy(w)]
        if self.num_buckets > 1:
            out.append(torch.from_numpy(piece_count_bucket(recs, self.num_buckets)))
        return tuple(out)


def make_loader(shard_dir, batch_size: int, *, num_buckets: int = 1,
                num_workers: int = 4, shuffle: bool = True, pin_memory: bool = False):
    """建一个按整批解码的 DataLoader。"""
    from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

    ds = BatchShardDataset(shard_dir, num_buckets=num_buckets)
    base = RandomSampler(ds) if shuffle else SequentialSampler(ds)
    sampler = BatchSampler(base, batch_size=batch_size, drop_last=True)
    loader = DataLoader(ds, sampler=sampler, batch_size=None,
                        num_workers=num_workers, pin_memory=pin_memory,
                        persistent_workers=num_workers > 0)
    return ds, loader
