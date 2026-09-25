"""R 的前向引擎：批量评估局面，给 kit 的 Player / Trainer 用。

与旧 ``unichess_r/engine/engine.py`` 的区别：那是一个完整的 UCI 引擎（开局书、残局表、
Python MCTS、搜索树复用），对弈路径已经整体交给 kit 的 PUCT；这里只保留**批量前向**，
不再有第二个搜索实现。旧引擎仍然需要的两个语义保留：

- **推理走 fp16 而不是 bf16**：``prec_bench.py`` 实测同样速度下数值保真度好一个数量级
  （策略最大偏差 1.75e-3 vs 1.28e-2）。训练用 bf16 autocast（见 configs/*.json）。
- **分桶头的子力数从平面 0-11 数出来**，与 ``chess.popcount(board.occupied)`` 相同，
  这样 kit 的 C++ PUCT 只给平面就能走同一条路。

权重文件格式：``{"model": state_dict, "cfg": NetConfig 实例的 __dict__, "step": int}``，
另有一支 ``{"best": {...}}`` 包装（iteration46 的 ``best.pt``）。
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .model import NetConfig, UniChessNet, cfg_conflicts

ROOT = Path(__file__).resolve().parent


def _resolve_ckpt(path) -> Path:
    """权重路径按仓库根目录解析（Server 适配层 engine.py 的约定一致）。"""
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"权重不存在：{p}")
    return p


def load_checkpoint(path) -> tuple[UniChessNet, NetConfig, int, dict]:
    """→ (eval 模式的模型, NetConfig, step, 原始 checkpoint 字典)。"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    inner = ckpt.get("best", ckpt) if isinstance(ckpt, dict) else ckpt
    cfg_dict = inner.get("cfg", {}) or {}
    if "d_model" in cfg_dict:
        raise ValueError(f"{path} 是 Transformer 检查点，不要用 ResNet 的 loader 加载")
    cfg = NetConfig(**cfg_dict)
    model = UniChessNet(cfg)
    model.load_state_dict(inner["model"])
    return model.eval(), cfg, int(inner.get("step", 0)), ckpt


def describe(path) -> dict:
    """只读权重元信息（加载模型之前就能报结构冲突）。"""
    _, cfg, step, _ = load_checkpoint(path)
    return {"cfg": asdict(cfg), "step": step, "name": cfg.name}


class ResNetEngine:
    """``evaluate_planes(xs)`` / ``evaluate_batch(boards)``，返回 (policy, promo, wdl) 概率。

    ``cfg`` 是加载的 ``NetConfig``；``step`` 是 checkpoint 里记的训练步。
    """

    def __init__(self, ckpt_path, *, device: str = "auto",
                 half: Optional[bool] = None):
        self.device = torch.device(device if device != "auto"
                                   else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.cfg, self.step, _ = load_checkpoint(ckpt_path)
        self.model.to(self.device)
        self.ckpt_path = str(Path(ckpt_path).resolve())
        self.num_buckets = int(self.cfg.num_buckets)
        self.channels_last = False
        # 只有 CUDA 上 fp16 有意义（CPU 的 fp16 没有加速且数值更差）
        self.half = bool(self.device.type == "cuda" and (True if half is None else half))
        self._bucket_cache: dict[int, int] = {}

    # ---- 分桶（见模块 docstring）----
    def bucket_of(self, pieces: int) -> int:
        nb = self.cfg.num_buckets
        return min(max((int(pieces) - 1) * nb // 32, 0), nb - 1)

    def _buckets(self, xs: np.ndarray) -> Optional[torch.Tensor]:
        if getattr(self.cfg, "num_buckets", 1) <= 1:
            return None
        pieces = np.rint(xs[:, :12].sum(axis=(1, 2, 3))).astype(np.int64)
        idx = [self._bucket_cache.setdefault(int(n), self.bucket_of(int(n))) for n in pieces]
        return torch.tensor(idx, device=self.device, dtype=torch.long)

    @torch.no_grad()
    def evaluate_planes(self, xs: np.ndarray):
        """xs: (N, 19, 8, 8) float32 → (policy[N,4096], promo[N,4], wdl[N,3]) 行棋方视角概率。"""
        xs = np.ascontiguousarray(xs, dtype=np.float32)
        x = torch.from_numpy(xs).to(self.device)
        bucket = self._buckets(xs)
        args = (x, bucket) if bucket is not None else (x,)
        if self.half:
            with torch.autocast("cuda", dtype=torch.float16):
                p_l, pr_l, w_l = self.model(*args)
        else:
            p_l, pr_l, w_l = self.model(*args)
        return (torch.softmax(p_l.float(), 1).cpu().numpy(),
                torch.softmax(pr_l.float(), 1).cpu().numpy(),
                torch.softmax(w_l.float(), 1).cpu().numpy())

    @torch.no_grad()
    def evaluate_batch(self, boards):
        from Kit.planes19 import encode
        return self.evaluate_planes(np.stack([encode(b) for b in boards]))

    # ---- 与 TrainTask 的接口 ----
    def build(self) -> UniChessNet:
        """新建一个同结构模型（未加载权重）；TrainTask.build_model 用它。"""
        return UniChessNet(self.cfg)

    def forward(self, model, x, bucket=None, *, mlh=False):
        """TrainTask 约定的前向：x 是 (N,19,8,8) float32。"""
        args = (x, bucket) if bucket is not None else (x,)
        return model(*args)

    def export(self, model, step: int) -> dict:
        """写出的 checkpoint 与旧 ``train.py`` 的格式一致（``{model, cfg, step}``）。"""
        return {"model": model.state_dict(), "cfg": asdict(self.cfg), "step": int(step)}


__all__ = ["ResNetEngine", "UniChessNet", "cfg_conflicts", "describe", "load_checkpoint"]
