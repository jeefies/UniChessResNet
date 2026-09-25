"""UniChessKit 接入：把 R 的前向包装成 kit 的 BatchEvaluator / PlayerFactory / TrainTask。

kit 的 PUCT / C++ PUCT 负责搜索、跨局攒批、裁决与统计；``Kit.planes19.task`` 负责训练循环、
数据流与损失口径。本模块只做三件事：加载权重、暴露模型结构、转发参数。

三个入口：

- ``make_player_factory``：对局用（``python -m Kit match`` / Server 的 models/R）。
- ``make_task``：训练用（``python -m Kit train``），实现 ``Kit.train.TrainTask``。
- ``make_evaluators``：只要评估器（基准复现、离线评估）。

对局配置示例::

    {"factory": "ResNet.kit:make_player_factory",
     "root": "/home/jeefy/UniChess",
     "kwargs": {"checkpoint": "ResNet/runs/stage1/ckpt_00187578.pt",
                "simulations": 800, "batch_size": 128}}

训练配置示例（``configs/stage1.json``）：``{"factory": "ResNet.kit:make_task", ...}``。
"""
from __future__ import annotations

import json
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Optional

from Kit.planes19 import BatchFnEvaluator, make_search_player_factory

from .model import PRESETS, NetConfig, UniChessNet, cfg_conflicts

KIT_SPI_VERSION = 2

ROOT = Path(__file__).resolve().parent
DEFAULT_CKPT = ROOT / "runs" / "stage1" / "ckpt_00187578.pt"


def _resolve_ckpt(path) -> Path:
    """权重路径按仓库根目录解析（Server 适配层 engine.py 的约定一致）。"""
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"权重不存在：{p}")
    return p


def _resolve(path) -> Path:
    return path if Path(path).is_absolute() else ROOT / path


# ---------------------------------------------------------------- 对局

# config.json（Server 预设）里与搜索相关的键 → make_player_factory 参数。
# description 只是说明文字，不进参数表；新增无法映射的键要显式忽略而不是静默丢弃。
_PRESET_IGNORED = {"description"}
_PRESET_KEYS = {"ckpt": "checkpoint", "mcts_sims": "simulations", "mcts_batch": "batch_size",
                "device": "device", "half": "half", "syzygy_path": "syzygy_path",
                "book_path": "book_path", "book_plies": "book_plies",
                "temperature": "temperature", "max_batch": "max_batch"}


def load_preset(name: str) -> dict:
    """config.json 的预设 → make_player_factory 的关键字参数。"""
    presets = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    if name not in presets:
        raise KeyError(f"config.json 没有预设 {name!r}（可选：{', '.join(presets)}）")
    unknown = set(presets[name]) - set(_PRESET_KEYS) - _PRESET_IGNORED
    if unknown:
        raise KeyError(f"预设 {name!r} 里有无法映射的键 {sorted(unknown)}"
                       f"（已知可忽略：{sorted(_PRESET_IGNORED)}）")
    return {_PRESET_KEYS[k]: v for k, v in presets[name].items() if k in _PRESET_KEYS}


def make_evaluators(checkpoint=DEFAULT_CKPT, *, device: str = "auto", half: Optional[bool] = None,
                    max_batch: Optional[int] = 256) -> tuple:
    """(棋盘评估器, 编码评估器)，共用同一个模型。编码评估器的负载是 (19,8,8) float32。"""
    import numpy as np

    from .evaluator import ResNetEngine

    engine = ResNetEngine(_resolve_ckpt(checkpoint), device=device, half=half)
    # 同一权重 + 同一精度才能拼进同一批前向
    key = f"R:{engine.ckpt_path}:{'fp16' if engine.half else 'fp32'}"
    return (BatchFnEvaluator(key, engine.evaluate_batch, max_batch),
            BatchFnEvaluator(key + ":planes", lambda xs: engine.evaluate_planes(np.stack(xs)),
                             max_batch))


def make_evaluator(checkpoint=DEFAULT_CKPT, *, device: str = "auto", half: Optional[bool] = None,
                   max_batch: Optional[int] = 256) -> BatchFnEvaluator:
    return make_evaluators(checkpoint, device=device, half=half, max_batch=max_batch)[0]


def make_player_factory(checkpoint=None, *, preset: Optional[str] = None, name: str = "R",
                        device: Optional[str] = None, half: Optional[bool] = None,
                        max_batch: Optional[int] = 256, search_impl: str = "auto", **search_kwargs):
    """``preset`` 取 config.json 的值作默认，显式参数优先；其余透传给
    ``make_search_player_factory``（simulations / batch_size / syzygy_path / book_path /
    book_plies / temperature / search_impl / PUCTConfig 字段）。"""
    # Server 的模型插件把 config.json 的预设 dict **原样**当 kwargs 传进来（不经 preset=，
    # 见 Server/models/__init__.py::resolve_kwargs）。这里与 preset= 路径共用同一套键映射，
    # 否则 ckpt / mcts_sims 会漏到 PUCTConfig 上（typcls: unexpected keyword）。
    search_kwargs = {_PRESET_KEYS.get(k, k): v for k, v in search_kwargs.items()
                     if k not in _PRESET_IGNORED}
    explicit = dict(checkpoint=checkpoint, device=device, half=half)
    opts = {**(load_preset(preset) if preset else {}),
            **{k: v for k, v in explicit.items() if v is not None},
            "search_impl": search_impl, **search_kwargs}
    evaluator, planes_evaluator = make_evaluators(
        opts.pop("checkpoint", DEFAULT_CKPT),
        device=opts.pop("device", "auto"),
        half=opts.pop("half", None),
        max_batch=opts.pop("max_batch", max_batch))
    for key in ("syzygy_path", "book_path"):
        if opts.get(key):
            opts[key] = str(_resolve(opts[key]))
    return make_search_player_factory(name, evaluator, planes_evaluator=planes_evaluator, **opts)


# ---------------------------------------------------------------- 训练


def _make_cfg(cfg) -> NetConfig:
    """``cfg`` 既可以是预设名，也可以是 NetConfig 字段的 dict（旧脚本用它指定 24x320 之类的
    非预设规模；两种写法产出的 ``cfg.__dict__`` 都会进导出的 checkpoint）。"""
    if isinstance(cfg, str):
        if cfg not in PRESETS:
            raise KeyError(f"没有模型预设 {cfg!r}（可选：{', '.join(PRESETS)}）")
        return PRESETS[cfg]
    if isinstance(cfg, dict):
        known = {f.name for f in fields(NetConfig)}
        unknown = sorted(set(cfg) - known)
        if unknown:
            raise KeyError(f"cfg 有未知字段 {unknown}（可选：{sorted(known)}）")
        return replace(NetConfig(), **cfg)
    raise TypeError(f"cfg 只能是预设名或字段 dict，得到 {type(cfg).__name__}")


class RTrainAdapter:
    """``Kit.planes19.task`` 需要的模型适配器（见该模块 docstring）。"""

    def __init__(self, *, cfg="medium", base_ckpt=None, channels_last: bool = False,
                 strict_base: bool = True):
        self.cfg: NetConfig = _make_cfg(cfg)
        self.num_buckets = int(self.cfg.num_buckets)
        self.channels_last = bool(channels_last)
        self.step = 0
        self.base_ckpt = str(base_ckpt) if base_ckpt else None
        self.strict_base = strict_base

    def build(self) -> UniChessNet:
        model = UniChessNet(self.cfg)
        if self.base_ckpt:
            import torch

            inner = torch.load(_resolve_ckpt(self.base_ckpt), map_location="cpu", weights_only=False)
            inner = inner.get("best", inner)
            conflicts = cfg_conflicts(inner.get("cfg", {}) or {}, self.cfg)
            if conflicts:
                raise ValueError(f"{self.base_ckpt} 与模型预设 {self.cfg!r} 结构冲突：{conflicts}")
            missing, unexpected = model.load_state_dict(inner["model"], strict=self.strict_base)
            if not self.strict_base:
                assert not unexpected, f"底座权重里有多余键 {unexpected[:4]}"
                assert not missing, f"底座权重缺键 {missing[:4]}"
            self.step = int(inner.get("step", 0))
        return model

    def forward(self, model, x, bucket=None, *, mlh: bool = False):
        if mlh:
            raise ValueError("ResNet 没有 MLH 头（那是 Transformer 的 P3 预训练头）")
        args = (x, bucket) if bucket is not None else (x,)
        return model(*args)

    def export(self, model, step: int) -> dict:
        """写出的 checkpoint 与旧 ``model/train.py`` 的格式一致（``{model, cfg, step}``）。"""
        return {"model": model.state_dict(), "cfg": asdict(self.cfg), "step": int(step)}


def make_adapter(**kwargs) -> RTrainAdapter:
    """``Kit.planes19.task`` 的 ``model.factory``：返回训练用的模型适配器。

    参数：``cfg``（模型预设名，或 NetConfig 字段的 dict，如
    ``{"blocks": 24, "filters": 320}``）、``base_ckpt``（底座权重，可选）、
    ``channels_last``（输入是否转 channels_last）、``strict_base``。
    """
    kw = {
        "cfg": kwargs.pop("cfg", "medium"),
        "base_ckpt": kwargs.pop("base_ckpt", kwargs.pop("checkpoint", None)),
        "channels_last": kwargs.pop("channels_last", False),
        "strict_base": kwargs.pop("strict_base", True),
    }
    if kwargs:
        raise TypeError(f"make_adapter 收到未知参数 {sorted(kwargs)}")
    return RTrainAdapter(**kw)


def make_task(runtime=None, **kwargs) -> "object":
    """``ResNet.kit:make_task`` → 一个 ``Kit.train.TrainTask``。

    ``env`` 里的 ``model`` 键（若有）只用于指定底座权重的路径，其余一律走顶层参数，
    避免和 ``Kit.planes19.task`` 的 model.kwargs 嵌套混在一起。
    """
    from Kit.planes19.task import Planes19Task

    model_kw = dict(kwargs.pop("model_kwargs", None) or {})
    # 顶层参数优先：env 里的同名键只是旧写法
    merged = {**model_kw, **{k: v for k, v in kwargs.items()
                             if k in ("cfg", "base_ckpt", "channels_last", "strict_base")}}
    for k in merged:
        kwargs.pop(k, None)
    merged.setdefault("cfg", "medium")
    merged.setdefault("channels_last", False)
    return Planes19Task(model={"factory": "ResNet.kit:make_adapter", "kwargs": merged},
                        runtime=runtime, **kwargs)
