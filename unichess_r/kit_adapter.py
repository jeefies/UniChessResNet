"""UniChessKit 接入：把 UniChessEngine 的批量前向包装成 kit 的 BatchEvaluator / PlayerFactory。

kit 的 PUCT 是 unichess_r/search/mcts.py 的协程化移植（有逐节点 parity 测试），
搜索、跨局攒批、裁决与统计都在 kit；本模块只负责加载权重。
搜索默认用 kit 的 C++ PUCT（``search_impl="auto"``，与 kit 的 Python PUCT 逐位一致，叶子编码由 C++ 写出、
经 ``evaluate_planes`` 前向）。``search_impl="python"`` 可切回 Python PUCT 对照，结果相同，宜放在 runtime 里。

批量对弈配置示例（``python -m unichess_kit.match``）::

    {"factory": "unichess_r.kit_adapter:make_player_factory",
     "root": "/home/jeefy/UniChess/ResNet",
     "kwargs": {"preset": "max_mcts"}}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from unichess_kit.contrib.planes19 import BatchFnEvaluator, make_search_player_factory

KIT_SPI_VERSION = 1

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = ROOT / "runs" / "stage1" / "ckpt_00187578.pt"

# config.json 预设里与搜索相关的键 → make_player_factory 参数
_PRESET_KEYS = {"ckpt": "checkpoint", "mcts_sims": "simulations", "mcts_batch": "batch_size",
                "device": "device", "half": "half", "syzygy_path": "syzygy_path",
                "book_path": "book_path", "book_plies": "book_plies",
                "temperature": "temperature"}


def load_preset(name: str) -> dict:
    presets = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    if name not in presets:
        raise KeyError(f"config.json 没有预设 {name!r}（可选：{', '.join(presets)}）")
    return {_PRESET_KEYS[k]: v for k, v in presets[name].items() if k in _PRESET_KEYS}


def _resolve(path) -> Path:
    """相对路径按仓库根目录解析（与 Server 适配层 engine.py 的约定一致）。"""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def make_evaluators(checkpoint=DEFAULT_CKPT, *, device: str = "auto", half: Optional[bool] = None,
                    max_batch: Optional[int] = 256) -> tuple:
    """(棋盘评估器, 编码评估器)，共用同一个引擎。编码评估器的负载是 (19, 8, 8) float32。"""
    import numpy as np
    import torch

    from unichess_r.engine.engine import UniChessEngine
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = _resolve(checkpoint).resolve()
    engine = UniChessEngine(ckpt, device=device, mcts_sims=0, half=half)
    # 同一权重 + 同一精度才能拼进同一批前向
    key = f"R:{ckpt}:{'fp16' if engine.half else 'fp32'}"
    return (BatchFnEvaluator(key, engine.evaluate_batch, max_batch),
            BatchFnEvaluator(key + ":planes", lambda xs: engine.evaluate_planes(np.stack(xs)),
                             max_batch))


def make_evaluator(checkpoint=DEFAULT_CKPT, *, device: str = "auto", half: Optional[bool] = None,
                   max_batch: Optional[int] = 256) -> BatchFnEvaluator:
    return make_evaluators(checkpoint, device=device, half=half, max_batch=max_batch)[0]


def make_player_factory(checkpoint=None, *, preset: Optional[str] = None, name: str = "R",
                        device: Optional[str] = None, half: Optional[bool] = None,
                        max_batch: Optional[int] = 256, **search_kwargs):
    """``preset`` 取 config.json 的值作默认，显式参数优先；其余参数透传给
    ``make_search_player_factory``（simulations / batch_size / syzygy_path / book_path /
    book_plies / temperature / search_impl / PUCTConfig 字段）。"""
    explicit = dict(checkpoint=checkpoint, device=device, half=half)
    opts = {**(load_preset(preset) if preset else {}),
            **{k: v for k, v in explicit.items() if v is not None}, **search_kwargs}
    evaluator, planes_evaluator = make_evaluators(
        opts.pop("checkpoint", DEFAULT_CKPT), device=opts.pop("device", "auto"),
        half=opts.pop("half", None), max_batch=max_batch)
    for key in ("syzygy_path", "book_path"):
        if opts.get(key):
            opts[key] = str(_resolve(opts[key]))
    return make_search_player_factory(name, evaluator, planes_evaluator=planes_evaluator, **opts)
