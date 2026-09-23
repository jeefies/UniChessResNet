"""UniChessKit 接入：把 UniChessEngine 的批量前向包装成 kit 的 BatchEvaluator / PlayerFactory。

kit 的 PUCT 是 unichess_r/search/mcts.py 的协程化移植（有逐节点 parity 测试），
搜索、跨局攒批、裁决与统计都在 kit；本模块只负责加载权重。

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


def make_evaluator(checkpoint=DEFAULT_CKPT, *, device: str = "auto", half: Optional[bool] = None,
                   max_batch: Optional[int] = 256) -> BatchFnEvaluator:
    import torch

    from unichess_r.engine.engine import UniChessEngine
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = _resolve(checkpoint).resolve()
    engine = UniChessEngine(ckpt, device=device, mcts_sims=0, half=half)
    # 同一权重 + 同一精度才能拼进同一批前向
    return BatchFnEvaluator(f"R:{ckpt}:{'fp16' if engine.half else 'fp32'}",
                            engine.evaluate_batch, max_batch)


def make_player_factory(checkpoint=None, *, preset: Optional[str] = None, name: str = "R",
                        device: Optional[str] = None, half: Optional[bool] = None,
                        max_batch: Optional[int] = 256, **search_kwargs):
    """``preset`` 取 config.json 的值作默认，显式参数优先；其余参数透传给
    ``make_search_player_factory``（simulations / batch_size / syzygy_path / book_path /
    book_plies / temperature / PUCTConfig 字段）。"""
    explicit = dict(checkpoint=checkpoint, device=device, half=half)
    opts = {**(load_preset(preset) if preset else {}),
            **{k: v for k, v in explicit.items() if v is not None}, **search_kwargs}
    evaluator = make_evaluator(opts.pop("checkpoint", DEFAULT_CKPT),
                               device=opts.pop("device", "auto"), half=opts.pop("half", None),
                               max_batch=max_batch)
    for key in ("syzygy_path", "book_path"):
        if opts.get(key):
            opts[key] = str(_resolve(opts[key]))
    return make_search_player_factory(name, evaluator, **opts)
