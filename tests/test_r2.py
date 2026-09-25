"""ResNet 重建后的测试：模型结构、检查点兼容、kit 接入、Server 插件、训练任务。

CPU 可跑的部分（不需要权重）：结构/参数计数、cfg_conflicts、export 协
议、preset 映射、训练适配器。需要权重的部分标 ``_HAS_CKPT``，远端存在时自动跑。
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

import chess  # noqa: E402

from ResNet import kit as rkit  # noqa: E402
from ResNet.evaluator import ResNetEngine, describe, load_checkpoint  # noqa: E402
from ResNet.model import NetConfig, PRESETS, UniChessNet, cfg_conflicts, count_params  # noqa: E402

_STAGE1 = ROOT / "runs" / "stage1" / "ckpt_00187578.pt"
_HAS_CKPT = _STAGE1.exists()


def _fake_ckpt(tmp: Path, cfg: NetConfig, step: int = 5) -> Path:
    """写一个随机权重的 checkpoint（用于 export / 加载往返，不碰真实权重）。"""
    import torch

    torch.manual_seed(0)
    p = tmp / "fake.pt"
    torch.save({"model": UniChessNet(cfg).state_dict(),
                "cfg": dataclasses.asdict(cfg), "step": step}, p)
    return p


class TestNetStructure(unittest.TestCase):
    def test_preset_count_and_defaults(self):
        self.assertEqual(sorted(PRESETS), ["large", "medium", "small", "tiny"])
        self.assertEqual(PRESETS["medium"], NetConfig(blocks=15, filters=192))
        self.assertEqual(NetConfig().name, "15x192")

    def test_forward_shapes_and_key_names(self):
        import torch

        net = UniChessNet(PRESETS["small"])
        p, pr, w = net(torch.zeros(2, 19, 8, 8))
        self.assertEqual(tuple(p.shape), (2, 4096))
        self.assertEqual(tuple(pr.shape), (2, 4))
        self.assertEqual(tuple(w.shape), (2, 3))
        # state_dict 键名冻结：权重文件靠它加载
        keys = set(net.state_dict())
        self.assertIn("policy_conv.weight", keys)
        self.assertIn("promo_head.0.weight", keys)
        self.assertIn("value_fc.0.weight", keys)
        self.assertIn("tower.4.se.fc2.weight", keys)
        self.assertNotIn("policy_bilinear.weight", keys)

    def test_bucketed_forward(self):
        import torch

        cfg = NetConfig(blocks=2, filters=16, num_buckets=4)
        net = UniChessNet(cfg)
        x = torch.zeros(3, 19, 8, 8)
        b = torch.tensor([0, 1, 3])
        p, pr, w = net(x, b)
        self.assertEqual(tuple(p.shape), (3, 4096))
        with self.assertRaises(ValueError):
            UniChessNet(cfg)(x)                    # num_buckets>1 必须给 bucket

    def test_cfg_conflicts(self):
        self.assertEqual(cfg_conflicts({"blocks": 15, "filters": 192}, PRESETS["medium"]), [])
        # filters 的默认值正好等于 medium 的值，所以「缺失」不算冲突——只 blocks 冲突
        self.assertEqual(cfg_conflicts({"blocks": 10}, PRESETS["medium"]),
                         ["blocks: checkpoint=10，当前=15"])
        self.assertEqual(cfg_conflicts({"blocks": 15, "filters": 128}, PRESETS["medium"]),
                         ["filters: checkpoint=128，当前=192"])
        # 只影响 policy_head / promo_head 的字段：结构不同，必须报出来
        self.assertEqual(len(cfg_conflicts({"blocks": 15, "filters": 192, "promo_head": "bn"},
                                           PRESETS["medium"])), 1)
        with self.assertRaises(TypeError):
            UniChessNet(NetConfig(**{"no_such_field": 1}))

    def test_params_not_blown_up(self):
        """策略头不得退化成 flatten + FC（50M 参数）——medium 的 policy_conv 只有 12352。"""
        g = count_params(UniChessNet(PRESETS["medium"]))
        self.assertEqual(g["policy_conv"], 12352)
        self.assertEqual(g["TOTAL"], 10567027)
        self.assertLess(g["policy_conv"], 0.01 * g["TOTAL"])


class TestCheckpointRoundTrip(unittest.TestCase):
    def test_export_and_reload(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            p = _fake_ckpt(Path(d), PRESETS["small"], step=42)
            model, cfg, step, _ = load_checkpoint(p)
            self.assertEqual(step, 42)
            self.assertEqual(cfg, PRESETS["small"])
            engine = ResNetEngine(p, device="cpu", half=False)
            self.assertFalse(engine.half)
            self.assertEqual(engine.num_buckets, 1)
            self.assertEqual(engine.bucket_of(32), 0)
            self.assertEqual(engine.bucket_of(1), 0)
            # 训练适配器 export 必须能重新加载
            from ResNet.kit import RTrainAdapter

            out = RTrainAdapter(cfg="small").export(model, 7)
            self.assertEqual(out["step"], 7)
            q = Path(d) / "again.pt"
            import torch

            torch.save(out, q)
            model2, _, step2, _ = load_checkpoint(q)
            self.assertEqual(step2, 7)
            self.assertEqual({k: v.shape for k, v in model.state_dict().items()},
                             {k: v.shape for k, v in model2.state_dict().items()})
    def test_reject_transformer_ckpt(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.pt"
            import torch

            torch.save({"model": {}, "cfg": {"d_model": 8}}, p)
            from ResNet.kit import RTrainAdapter

            with self.assertRaises(ValueError):
                ResNetEngine(p, device="cpu")
            with self.assertRaises(ValueError):
                RTrainAdapter(cfg="small", base_ckpt=p).build()


class TestConfigs(unittest.TestCase):
    """``configs/*.json`` 必须逐字段复刻旧配方的模型与数据口径。

    历史坑：预设置信地把 ``large`` 当成 iteration46 的规模，实际旧脚本写的是
    ``NetConfig(blocks=24, filters=320)``（46.3M 参数），large 只有 24.8M——
    训练照样跑完、loss 曲线看着也正常，训出来的权重却与生产检查点结构不同，
    永远对不上。所以这里直接从磁盘读配置、按模型参数量钉死。
    """

    def _load(self, name: str) -> dict:
        cfg = json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))
        self.assertEqual(cfg["task"]["factory"], "ResNet.kit:make_task")
        return cfg

    def test_stage1(self):
        cfg = self._load("stage1.json")
        kw = cfg["task"]["kwargs"]
        self.assertEqual(kw["cfg"], "medium")
        self.assertEqual(kw["data"]["kind"], "loader")
        self.assertEqual(kw["data"]["batch_size"], 1024)
        self.assertEqual(kw["loss"]["kind"], "r_stage1")
        self.assertTrue(kw["loss"]["legal_mask"])
        self.assertTrue(cfg["grad_scaler"])
        self.assertEqual(cfg["precision"], "bf16")
        self.assertEqual(cfg["accum"], 1)
        self.assertEqual(cfg["seed"], 20260908)
        self.assertEqual(cfg["steps"], 187578)
        self.assertEqual(cfg["schedule"], {"kind": "onecycle", "pct_start": 0.05})
        self.assertNotIn("fused", cfg["optimizer"])     # None → CUDA 上自动 fused（旧脚本同款）
        task = rkit.make_task(**kw)
        self.assertEqual(task.adapter.cfg, PRESETS["medium"])
        self.assertEqual(sum(p.numel() for p in task.build_model().parameters()), 10567027)

    def test_iteration46(self):
        cfg = self._load("iteration46.json")
        kw = cfg["task"]["kwargs"]
        self.assertEqual(kw["cfg"], {"blocks": 24, "filters": 320})
        self.assertEqual(kw["data"]["kind"], "pool")
        self.assertEqual(kw["data"]["shards"]["slice"], [None, -4])       # 训练分片
        self.assertEqual(kw["validation"]["shards"]["slice"], [-4, None])  # held-out 分片
        self.assertEqual(kw["data"]["cap"], 50000)
        self.assertEqual(kw["data"]["mode"], "mixed")
        self.assertEqual(cfg["accum"], 8)
        self.assertEqual(cfg["seed"], 20260909)
        self.assertEqual(cfg["optimizer"]["fused"], True)
        self.assertTrue(kw["channels_last"])
        self.assertEqual(cfg["torch"], {"num_threads": 4, "cudnn_benchmark": True})
        self.assertEqual(cfg["schedule"],
                         {"kind": "warmup_cosine_floor", "warmup": 2000, "floor": 0.05,
                          "scale": 0.95})
        task = rkit.make_task(**kw)
        self.assertEqual(task.adapter.cfg, NetConfig(blocks=24, filters=320))
        self.assertEqual(sum(p.numel() for p in task.build_model().parameters()), 46340899)
        self.assertEqual(
            sorted(k for k in task.export(task.build_model(), 3)), ["cfg", "model", "step"])


class TestKitAdapters(unittest.TestCase):
    def test_preset_mapping(self):
        for name in ("max_mcts", "fast", "policy", "cpu"):
            kw = rkit.load_preset(name)
            self.assertEqual(kw["simulations"], {"max_mcts": 800, "fast": 200,
                                                 "policy": 0, "cpu": 80}[name])
            self.assertIn("checkpoint", kw)
            self.assertIn("syzygy_path", kw)
        with self.assertRaises(KeyError):
            rkit.load_preset("nope")

    def test_task_construction(self):
        # 只建任务（不加载权重）：data/loss kind 校验 + 模型预设映射
        task = rkit.make_task(cfg="small",
                               data={"kind": "loader", "shards": {"dir": "x"}, "batch_size": 8},
                               loss={"kind": "r_stage1"})
        self.assertEqual(task.num_buckets, 1)
        self.assertFalse(task.channels_last)
        self.assertTrue(len(list(task.build_model().parameters())) > 0)
        # cfg 既接受预设名，也接受 NetConfig 字段 dict（旧脚本用它指定非预设规模）
        big = rkit.make_task(cfg={"blocks": 3, "filters": 32},
                             data={"kind": "loader", "shards": {"dir": "x"}, "batch_size": 8},
                             loss={"kind": "r_stage1"})
        self.assertEqual(big.adapter.cfg, NetConfig(blocks=3, filters=32))
        with self.assertRaises(KeyError):
            rkit.make_task(cfg={"blocks": 3, "nope": 1},
                           data={"kind": "loader", "shards": {"dir": "x"}, "batch_size": 8},
                           loss={"kind": "r_stage1"})
        with self.assertRaises(TypeError):
            rkit.make_task(cfg=15,
                           data={"kind": "loader", "shards": {"dir": "x"}, "batch_size": 8},
                           loss={"kind": "r_stage1"})
        with self.assertRaises(KeyError):
            rkit.make_task(cfg="giant",
                           data={"kind": "loader", "shards": {"dir": "x"}}, loss={"kind": "r_stage1"})
        with self.assertRaises(ValueError):
            rkit.make_task(cfg="small", data={"kind": "nope", "shards": {"dir": "x"}},
                           loss={"kind": "r_stage1"})

    def test_engine_module_contract(self):
        """Server 按路径加载本仓库的 engine.py：KIT_FACTORY 与 GameEngine 必须存在。

        服务端用 ``spec_from_file_location`` 起了一个合成的模块名，与本测试里
        直接 exec_module 得到的是**两个**不同 module 对象，所以类不能按身份比较——
        这里断言契约本身。
        """
        from Kit.serving import load_engine_class

        spec = importlib.util.spec_from_file_location("_r_engine_test", ROOT / "engine.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertEqual(mod.KIT_FACTORY, "ResNet.kit:make_player_factory")
        self.assertTrue(mod.GameEngine.IMPLEMENTED)
        self.assertEqual(mod.GameEngine.__name__, "ResNetEngine")
        cls = load_engine_class(ROOT / "engine.py")
        self.assertEqual(cls.KIT_FACTORY, "ResNet.kit:make_player_factory")
        self.assertTrue(cls.IMPLEMENTED)
        self.assertEqual(cls.__name__, "ResNetEngine")


if __name__ == "__main__":
    unittest.main()
