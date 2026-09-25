# AGENTS.md — UniChess ResNet

> 面向 AI 编码 agent。最后更新：2026-09-25（扁平化重构后）。

## 仓库形态

**仓库根目录即包**：`import ResNet`，import 根是 `~/UniChess`（SSM / Kit / ResNet /
Transformer / Server 五个仓库的共同父目录）。7 个文件之外没有别的东西：

| 文件 | 职责 |
|---|---|
| `model.py` | 网络结构。`state_dict` 键名与重构前**逐字节兼容**——`runs/` 里的旧权重能直接续训 |
| `evaluator.py` | 特征前端：fp16 autocast、分桶子批数写在平面 0-11 通道里 |
| `kit.py` | kit 接入：`make_player_factory` / `make_evaluators` / `make_task` / `load_preset` |
| `engine.py` | Server 六方法插件（`KIT_FACTORY="ResNet.kit:make_player_factory"`） |
| `configs/{stage1,iteration46}.json` | 训练口径（Trainer 顶层字段全在这里） |
| `tests/test_r2.py` | 单测（12 项） |
| `__init__.py` | 包声明 |

**已删除**（git 历史可查，勿 recreate）：`unichess_r/` 包、`data/*.py`、`eval/`、
`gpubench` / `prec_bench` / `split_bench` / `scripts_*`、`run_*.sh`、`unichess*.sh`、`uci.py`。

## 常用命令

```bash
cd ~/UniChess
# 训练（搜索、数据构建都不在本仓库）
python -m Kit train ResNet/configs/stage1.json
python -m Kit train ResNet/configs/iteration46.json
# 单测
python -m unittest ResNet.tests.test_r2            # 12 项
# 对局 / 观战
python -m Kit match <config.json> --out runs/<name>/results.jsonl
```

权重在 `runs/`（未跟随 git）。推理路径、batch 约定见 `evaluator.py` 的 docstring。

## 平面与记录格式（仍然有效，改动前先看单测）

- 输入平面 19 层 8×8，**第 0-11 通道携带分桶子批数**——`evaluator.py` 从平面读它，
  不是从配置读。改记录格式会同时打断 R 自己的推理和 kit 的 planes19 编码 parity。
- `model.py` 的 `state_dict` 键名是权重兼容的底线；要加模块就加，不要改名。
- Elo / KGP 等性能数字记录在 `README.md`（历史实测值，不要回填新数去"对齐"它们）。

## 架构概览

ResNet 引擎：平面编码 → ResNet 主干 → policy（平方到平方双线性头 + promo 专用头）与
WDL 值头。训练期数据管线、损失、调度器全在 kit（`Kit/planes19/` + `Kit/train/`），
本仓库只保留网络与前端。批量对弈与观战走 kit 原生 Player（跨局攒批），
Server 通过 `engine.py` 以六方法契约驱动。

历史契约（对局复现性、eval 换算、GSPBT 口径）已由 kit 的回归测试接替：`Kit/tests/`。
