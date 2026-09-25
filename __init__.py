"""UniChess ResNet 引擎（Stockfish 蒸馏 + 自对弈）。

仓库根目录即包：``import ResNet``，import 根是它的父目录（远端 ``~/UniChess``）。
- ``model.py``     网络结构（state_dict 键名冻结，权重靠它加载）
- ``evaluator.py`` 批量前向（kit 的 Player / Trainer 只用这个）
- ``kit.py``       kit 接入：对局 PlayerFactory + 训练 TrainTask
- ``configs/``    训练与对局的 JSON 配置
- ``engine.py``   Server 的模型插件（六方法 GameEngine）
"""
from .model import PRESETS, NetConfig, UniChessNet

__all__ = ["PRESETS", "NetConfig", "UniChessNet"]
