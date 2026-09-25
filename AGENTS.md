# AGENTS.md — UniChess ResNet

> **本文件尚未按重构后的布局重写**（全量重构 R0–R6 进行中）。下面的条目大部分描述的是
> 已删除的 `unichess_r` 包与旧脚本，**已失效**；动手前先读这一节，能用的信息以本仓库的
> `README.md`「用法」一节和实际存在的文件为准。
>
> 重构后（当前 `rebuild` 分支，已推 GitHub）的真实情况：
>
> - **仓库根即包**：`import ResNet`，import 根是 `~/UniChess`。文件只有
>   `model.py`（结构，state_dict 键名冻结）、`evaluator.py`（批量前向）、
>   `kit.py`（kit 接入：`make_player_factory` / `make_evaluators` / `make_task` /
>   `make_adapter`）、`engine.py`（Server 插件，`KIT_FACTORY="ResNet.kit:make_player_factory"`）、
>   `configs/{stage1,iteration46}.json`、`tests/test_r2.py`。
> - 旧 `unichess_r` 包、`data/*.py`、`eval/`、`gpubench` / `prec_bench` / `split_bench` /
>   `scripts_*`、`run_*.sh`、`unichess*.sh`、`uci.py` 已删除（git 历史里可查）。
> - 训练、搜索、数据构建都不在本仓库：`python -m Kit train ResNet/configs/<name>.json`，
>   搜索是 kit 的 C++ PUCT，数据构建是 `Kit/planes19/build`。
> - 权重在 `runs/`（未跟踪），推理口径见 `evaluator.py` 的 docstring：
>   推理走 fp16 autocast、分桶子力数从平面 0-11 数。
>
> 仍然有效的部分：下面的「平面与记录格式」「MCTS 不变量」（搜索已挪到 kit，坑本身已由
> kit 的回归测试接管）、「危险命令」、性能测量记录（README 里的 Elo / 吞吐数字）。
