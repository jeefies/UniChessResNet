"""UniChess 推理引擎（Stage 1：无搜索，一次前向出招）。

出招优先级：
  1. 残局表：子力数 <= 5 时查 Syzygy，直接拿精确结果
     —— 无搜索网络最典型的丢人失败就是「赢棋收不掉，拖成 50 步和棋」
  2. 开局书：前若干步从 Polyglot 随机抽，避免每局开局雷同
  3. 网络：前向 -> 屏蔽非法走法 -> 取最大（或按温度采样）

Stage 2 会在这上面套 MCTS；接口保持不变。
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import chess
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.encoding import encode, orient_move, unorient_move
from core.moves import move_to_index, move_to_promo_index, PROMO_PIECES
from model.net import NetConfig, UniChessNet
# MCTS 原先在 __init__ 里按 mcts_sims>0 延迟导入。提到顶层是为了让根目录 engine.py
# 的导入隔离能一次性把 search.mcts 也纳入（函数作用域 import 会在调用时重新经由
# sys.modules 解析，届时 'search.mcts' 可能已被同机共存的 Transformer 仓库占住）。
# 无循环依赖：search/mcts.py 只导入 chess / numpy / core.encoding / core.moves，
# 不导入 engine.*，core 两个模块也不导入 search.*。
from search.mcts import MCTS, MCTSConfig


class UniChessEngine:
    def __init__(self, ckpt_path: str | Path, *, device: str = "cpu",
                 syzygy_path: str | Path | None = None,
                 book_path: str | Path | None = None,
                 book_plies: int = 10, temperature: float = 0.0,
                 seed: int | None = None, mcts_sims: int = 0,
                 mcts_batch: int = 128, half: bool | None = None):
        self.device = torch.device(device)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        cfg_dict = ckpt.get("cfg", {})
        if "d_model" in cfg_dict and "num_layers" in cfg_dict:
            try:
                from UniChessTransformer.model.transformer import ChessTransformer, TransformerConfig
            except ImportError:
                parent_dir = str(Path(__file__).resolve().parent.parent.parent)
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)
                from UniChessTransformer.model.transformer import ChessTransformer, TransformerConfig
            t_cfg = TransformerConfig.from_dict(cfg_dict)
            if not hasattr(t_cfg, "name"):
                t_cfg.name = f"Transformer-{t_cfg.d_model}x{t_cfg.num_layers}"
            self.cfg = t_cfg
            self.model = ChessTransformer(t_cfg).to(self.device).eval()
            state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
            self.model.load_state_dict(state_dict)
        else:
            cfg = NetConfig(**cfg_dict)
            self.model = UniChessNet(cfg).to(self.device).eval()
            self.model.load_state_dict(ckpt["model"])
            self.cfg = cfg
        # 推理用 fp16：实测 11,010 -> 14,302 局面/秒（1.3x）。
        # 选 fp16 而非 bf16 是因为同样速度下数值保真度好一个数量级
        # （策略最大偏差 1.75e-3 vs 1.28e-2）。
        self.half = (self.device.type == "cuda") if half is None else half
        self.temperature = temperature
        self.book_plies = book_plies
        self.rng = random.Random(seed)

        self.tablebase = None
        self._tb_missing: set[str] = set()
        if syzygy_path and Path(syzygy_path).is_dir():
            try:
                import chess.syzygy
                self.tablebase = chess.syzygy.open_tablebase(str(syzygy_path))
            except Exception as e:
                print(f"info string 残局表加载失败: {e}", file=sys.stderr)

        self.mcts_sims = mcts_sims
        self.mcts = None

        # 搜索树跨手复用的状态（见 _take_root）。_root_ply / _root_epd 一起
        # 构成「这棵树属于哪个局面」的凭证，缺一不可。
        self._root = None
        self._root_ply: int = -1
        self._root_epd: str | None = None
        self.last_source: str = "network"

        self.book = None
        if book_path and Path(book_path).exists():
            try:
                import chess.polyglot
                self.book = chess.polyglot.open_reader(str(book_path))
            except Exception as e:
                print(f"info string 开局书加载失败: {e}", file=sys.stderr)

        if mcts_sims > 0:
            # MCTS / MCTSConfig 已在模块顶层导入（原因见顶部 import 段的注释）
            self.mcts = MCTS(
                self.evaluate_batch,
                MCTSConfig(simulations=mcts_sims, batch_size=mcts_batch,
                           temperature=temperature),
                tablebase=self.tablebase)

    # ---------- 各出招来源 ----------

    def _from_book(self, board: chess.Board) -> chess.Move | None:
        if self.book is None or board.fullmove_number * 2 > self.book_plies:
            return None
        try:
            entries = list(self.book.find_all(board))
        except Exception:
            return None
        if not entries:
            return None
        weights = [max(e.weight, 1) for e in entries]
        return self.rng.choices([e.move for e in entries], weights=weights)[0]

    def _from_tablebase(self, board: chess.Board) -> chess.Move | None:
        """子力 <=5 时用 Syzygy 精确收官。

        排序键 (对手WDL, 是否非清零着法, DTZ)：
          1. 先保住最好的结果——对手 WDL 越小越好
          2. **在结果相同的前提下优先清零着法（推兵/吃子）**
          3. 最后取 DTZ 小的

        第 2 条是必需的，不是优化。DTZ 是「距下一次清零的步数」，
        只要盘上有兵，推兵随时可清零，DTZ 就恒为 2 左右，
        在兵残局里完全不提供梯度——王会原地打转直到 50 步和棋。
        推兵是不可逆的真进展，升变后变成无兵残局，DTZ 的梯度才恢复。
        """
        if self.tablebase is None or chess.popcount(board.occupied) > 5:
            return None
        if not board.is_valid():
            # 非法局面（例如「未行棋方正被将军」）会让 python-chess 生成「吃王」走法，
            # 进而探到不存在的表。直接放弃残局表。
            return None
        best, best_key = None, None
        for mv in board.legal_moves:
            board.push(mv)
            try:
                if board.is_checkmate():
                    wdl, dtz = -2, 0          # 对手被将死 = 我方必胜且最快
                elif board.is_stalemate() or board.is_insufficient_material():
                    wdl, dtz = 0, 0
                else:
                    wdl = self.tablebase.probe_wdl(board)   # 对手视角
                    dtz = abs(self.tablebase.probe_dtz(board))
                    # 50 步规则先于残局表的结论生效。DTZ 不计半步钟，但实战里
                    # halfmove_clock + dtz >= 100 就先判和了，这个「必胜」兑现不了。
                    # search/mcts.py:_exact_value(:176) 一直有这层保护，这里没有——
                    # 而残局表在 play() 里的优先级**高于 MCTS**，于是整条正确的
                    # 搜索逻辑会被一个兑现不了的结论盖掉，和棋当赢棋走。
                    # 降级成 0（实际结果就是和），让排序去挑真正能在 50 步内
                    # 兑现的着法；若全都兑现不了，上面的 zeroing 会优先推兵/吃子
                    # 来重置半步钟，这正是唯一还有希望的下法。
                    # wdl=+2（我方必负）同样降级：对手也一样收不掉，那就是和棋，
                    # 排序上理应好过真的输棋。
                    if abs(wdl) == 2 and board.halfmove_clock + dtz >= 100:
                        wdl = 0
            except Exception as e:
                err_key = type(e).__name__ + ":" + str(e)
                if err_key not in self._tb_missing:
                    self._tb_missing.add(err_key)
                    print(f"info string 残局表探测失败，本局面回退到网络: {e}",
                          file=sys.stderr)
                return None
            finally:
                board.pop()
            # 清零着法 = 推兵或吃子；zeroing=0 排在 zeroing=1 前面
            zeroing = 0 if board.is_zeroing(mv) else 1
            key = (wdl, zeroing, dtz)
            if best_key is None or key < best_key:
                best, best_key = mv, key
        return best

    def _from_network(self, board: chess.Board) -> chess.Move:
        policy, promo, _ = self.evaluate(board)
        legal = list(board.legal_moves)
        scores = []
        for mv in legal:
            om = orient_move(mv, board.turn)
            s = policy[move_to_index(om)]
            pi = move_to_promo_index(om)
            if pi is not None:
                s = s * promo[pi]
            scores.append(s)
        scores = np.asarray(scores, dtype=np.float64)

        if self.temperature <= 0:
            return legal[int(scores.argmax())]
        p = scores ** (1.0 / self.temperature)
        total = p.sum()
        if not np.isfinite(total) or total <= 0:
            return legal[int(scores.argmax())]
        return legal[int(self.rng.choices(range(len(legal)), weights=(p / total))[0])]

    # ---------- 对外接口 ----------

    @torch.no_grad()
    def evaluate_batch(self, boards: list[chess.Board]):
        """一次前向评估一批局面。MCTS 靠它把 GPU 喂满。

        返回 (policy[N,4096], promo[N,4], wdl[N,3])，均为各自行棋方视角的概率。
        单条推理和 256 条推理在 GPU 上耗时相近，所以 MCTS 必须批量收集叶子
        再调这里，而不是每个叶子调一次 evaluate()。
        """
        xs = np.stack([encode(b) for b in boards])
        x = torch.from_numpy(xs).to(self.device)
        bucket = None
        if getattr(self.cfg, "num_buckets", 1) > 1:
            nb = self.cfg.num_buckets
            idx = [min(max((chess.popcount(b.occupied) - 1) * nb // 32, 0), nb - 1)
                   for b in boards]
            bucket = torch.tensor(idx, device=self.device)
        args = (x, bucket) if bucket is not None or not hasattr(self.cfg, "d_model") else (x,)
        if self.half:
            with torch.autocast("cuda", dtype=torch.float16):
                p_l, pr_l, w_l = self.model(*args)
        else:
            p_l, pr_l, w_l = self.model(*args)
        return (torch.softmax(p_l.float(), 1).cpu().numpy(),
                torch.softmax(pr_l.float(), 1).cpu().numpy(),
                torch.softmax(w_l.float(), 1).cpu().numpy())

    @torch.no_grad()
    def evaluate(self, board: chess.Board) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 (policy[4096] 概率, promo[4] 概率, wdl[3] 概率)，均为当前行棋方视角。"""
        x = torch.from_numpy(encode(board)).unsqueeze(0).to(self.device)
        bucket = None
        if getattr(self.cfg, "num_buckets", 1) > 1:
            n = chess.popcount(board.occupied)
            b = min(max((n - 1) * self.cfg.num_buckets // 32, 0), self.cfg.num_buckets - 1)
            bucket = torch.tensor([b], device=self.device)
        args = (x, bucket) if bucket is not None or not hasattr(self.cfg, "d_model") else (x,)
        if self.half:
            with torch.autocast("cuda", dtype=torch.float16):
                p_l, pr_l, w_l = self.model(*args)
        else:
            p_l, pr_l, w_l = self.model(*args)
        return (torch.softmax(p_l[0].float(), 0).cpu().numpy(),
                torch.softmax(pr_l[0].float(), 0).cpu().numpy(),
                torch.softmax(w_l[0].float(), 0).cpu().numpy())

    # ---------- 搜索树跨手复用 ----------

    _ROOT_MAX_SKIP = 4          # 跳过这么多手以上就重建：重放的收益追不上失配的风险

    def _take_root(self, board: chess.Board):
        """把上一步留下的搜索树对齐到当前局面；对不上就返回 None（从零重建）。

        复用的收益全来自对手走的那一手：它的子树上一轮已经搜过，省下的是
        一整批网络前向，典型能少算三到六成模拟。

        风险是**把别的对局的树接到当前局面上**。`MCTS.search` 在
        `root.expanded` 为真时会跳过根节点展开（search/mcts.py:313），
        不去核对 root 是否真的对应这个 board——接错了不会报错，会静默地
        按另一个局面下棋。UCI 下这完全可能发生：`ucinewgame` 之后对手可能
        已经先走了几手，单看 move_stack 的长度分不出是同一局还是新的一局。

        所以这里真把局面退回去比对 EPD。代价是几次 pop，比一次网络前向
        便宜两个数量级，没有任何理由为它省。
        """
        root, ply, epd = self._root, self._root_ply, self._root_epd
        self._root, self._root_ply, self._root_epd = None, -1, None
        if root is None or ply < 0 or epd is None:
            return None
        back = len(board.move_stack) - ply
        # back == 0：同一局面被重复搜索（arena 重发 go），照样可以复用
        if back < 0 or back > self._ROOT_MAX_SKIP:
            return None
        probe = board.copy(stack=True)
        for _ in range(back):
            probe.pop()
        if probe.epd() != epd:
            return None                       # 不是同一条棋路
        for mv in board.move_stack[ply:]:
            root = MCTS.advance_root(root, mv)
            if root is None:
                return None                   # 这一手当时没被搜到，无树可复用
        return root

    def reset_search(self) -> None:
        """丢弃搜索树。新开一局（UCI 的 ucinewgame）必须调，否则会跨局残留。"""
        self._root, self._root_ply, self._root_epd = None, -1, None
        self.last_source = "network"

    def search_info(self) -> dict:
        """上一次 MCTS 搜索的可上报信息，供 UCI 的 info 行使用。

        score 取根节点访问数最大那一枝的 Q（**搜索之后**的值），而不是网络
        直出的 WDL：前者才是引擎真正据以决策的数，而且省掉一次额外前向。
        非 MCTS 出招（残局表/开局书/网络直出）时 self._root 是上一步的旧树，
        报出去就是错的，所以用 last_source 挡住。
        """
        info = {"sims": 0, "nodes": 0, "depth": 0, "q": None, "pv": [],
                "reused": False, "stopped_early": False}
        if self.last_source != "mcts" or self.mcts is None:
            return info
        m = getattr(self.mcts, "last_metrics", {})
        info["sims"] = int(m.get("simulations", 0))
        info["nodes"] = int(m.get("network_positions", 0))
        info["depth"] = int(m.get("max_depth", 0))
        info["reused"] = bool(m.get("reused_root", False))
        info["stopped_early"] = bool(m.get("stopped_early", False))

        root = self._root
        if root is None or not root.moves or root.sum_N <= 0:
            return info
        i = int(np.argmax(root.N))
        if root.N[i] > 0:
            info["q"] = float(root.W[i]) / float(root.N[i])
        # 主变：每层取访问数最大的一枝，直到没有已展开的子节点
        node, pv = root, []
        while node is not None and node.moves and len(pv) < 24:
            j = int(np.argmax(node.N))
            if node.N[j] <= 0:
                break
            pv.append(node.moves[j])
            node = node.children[j]
        info["pv"] = pv
        return info

    def _from_mcts(self, board: chess.Board, sims: int | None = None,
                   deadline: float | None = None) -> chess.Move | None:
        n = self.mcts_sims if sims is None else sims
        if self.mcts is None or n <= 0:
            return None
        reused = self._take_root(board)
        try:
            mv, root = self.mcts.best_move(board, simulations=n,
                                           temperature=self.temperature,
                                           root=reused, deadline=deadline)
        except Exception:
            if reused is None:
                raise
            # 复用的树出问题只会炸在这里。从零重搜一次，别把这一步降级成网络直出。
            print("info string 搜索树复用失败，本步从零重建", file=sys.stderr)
            mv, root = self.mcts.best_move(board, simulations=n,
                                           temperature=self.temperature,
                                           deadline=deadline)
        self._root = root
        self._root_ply = len(board.move_stack)
        self._root_epd = board.epd()
        return mv

    def play(self, board: chess.Board, *, sims: int | None = None,
             deadline: float | None = None) -> chess.Move:
        """选出一步棋。保证返回合法走法。

        优先级：残局表 > 开局书 > MCTS（若开启）> 网络直出。

        sims     本步的 MCTS 模拟次数上限，不传就沿用构造时的 mcts_sims。
        deadline 墙钟截止（time.perf_counter() 的刻度）。**计时赛必须传它**：
                 「模拟次数」换算成「时间」依赖一个 nps 估计，而 nps 随设备、
                 网络规模、这一步能复用多少树而变，第一步更是完全没有实测值。
                 只靠 sims 封顶就会在首步超时判负（实测 CPU 上差了一个数量级）。
        """
        self.last_source = "network"
        for name, source in (("tablebase", self._from_tablebase),
                             ("book", self._from_book),
                             ("mcts", lambda b: self._from_mcts(b, sims, deadline))):
            mv = source(board)
            if mv is not None and mv in board.legal_moves:
                self.last_source = name
                return mv
        return self._from_network(board)
