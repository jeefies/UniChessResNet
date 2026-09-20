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

        self.book = None
        if book_path and Path(book_path).exists():
            try:
                import chess.polyglot
                self.book = chess.polyglot.open_reader(str(book_path))
            except Exception as e:
                print(f"info string 开局书加载失败: {e}", file=sys.stderr)

        if mcts_sims > 0:
            from search.mcts import MCTS, MCTSConfig
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

    def _from_mcts(self, board: chess.Board) -> chess.Move | None:
        if self.mcts is None or self.mcts_sims <= 0:
            return None
        mv, _ = self.mcts.best_move(board, simulations=self.mcts_sims,
                                    temperature=self.temperature)
        return mv

    def play(self, board: chess.Board) -> chess.Move:
        """选出一步棋。保证返回合法走法。

        优先级：残局表 > 开局书 > MCTS（若开启）> 网络直出。
        """
        for source in (self._from_tablebase, self._from_book, self._from_mcts):
            mv = source(board)
            if mv is not None and mv in board.legal_moves:
                return mv
        return self._from_network(board)
