"""Stage 2：PUCT 蒙特卡洛树搜索。

设计要点（都直接对应方案里的结论）：

**批量是第一位的。** 单个局面前向会浪费 GPU——5070 Ti 对 15x192 网络单条推理
和 256 条推理的耗时差不多。所以用 virtual loss 一次收集 128~512 个叶子，
再一次性喂给网络。这也是为什么不能用 MoE/LoRA 路由：每个叶子走不同的
适配器就没法凑成一次大 GEMM。

**每个节点内部用 numpy 数组存 N/W/P。** PUCT 的 argmax 是最热的操作，
向量化之后比逐个子节点算快一个数量级。这是方案里说的「先试中间档，
很多情况下就够了，能省掉大半移植工作」。

**残局表在内部节点探测。** 子力 <=5 时直接返回精确结果并剪掉整棵子树，
比让网络去猜准得多也快得多。

价值约定：全部是**当前行棋方视角**。WDL 三头折成标量 Q = P(胜) - P(负)。
每上升一层取负。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import chess
import numpy as np


@dataclass
class MCTSConfig:
    simulations: int = 800
    batch_size: int = 128          # 一次收集多少叶子再批量推理
    c_puct: float = 1.8            # PUCT 探索常数
    c_puct_base: float = 19652.0   # cpuct 随访问数缓慢增长（AlphaZero 的做法）
    c_puct_init: float = 1.8
    dirichlet_alpha: float = 0.3   # 国际象棋用 0.3
    dirichlet_eps: float = 0.25
    fpu_reduction: float = 0.2     # 未访问子节点的先验价值折扣
    temperature: float = 0.0       # 0 = 取访问数最大者
    temp_moves: int = 0            # 前多少步用温度采样（自对弈时设 15）
    virtual_loss: float = 1.0
    tablebase_pieces: int = 5
    claim_draw: bool = False
    max_collision: int = 8         # 一批里同一叶子最多被选中几次
    root_min_visits: int = 1       # 根节点每个合法着法至少访问几次


class Node:
    """一个局面。子节点统计量用 numpy 数组存，便于向量化 PUCT。"""

    __slots__ = ("moves", "P", "N", "W", "VL", "children",
                 "expanded", "terminal_value", "sum_N")

    def __init__(self):
        self.moves: list[chess.Move] = []
        self.P: np.ndarray = np.zeros(0, dtype=np.float32)   # 先验
        self.N: np.ndarray = np.zeros(0, dtype=np.int32)     # 访问数
        self.W: np.ndarray = np.zeros(0, dtype=np.float32)   # 累计价值
        self.VL: np.ndarray = np.zeros(0, dtype=np.float32)  # virtual loss
        self.children: list[Node | None] = []
        self.expanded: bool = False
        self.terminal_value: float | None = None   # 终局/残局表给出的精确值
        self.sum_N: int = 0

    def expand(self, moves: list[chess.Move], priors: np.ndarray) -> None:
        n = len(moves)
        self.moves = moves
        self.P = priors.astype(np.float32)
        self.N = np.zeros(n, dtype=np.int32)
        self.W = np.zeros(n, dtype=np.float32)
        self.VL = np.zeros(n, dtype=np.float32)
        self.children = [None] * n
        self.expanded = True

    def q(self) -> np.ndarray:
        """子节点的 Q 值（本节点行棋方视角）。未访问的按 FPU 处理，由调用方给基准。"""
        denom = self.N + self.VL
        out = np.zeros_like(self.W)
        nz = denom > 0
        out[nz] = self.W[nz] / denom[nz]
        return out

    def best_child(self, cfg: MCTSConfig) -> int:
        """PUCT：argmax( Q + c_puct * P * sqrt(sum_N) / (1 + N) )。"""
        total = max(self.sum_N, 1)
        c = (math.log((1 + total + cfg.c_puct_base) / cfg.c_puct_base)
             + cfg.c_puct_init)

        denom = self.N + self.VL
        q = np.zeros_like(self.W)
        visited = denom > 0
        # Treat the temporary virtual loss as a negative value as well as an extra visit.
        # Otherwise pending paths still look neutral and repeatedly collide.
        q[visited] = (self.W[visited] - cfg.virtual_loss * self.VL[visited]) / denom[visited]

        # FPU：未访问的子节点用「父节点已探明部分的价值 - 折扣」作为初值，
        # 避免一上来就把每个子节点都试一遍。
        if visited.any():
            parent_q = float((self.W[visited] - cfg.virtual_loss * self.VL[visited]).sum() / denom[visited].sum())
        else:
            parent_q = 0.0
        q[~visited] = parent_q - cfg.fpu_reduction

        u = c * self.P * math.sqrt(total) / (1.0 + denom)
        return int(np.argmax(q + u))


# ----------------------------------------------------------------- 先验

def priors_from_policy(board: chess.Board, policy: np.ndarray,
                       promo: np.ndarray) -> tuple[list[chess.Move], np.ndarray]:
    """把 4096 维策略 + 4 维升变头，映射到该局面的合法走法先验上。"""
    from core.encoding import orient_move
    from core.moves import move_to_index, move_to_promo_index

    moves = list(board.legal_moves)
    if not moves:
        return [], np.zeros(0, dtype=np.float32)
    scores = np.empty(len(moves), dtype=np.float32)
    for i, mv in enumerate(moves):
        om = orient_move(mv, board.turn)
        s = float(policy[move_to_index(om)])
        pi = move_to_promo_index(om)
        if pi is not None:
            s *= float(promo[pi])
        scores[i] = s
    total = scores.sum()
    if total <= 0:
        scores[:] = 1.0 / len(moves)
    else:
        scores /= total
    return moves, scores


# ----------------------------------------------------------------- 搜索

class MCTS:
    """
    evaluator: 接收 list[chess.Board]，返回 (policy[N,4096], promo[N,4], wdl[N,3])，
               均为**当前行棋方视角**的概率。
    tablebase: 可选的 chess.syzygy.Tablebase。
    """

    def __init__(self, evaluator, cfg: MCTSConfig | None = None, tablebase=None,
                 rng: np.random.Generator | None = None):
        self.evaluator = evaluator
        self.cfg = cfg or MCTSConfig()
        self.tablebase = tablebase
        self.rng = rng or np.random.default_rng()

    # ---------- 终局 / 残局表 ----------

    def _exact_value(self, board: chess.Board) -> float | None:
        """能精确判定就返回该值（行棋方视角），否则 None。"""
        if board.is_checkmate():
            return -1.0                     # 轮到我走却已被将死
        if self.cfg.claim_draw and (board.is_repetition(3) or board.is_fifty_moves()):
            return 0.0
        if (board.is_stalemate() or board.is_insufficient_material()
                or board.is_seventyfive_moves() or board.is_fivefold_repetition()):
            return 0.0
        if (self.tablebase is not None
                and chess.popcount(board.occupied) <= self.cfg.tablebase_pieces
                and board.is_valid()):
            try:
                wdl = self.tablebase.probe_wdl(board)
            except Exception:
                return None
            # WDL: 2 必胜 / 1 困难胜 / 0 和 / -1 困难负 / -2 必负
            if abs(wdl) == 2 and board.halfmove_clock:
                try:
                    dtz = abs(self.tablebase.probe_dtz(board))
                except Exception:
                    return None
                if board.halfmove_clock + dtz >= 100:
                    return None  # Need rule-aware search near a possible 50-move claim.
            return float((wdl == 2) - (wdl == -2))
        return None

    # ---------- 一次批量迭代 ----------

    def _collect(self, root: Node, root_board: chess.Board, want: int):
        """下探收集若干叶子，路上打 virtual loss。

        返回 (leaves, terminal_sims)：
          leaves        需要送网络评估的叶子
          terminal_sims 直接由终局/残局表结算、不需要评估的下探次数

        **两者都算作「一次模拟」。** 早先只把 leaves 计入预算，结果一旦搜到
        杀棋，后续下探全撞在那个终局节点上、反复回传却不消耗预算，
        循环空转——实测 sims=200 时某个着法访问数涨到 7661。
        """
        leaves = []
        terminal_sims = 0
        pending = set()
        spins = 0
        cfg_root_min = self.cfg.root_min_visits
        budget = want
        while (len(leaves) + terminal_sims) < budget and spins < self.cfg.max_collision * want:
            node = root
            board = root_board.copy(stack=True)
            path: list[tuple[Node, int]] = []

            first = True
            while node.expanded and node.terminal_value is None:
                if not node.moves:
                    break
                # 根节点先把每个合法着法都访问到，再交给 PUCT。
                # 否则先验极低的着法会被 U 项和 FPU 一起压死——实测中一个
                # P=0.0001 的一步杀在 800 次模拟里一次都没被访问到。
                # 代价只有 n_legal 次模拟，相对总量可忽略。
                if first and cfg_root_min > 0:
                    unvisited = np.flatnonzero(node.N + node.VL < cfg_root_min)
                    i = int(unvisited[0]) if unvisited.size else node.best_child(self.cfg)
                else:
                    i = node.best_child(self.cfg)
                first = False
                node.VL[i] += self.cfg.virtual_loss
                path.append((node, i))
                board.push(node.moves[i])
                child = node.children[i]
                if child is None:
                    child = Node()
                    node.children[i] = child
                node = child

            if node.terminal_value is not None:
                # 已知精确值，直接回传，不占推理批次，但算一次模拟
                self._backup(path, node.terminal_value)
                terminal_sims += 1
                spins += 1
                continue

            if node.expanded:
                # 已展开但无合法走法（理论上已被 terminal 覆盖），保险起见
                self._backup(path, 0.0)
                terminal_sims += 1
                spins += 1
                continue

            exact = self._exact_value(board)
            if exact is not None:
                node.expanded = True
                node.terminal_value = exact
                self._backup(path, exact)
                terminal_sims += 1
                spins += 1
                continue

            if id(node) in pending:
                self.last_metrics['collisions'] += 1
                # The node is already pending in this batch. Undo only this
                # candidate's virtual loss and keep looking for another leaf;
                # flushing here fragments an 800-simulation search into many
                # tiny GPU batches whenever PUCT revisits a pending leaf.
                for parent, edge in path:
                    parent.VL[edge] -= self.cfg.virtual_loss
                spins += 1
                continue
            pending.add(id(node))
            leaves.append((node, board, path))
            spins += 1
        return leaves, terminal_sims

    def _backup(self, path: list[tuple[Node, int]], value: float) -> None:
        """把叶子的价值沿路径回传，并撤销 virtual loss。value 是叶子行棋方视角。"""
        v = value
        for node, i in reversed(path):
            v = -v                      # 换边，视角取反
            node.N[i] += 1
            node.W[i] += v
            node.VL[i] -= self.cfg.virtual_loss
            node.sum_N += 1

    def _evaluate_and_expand(self, leaves) -> None:
        if not leaves:
            return
        self.last_metrics['network_positions'] += len(leaves)
        self.last_metrics['network_batches'] += 1
        self.last_metrics['max_depth'] = max(self.last_metrics['max_depth'], max(len(p) for _, _, p in leaves))
        boards = [b for _, b, _ in leaves]
        policy, promo, wdl = self.evaluator(boards)
        for k, (node, board, path) in enumerate(leaves):
            moves, priors = priors_from_policy(board, policy[k], promo[k])
            if not moves:
                node.expanded = True
                node.terminal_value = 0.0
                self._backup(path, 0.0)
                continue
            node.expand(moves, priors)
            v = float(wdl[k][0] - wdl[k][2])      # Q = P(胜) - P(负)
            self._backup(path, v)

    # ---------- 对外 ----------

    def search(self, board: chess.Board, simulations: int | None = None,
               add_noise: bool = False, root: Node | None = None) -> Node:
        root_reused = root is not None and root.expanded
        self.last_metrics = {'network_positions': 0, 'network_batches': 0, 'max_depth': 0, 'collisions': 0, 'reused_root': root_reused}
        cfg = self.cfg
        sims = simulations or cfg.simulations
        root = root or Node()

        if not root_reused:
            exact = self._exact_value(board)
            if exact is not None:
                root.expanded = True
                root.terminal_value = exact
                return root

            policy, promo, wdl = self.evaluator([board])
            self.last_metrics['network_positions'] += 1
            self.last_metrics['network_batches'] += 1
            moves, priors = priors_from_policy(board, policy[0], promo[0])
            if not moves:
                root.expanded = True
                root.terminal_value = 0.0
                return root
            root.expand(moves, priors)

        if add_noise and len(root.moves) > 1:
            # 根节点加 Dirichlet 噪声，保证自对弈的探索多样性
            noise = self.rng.dirichlet([cfg.dirichlet_alpha] * len(root.moves))
            root.P = ((1 - cfg.dirichlet_eps) * root.P
                      + cfg.dirichlet_eps * noise).astype(np.float32)

        done = 0
        while done < sims:
            want = min(cfg.batch_size, sims - done)
            leaves, terminal_sims = self._collect(root, board, want)
            if not leaves and terminal_sims == 0:
                break
            self._evaluate_and_expand(leaves)
            done += len(leaves) + terminal_sims
        return root

    @staticmethod
    def advance_root(root: Node | None, move: chess.Move) -> Node | None:
        """Reuse the searched child after a real move; unrelated roots stay separate."""
        if root is None or not root.expanded:
            return None
        for index, candidate in enumerate(root.moves):
            if candidate == move:
                child = root.children[index]
                if child is not None:
                    child.VL.fill(0.0)
                return child
        return None

    def best_move(self, board: chess.Board, simulations: int | None = None,
                  temperature: float | None = None,
                  add_noise: bool = False, root: Node | None = None) -> tuple[chess.Move, Node]:
        root = self.search(board, simulations, add_noise=add_noise, root=root)
        if not root.moves:
            legal = list(board.legal_moves)
            if not legal:
                raise ValueError("无合法走法")
            if self.tablebase is not None and not board.is_game_over():
                ranked = []
                for move in legal:
                    child = board.copy(stack=True)
                    child.push(move)
                    value = self._exact_value(child)
                    if value is None:
                        continue
                    try:
                        dtz = abs(self.tablebase.probe_dtz(child))
                    except Exception:
                        dtz = 0 if child.is_game_over() else 1000
                    # Preserve WDL first; shorten winning DTZ, prolong losing DTZ.
                    ranked.append((-value, -dtz if value < 0 else dtz, move))
                if ranked and (len(ranked) == len(legal) or max(r[0] for r in ranked) == 1):
                    return max(ranked, key=lambda r:r[:2])[2], root
                # Never choose an arbitrary legal move when root TB lacks child DTZ.
                return MCTS(self.evaluator, self.cfg, rng=self.rng).best_move(
                    board, simulations, temperature, add_noise)
            return legal[0], root

        t = self.cfg.temperature if temperature is None else temperature
        if t <= 0:
            i = int(np.argmax(root.N))
        else:
            counts = root.N.astype(np.float64) ** (1.0 / t)
            s = counts.sum()
            if s <= 0:
                i = int(np.argmax(root.N))
            else:
                i = int(self.rng.choice(len(counts), p=counts / s))
        return root.moves[i], root

    @staticmethod
    def visit_policy(root: Node) -> list[tuple[chess.Move, float]]:
        """访问数分布——自对弈的策略训练目标就是它。"""
        total = int(root.N.sum())
        if total <= 0:
            return []
        return [(mv, int(n) / total) for mv, n in zip(root.moves, root.N)]

    @staticmethod
    def root_value(root: Node) -> float:
        """根节点的 Q（行棋方视角）。"""
        denom = int(root.N.sum())
        if denom <= 0:
            return 0.0 if root.terminal_value is None else root.terminal_value
        return float(root.W.sum() / denom)
