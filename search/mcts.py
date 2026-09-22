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
import time
from dataclasses import dataclass, field

import chess
import numpy as np

# 这两个 import 原先写在 priors_from_policy 内部（函数作用域）。必须放在模块顶层：
# 函数作用域的 import 会在**每次调用时**重新经由 sys.modules 按全限定名解析，
# 于是根目录下的 engine.py 做的导入隔离（见 engine.py 的 _ISOLATED_MODULES 一节）
# 对它完全无效——服务端同时挂载的 Transformer 仓库会把 'core.encoding' 占住，
# 调用时拿到的就是对方那套 64 token 编码，无声算错而不是报错。
# 没有循环依赖：core.encoding 只依赖 chess + numpy，core.moves 只依赖 chess，
# 两者都不 import search.*。
from core.encoding import orient_move
from core.moves import move_to_index, move_to_promo_index

# 带墙钟限制时第一批的模拟数。这一批纯粹是为了测速率，所以要小；
# 但太小会让每步都多付一次 _collect 的固定开销，8 是个折中。
DEADLINE_PROBE_SIMS = 8


@dataclass
class MCTSConfig:
    simulations: int = 800
    batch_size: int = 128          # 一次收集多少叶子再批量推理
    c_puct: float = 1.8            # 已废弃：实际用的是下面两个，保留仅为兼容旧构造参数
    c_puct_base: float = 19652.0   # cpuct 随访问数缓慢增长（AlphaZero 的做法）
    c_puct_init: float = 1.8
    dirichlet_alpha: float = 0.3   # 国际象棋用 0.3
    dirichlet_eps: float = 0.25
    fpu_reduction: float = 0.2     # 未访问子节点的先验价值折扣
    temperature: float = 0.0       # 0 = 取访问数最大者
    temp_moves: int = 0            # 已废弃：温度衰减由调用方（autoloop/worker.py）自己控制
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
        """PUCT：argmax( Q + c * P * sqrt(sum_N) / (1 + N) )，其中 c 由
        c_puct_base / c_puct_init 随访问数缓慢增长（**不是** cfg.c_puct）。"""
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
    """把 4096 维策略 + 4 维升变头，映射到该局面的合法走法先验上。

    orient_move / move_to_index / move_to_promo_index 已提升到模块顶层导入，
    原因见文件头部那段注释。
    """
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
        # 上一次搜索实测的「每秒模拟数」。带墙钟限制时用它给第一批定大小——
        # 复用根节点的那种情况没有根前向可以估，只能靠上一步留下的速率。
        self._sim_rate: float | None = None

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
               add_noise: bool = False, root: Node | None = None,
               deadline: float | None = None) -> Node:
        root_reused = root is not None and root.expanded
        t_enter = time.perf_counter()
        self.last_metrics = {'simulations': 0, 'stopped_early': False, 'network_positions': 0, 'network_batches': 0, 'max_depth': 0, 'collisions': 0, 'reused_root': root_reused}
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
        t0 = time.perf_counter()
        # 根节点展开就是一次单局面前向。带墙钟限制时它是循环开始前唯一的
        # 时间样本，用来给第一批定大小（下面）。复用根节点时没跑这次前向。
        t_root = t0 - t_enter if not root_reused else 0.0
        stopped_early = False
        while done < sims:
            want = min(cfg.batch_size, sims - done)
            if deadline is not None:
                now = time.perf_counter()
                if now >= deadline:
                    stopped_early = True
                    break
                left = deadline - now
                # **批量大小本身必须受时限约束。** 只在批次末尾比对 deadline
                # 远远不够：那样最少也要跑完一整批，而 CPU 上一批 128 次模拟
                # 就是十几秒，一批就能把整盘的时间烧光。实测过这个坑——
                # 2.0s 的预算被一个 8 次模拟的「小」探测批顶到了 2.67s。
                if done > 0:
                    want = max(1, min(want, int(done / max(now - t0, 1e-9) * left)))
                elif self._sim_rate:
                    # 上一步测过速率，直接用（复用根节点时这是唯一可用的估计）
                    want = max(1, min(want, int(self._sim_rate * left)))
                elif t_root > 1e-4:
                    # 全新的树、也没有历史速率：拿刚付掉的那次根前向当尺子。
                    # CPU 上小批量近似线性，GPU 上会严重低估吞吐——但只影响
                    # 第一批，第二批起就换成实测速率了，保守一点不亏。
                    want = min(want, max(1, int(left / t_root)))
                    want = min(want, DEADLINE_PROBE_SIMS)
                else:
                    want = min(want, DEADLINE_PROBE_SIMS)
            leaves, terminal_sims = self._collect(root, board, want)
            if not leaves and terminal_sims == 0:
                break
            self._evaluate_and_expand(leaves)
            done += len(leaves) + terminal_sims
        # 实际消耗的模拟数（含终局/残局表直接结算的那些）。UCI 侧靠它把
        # 「剩余时间」换算成「模拟次数」——没有实测速率就只能瞎猜一个 nps。
        self.last_metrics['simulations'] = done
        self.last_metrics['stopped_early'] = stopped_early
        spent = time.perf_counter() - t0
        if done > 0 and spent > 1e-6:
            self._sim_rate = done / spent
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
                  add_noise: bool = False, root: Node | None = None,
                  deadline: float | None = None) -> tuple[chess.Move, Node]:
        root = self.search(board, simulations, add_noise=add_noise, root=root,
                           deadline=deadline)
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
                    # 排序键 (对手WDL, 清零, 有向DTZ)，取 max：
                    #   1. -value 最大 = 对手最输 = 我方结果最好，结果永远优先
                    #   2. **赢棋时清零着法（推兵/吃子）优先**。这一条原先漏了，
                    #      和 engine/engine.py:152 是同一个理由：DTZ 是「距下一次
                    #      清零的步数」，兵残局里随时可以推兵，DTZ 就恒在 2 附近，
                    #      完全不提供梯度——王会原地打转直到 50 步和棋。
                    #      只在赢棋（value < 0，即对手输）时加权：轮到自己要输或
                    #      要和的时候，清零反而重置 50 步计数，把能和的棋走输。
                    #   3. 最后比 DTZ：赢棋取短（-dtz 最大），非赢棋取长。
                    zeroing = 1 if (value < 0 and board.is_zeroing(move)) else 0
                    ranked.append((-value, zeroing, -dtz if value < 0 else dtz, move))
                if ranked and (len(ranked) == len(legal) or max(r[0] for r in ranked) == 1):
                    return max(ranked, key=lambda r: r[:3])[3], root
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
