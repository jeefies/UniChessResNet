"""UniChess Server 适配层：把本仓库（ResNet + MCTS）包装成服务端要求的 GameEngine。

把整个 ResNet 仓库目录（或指向它的符号链接）放进 `Server/models/<name>/`，
服务端 `models/__init__.py` 就能发现并驱动它。契约见
`Server/models/__init__.py:12-45`，唯一调用方是 `Server/session_manager.py`。

必须实现的六个方法：setup / human_move / engine_move / state / undo / cleanup。

几个由加载方式决定的设计约束（`Server/models/__init__.py:134-139`）：

* 服务端用 `spec_from_file_location('unichess_server_models.<name>.engine', ...)`
  + `exec_module` 加载本文件。模块名是合成的，没有真实父包，且模型目录
  **不会**被加入 `sys.path`。所以相对导入不可能，必须自己把仓库根目录
  挂到 `sys.path` 上，之后才能 `from unichess_r.engine.engine import ...`。
* 服务端的工作目录是 `Server/`（`Server/README.md:54-58`），不是本仓库根目录。
  因此 config.json 里写的相对路径（如 `runs/stage1/ckpt_00187578.pt`）一律
  相对 `RESNET_ROOT` 解析，见 `_resolve_path`。这也是这个目录可以整体搬走
  或做符号链接的原因。
* `/api/models` 的 `describe_model`（`:189-213`）仅为上报状态就会 import 本模块，
  所以模块顶层**只**导入标准库和 chess：torch / numpy / 仓库内部模块全部延迟到
  `_ensure_backend()`。否则服务启动就要吃掉 torch 的导入耗时，而缺 checkpoint
  之类的问题会让整个模型被报成 `error`。

引擎代码都在 `unichess_r/` 包里（Transformer 是 `unichess_t/`），两个模型同进程
挂载时不会再抢 core / engine / model / search 这些顶层名字。
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any

import chess

# 仓库根目录：所有相对路径的锚点（服务端 cwd 是 Server/，不能依赖 cwd）
RESNET_ROOT = Path(__file__).resolve().parent
# engine/engine.py:21 自己也做了一次无保护的 insert；这里加判断避免重复堆积
if str(RESNET_ROOT) not in sys.path:
    sys.path.insert(0, str(RESNET_ROOT))

logger = logging.getLogger(__name__)

# 延迟导入的后端句柄。cleanup() 需要在 torch 已加载时调 empty_cache，
# 所以 torch 存成模块全局而不是局部变量。
_torch: Any = None
_np: Any = None
_UniChessEngine: Any = None
_MCTS: Any = None
_MCTSConfig: Any = None

# 共享权重：同一份 (ckpt, device, half, syzygy, book, book_plies) 只加载一次。
# 服务端最多同时 4 个会话（MAX_SESSIONS），它们共用同一个 UniChessEngine，
# 各自只持有自己的 MCTS 和搜索树。
_SHARED_ENGINES: dict[tuple, Any] = {}
_SHARED_LOCK = threading.Lock()


def _resolve_path(value: str | Path | None) -> str | None:
    """相对路径按 RESNET_ROOT 解析，绝对路径原样返回；空值返回 None。

    服务端从 `Server/` 启动，若直接把 `runs/stage1/ckpt.pt` 交给 torch.load，
    会解析到 `Server/runs/...` 而不是本仓库。
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    # Windows 下 PurePath("/home/x").is_absolute() 为 False（缺盘符），
    # 但从 Linux 配置里拷来的 POSIX 绝对路径不该被拼到 RESNET_ROOT 后面，
    # 也不该被改写成反斜杠形式，所以绝对路径一律原样返回。
    if Path(text).is_absolute() or text.startswith(("/", "\\")):
        return text
    return str(RESNET_ROOT / text)


def _ensure_backend() -> None:
    """真正需要推理时才导入 torch / numpy / 仓库内部模块。

    只导入 `unichess_r.engine.engine` 和 `unichess_r.search.mcts`：这两条链路只依赖
    torch + python-chess + numpy。绝不导入 `unichess_r.model.dataset` / `train*`
    ——它们需要本 checkout 里缺失的 `data/record.py`。
    """
    global _torch, _np, _UniChessEngine, _MCTS, _MCTSConfig
    if _UniChessEngine is not None:
        return
    import numpy as np
    import torch

    # 包名是 unichess_r（与 Transformer 的 unichess_t 不再同名），直接导入即可；
    # 历史上这里有一段 150 行的 sys.modules 隔离导入，改包名后已删除。
    with _SHARED_LOCK:
        if _UniChessEngine is not None:
            return
        from unichess_r.engine.engine import UniChessEngine
        from unichess_r.search.mcts import MCTS, MCTSConfig

        _torch = torch
        _np = np
        _UniChessEngine = UniChessEngine
        _MCTS = MCTS
        _MCTSConfig = MCTSConfig


def _resolve_device(device: str) -> str:
    """"auto" 时按 CUDA 可用性决定；显式 cuda/cpu 原样透传。

    UniChessEngine.__init__ 里是裸 `torch.device(device)`（engine/engine.py:34），
    没有任何自动探测，必须由调用方决定。做 auto 也让契约测试能在纯 CPU 机器上跑。
    """
    text = (str(device) or "auto").strip().lower()
    if text in ("", "auto"):
        _ensure_backend()
        try:
            return "cuda" if _torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return text


def get_shared_engine(
    ckpt: str,
    device: str,
    half: bool,
    syzygy_path: str | None,
    book_path: str | None,
    book_plies: int,
) -> Any:
    """按参数取（或首次构造）共享的 UniChessEngine。

    传 `mcts_sims=0`：共享对象永远不持有 MCTS（engine/engine.py:85 只在
    mcts_sims>0 时才构造）。每个 GameEngine 会话自己建 MCTS，这样
    `MCTS.last_metrics` 和搜索树不会在并发会话之间互相踩。
    同理不设 temperature，避免会话之间改到同一个共享属性。
    """
    _ensure_backend()
    key = (ckpt, device, bool(half), syzygy_path, book_path, int(book_plies))
    with _SHARED_LOCK:
        engine = _SHARED_ENGINES.get(key)
        if engine is None:
            engine = _UniChessEngine(
                ckpt,
                device=device,
                syzygy_path=syzygy_path,
                book_path=book_path,
                book_plies=int(book_plies),
                temperature=0.0,
                mcts_sims=0,
                half=bool(half),
            )
            if device == "cuda":
                # channels-last 不在 UniChessEngine 内部做，由调用方补。
                # 出错绝不能影响启动，所以整段包在 try 里。
                try:
                    engine.model.to(memory_format=_torch.channels_last)
                except Exception:
                    logger.warning("channels_last 转换失败，继续使用默认内存布局", exc_info=True)
            _SHARED_ENGINES[key] = engine
        return engine


class GameEngine:
    """ResNet 引擎的服务端适配器（一个实例 = 一局对弈会话）。

    出招优先级与 `UniChessEngine.play`（engine/engine.py:228-237）一致：
    残局表 > 开局书 > MCTS > 网络直出。这里不直接调 `play()`，因为
    `_from_mcts`（:221-226）会丢掉搜索树根节点，无法跨手复用；本类自己
    保存 root 并用 `MCTS.advance_root` 前进。

    注意 temperature 只作用于 MCTS 的取样：网络直出走的是共享 UniChessEngine，
    其 temperature 固定为 0（argmax），不会为了某个会话去改共享属性。
    """

    # 服务端据此如实上报 /api/models 状态（Server/models/__init__.py:181,207,237）
    IMPLEMENTED: bool = True
    NOT_IMPLEMENTED_REASON: str = ""

    def __init__(
        self,
        ckpt: str = "runs/stage1/ckpt_00187578.pt",
        device: str = "auto",
        mcts_sims: int = 800,
        mcts_batch: int = 128,
        syzygy_path: str | None = None,
        book_path: str | None = None,
        book_plies: int = 10,
        half: bool | None = None,
        temperature: float = 0.0,
        c_puct_init: float = 1.8,
        fpu_reduction: float = 0.2,
        tablebase_pieces: int = 5,
        root_min_visits: int = 1,
        claim_draw: bool = True,
        **kwargs: Any,
    ):
        """kwargs 里的未知键一律忽略。

        config.json 的预设允许带 `description`（Transformer 的预设就有），
        服务端把整个预设 dict 原样 `engine_cls(**kwargs)` 展开
        （Server/models/__init__.py:242-243），所以必须容忍多余键。
        Transformer 专有的 `precision` 也会落到这里被安静吞掉——
        ResNet 对应的开关是 `half`，不要臆造 precision 参数。
        """
        self.ckpt = _resolve_path(ckpt)
        self.device = _resolve_device(device)
        self.syzygy_path = _resolve_path(syzygy_path)
        self.book_path = _resolve_path(book_path)
        self.book_plies = int(book_plies)
        # engine/engine.py:60 的默认逻辑；这里提前定下来，
        # 好让共享缓存的 key 是确定值而不是 None。
        self.half = (self.device == "cuda") if half is None else bool(half)
        self.mcts_sims = int(mcts_sims)
        self.mcts_batch = int(mcts_batch)
        self.temperature = float(temperature)
        self.extra_kwargs = dict(kwargs)
        if kwargs:
            logger.debug("GameEngine 忽略未知预设参数: %s", sorted(kwargs))

        self.engine = get_shared_engine(
            ckpt=self.ckpt,
            device=self.device,
            half=self.half,
            syzygy_path=self.syzygy_path,
            book_path=self.book_path,
            book_plies=self.book_plies,
        )

        # 每会话独立的 MCTS：自己的 cfg、自己的 numpy Generator。
        # UniChessEngine.rng 是 random.Random（engine/engine.py:63），
        # 不是 np.random.Generator，不能当 MCTS(rng=...) 用。
        self.mcts = None
        if self.mcts_sims > 0:
            self.mcts = _MCTS(
                self.engine.evaluate_batch,
                cfg=_MCTSConfig(
                    simulations=self.mcts_sims,
                    batch_size=self.mcts_batch,
                    c_puct_init=float(c_puct_init),
                    fpu_reduction=float(fpu_reduction),
                    temperature=self.temperature,
                    tablebase_pieces=int(tablebase_pieces),
                    root_min_visits=int(root_min_visits),
                    # 人机对弈要把三次重复/50 步当和棋算进搜索，
                    # 否则引擎会在已经和了的局面里继续"找赢"。
                    # UniChessEngine 自己没开，由调用方决定。
                    claim_draw=bool(claim_draw),
                ),
                tablebase=self.engine.tablebase,
                rng=_np.random.default_rng(),
            )

        # ---- 每会话状态 ----
        self._lock = threading.RLock()
        self.board = chess.Board()
        self.san_history: list[str] = []
        self.root = None                      # MCTS 搜索树根，跨手复用
        self.last_source: str | None = None   # 上一步引擎走法的来源
        self.last_engine_ms: int | None = None
        self._eval_cache: tuple[str, dict] | None = None   # 单条 memo，键是 FEN

    # ---------- 契约方法 ----------

    def setup(self, fen: str | None = None) -> dict:
        """重置局面。服务层总是传具体 FEN（session_manager.py:115），但按契约也接受 None。"""
        with self._lock:
            self.board = chess.Board(fen) if fen else chess.Board()
            # 从中局 FEN 开局时着法记录为空，这是预期行为
            self.san_history = []
            self.root = None
            self.last_source = None
            self.last_engine_ms = None
            self._eval_cache = None
            return self.state()

    def human_move(self, uci: str) -> dict:
        """只应用人类走法，不思考。

        服务层已经校验过合法性并推进了自己的权威 Board
        （session_manager.py:137-142）；这里抛异常会让它回滚（:143-147）。
        """
        with self._lock:
            move = chess.Move.from_uci(uci)
            if move not in self.board.legal_moves:
                raise ValueError(f'走法 "{uci}" 在引擎当前局面下不合法：{self.board.fen()}')
            self._push(move)
            return self.state()

    def engine_move(self) -> dict:
        """按四级优先级选一步并走掉，返回必含非空 "engine_move" 的 dict。

        服务层只在"未终局且轮到引擎"时调用（session_manager.py:119-122、
        :149-152、:178-181）。终局时这里直接抛错而不是返回 engine_move=None：
        返回 None 会在服务层变成一条"返回值格式错误"的 SessionError
        （:85-90），比直接说清原因难排查得多。
        """
        with self._lock:
            board = self.board
            if board.is_game_over():
                raise RuntimeError(f"对局已结束，无法继续走子：{board.fen()}")
            if not any(board.legal_moves):
                raise RuntimeError(f"当前局面无合法走法：{board.fen()}")

            t0 = time.perf_counter()
            move, source = self._select_move(board)
            elapsed_ms = int(round((time.perf_counter() - t0) * 1000))

            self._push(move)
            self.last_source = source
            self.last_engine_ms = elapsed_ms
            return {
                "engine_move": move.uci(),
                "fen": self.board.fen(),
                "done": self.board.is_game_over(),
                "source": source,
                "engine_ms": elapsed_ms,
            }

    def state(self) -> dict:
        """局面快照。

        服务层会用自己的权威 Board 覆盖 fen / legal_moves / is_game_over /
        engine_white（session_manager.py:156-166），其余字段原样透给前端。
        这里额外给的 last_move / san_history / eval / source / engine_ms /
        in_check 都是前端已支持的可选字段（Server/static/index.html:246、
        :186-189、:291-310、:318、:319、:234-241），缺失时前端自动降级。
        """
        with self._lock:
            payload: dict[str, Any] = {
                "fen": self.board.fen(),
                # done 前端不读，但保持与 Transformer 适配器对称
                "done": self.board.is_game_over(),
                "san_history": list(self.san_history),
                "in_check": self.board.is_check(),
            }
            if self.board.move_stack:
                payload["last_move"] = self.board.move_stack[-1].uci()
            if self.last_source is not None:
                payload["source"] = self.last_source
            if self.last_engine_ms is not None:
                payload["engine_ms"] = self.last_engine_ms
            evaluation = self._eval_payload()
            if evaluation is not None:
                payload["eval"] = evaluation
            return payload

    def undo(self) -> dict:
        """悔棋：能退两步就退两步（人类 + 引擎应答），否则退一步。

        服务层在调用本方法之后会从自己的 Board 上弹掉恰好 2 步
        （session_manager.py:168-174），两边必须对齐。
        """
        with self._lock:
            popped = 0
            for _ in range(2):
                if self.board.move_stack:
                    self.board.pop()
                    popped += 1
            if popped:
                del self.san_history[len(self.san_history) - popped:]
            # 树根和"上一步引擎信息"都已失效
            self.root = None
            self.last_source = None
            self.last_engine_ms = None
            self._eval_cache = None
            return self.state()

    def cleanup(self) -> None:
        """释放会话级资源。

        **故意不关 self.engine.tablebase / self.engine.book，也不丢 model**：
        它们属于跨会话共享的 UniChessEngine，别的活跃会话还在用。
        单所有者的场景可以 close 残局表，但这里的 engine 被多个会话共享。

        本方法可能在**构造未完成**的实例上被调用：create_engine 在 setup
        抛异常后会兜底调 cleanup（session_manager.py:216-225），此时
        self.board / self.mcts 等属性可能还不存在，所以一律走 getattr。
        """
        lock = getattr(self, "_lock", None)
        try:
            if lock is not None:
                lock.acquire()
            self.root = None
            self.mcts = None
            self._eval_cache = None
        finally:
            if lock is not None:
                lock.release()
        if _torch is not None:
            try:
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            except Exception:
                logger.debug("empty_cache 失败，忽略", exc_info=True)

    # ---------- 内部实现 ----------

    def _push(self, move: chess.Move) -> None:
        """推进局面：SAN 必须在 push 之前算，随后同步前进搜索树根。"""
        san = self.board.san(move)
        self.board.push(move)
        self.san_history.append(san)
        if self.root is not None:
            try:
                self.root = _MCTS.advance_root(self.root, move)
            except Exception:
                self.root = None
        self._eval_cache = None

    def _select_move(self, board: chess.Board) -> tuple[chess.Move, str]:
        """四级优先级链：残局表 > 开局书 > MCTS > 网络直出。

        残局表和开局书的候选必须自己校验合法性——`play()` 内部就是这么做的
        （engine/engine.py:235），直接调 `_from_*` 就得把这层校验补回来。
        """
        candidate = self.engine._from_tablebase(board)
        if candidate is not None and candidate in board.legal_moves:
            return candidate, "tablebase"

        candidate = self.engine._from_book(board)
        if candidate is not None and candidate in board.legal_moves:
            return candidate, "book"

        if self.mcts is not None:
            try:
                candidate, root = self.mcts.best_move(
                    board,
                    simulations=self.mcts_sims,
                    temperature=self.temperature,
                    root=self.root,
                )
            except Exception:
                logger.warning("MCTS 搜索失败，回退到网络直出", exc_info=True)
                candidate, root = None, None
            if candidate is not None and candidate in board.legal_moves:
                self.root = root
                # last_metrics 只在 search() 跑过之后才存在（search/mcts.py:299）
                metrics = getattr(self.mcts, "last_metrics", {})
                logger.debug("MCTS sims=%s metrics=%s", self.mcts_sims, metrics)
                return candidate, "mcts"
            self.root = None

        # _from_network 直接索引 list(board.legal_moves)，空着法列表会炸；
        # 调用方已在 engine_move 里做过终局检查（同 uci.py:61-63 的做法）。
        return self.engine._from_network(board), "network"

    def _eval_payload(self) -> dict | None:
        """WDL 评估，转成**白方视角**并标 pov="white"。

        前端在 `e.pov !== 'white' && !S.engine_white` 时会交换 win/loss
        （Server/static/index.html:294）；自己转成白方视角并声明 pov
        就彻底绕开了这个歧义。

        终局或无合法走法时整个字段省略（前端会隐藏评估条，:307-310）；
        推理异常也只降级不抛，否则 /api/games/{id}/state 会变成 500。
        """
        board = self.board
        if board.is_game_over() or not any(board.legal_moves):
            return None
        fen = board.fen()
        cached = self._eval_cache
        if cached is not None and cached[0] == fen:
            return cached[1]
        try:
            _, _, wdl = self.engine.evaluate(board)
            win, draw, loss = float(wdl[0]), float(wdl[1]), float(wdl[2])
        except Exception:
            logger.warning("WDL 评估失败，本次 state() 不带 eval", exc_info=True)
            return None
        # evaluate() 返回的是行棋方视角（engine/engine.py:204），黑方走时交换
        if board.turn == chess.BLACK:
            win, loss = loss, win
        payload = {"win": win, "draw": draw, "loss": loss, "pov": "white"}
        self._eval_cache = (fen, payload)
        return payload
