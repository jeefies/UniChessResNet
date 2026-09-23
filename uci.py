"""UCI 协议适配器。

不实现它就没法用 cutechess-cli 跟 Stockfish 对打，也就测不出 Elo——
而分阶段推进的前提是每阶段都能被量化，所以这个不是可选项。

**时间管理在这一层，不在引擎里。** `go` 带来的 wtime/btime/movetime/nodes
原先被整条丢掉（旧代码的 `_go()` 连 parts 都不收），每一步都固定搜
`--mcts-sims` 次：短时限必然超时判负，长时限则把剩下的时间白白扔掉。
这里负责把 `go` 的时间参数折成「这一步可以花多少墙钟 + 最多搜多少次」，
两个一起交给 `UniChessEngine.play(sims=..., deadline=...)`。

**墙钟是主，模拟次数是辅。** 只发模拟次数的那版实测就翻车了：换算要乘一个
nps（每秒模拟数），而 nps 取决于设备、网络规模、这一步能复用多少搜索树，
远端 CPU 上实测 11 而初值猜的是 150，于是第一步搜了 15.9 秒——分给它的
只有 2.8 秒。nps 要好几步才收敛，第一步则根本没有实测值，所以它只能当
兜底上限，真正咬住时限的必须是搜索内部的墙钟截止。

nps 仍然要估，用来把 sims 上限放在合理量级：每步结束后用
`search_info()['sims'] / 实际耗时` 更新，首次实测直接取代初值。

用法（cutechess-cli）：
    cutechess-cli -engine cmd=./unichess.sh -engine cmd=stockfish ...
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unichess_r.engine.engine import UniChessEngine

NAME = "UniChess"
AUTHOR = "jeefy"

# ---------- 时间管理常数 ----------
MOVE_OVERHEAD_MS = 60.0    # 留给进程通信与解码的余量。超时判负比走弱一步贵太多。
MOVES_TO_GO = 28.0         # 未给 movestogo 时，假设这盘还要走多少手
MAX_TIME_FRACTION = 0.4    # 单步最多花掉剩余时间的这个比例，绝不一次梭哈
INC_FRACTION = 0.75        # 加秒只敢用掉一部分，留一点对冲 nps 估计误差
MIN_SIMS = 8               # 再急也要搜这么多次，否则还不如直接走网络直出
DEFAULT_NPS = 150.0        # 首次搜索前对「每秒多少次模拟」的初始猜测
NPS_SMOOTH = 0.3           # 实测速率进入估计的权重（首次实测直接取代初值）
NPS_SAFETY = 4.0           # 计时赛的模拟数上限放宽到估计值的这么多倍，让 deadline 去咬

_GO_INTS = ("wtime", "btime", "winc", "binc", "movestogo",
            "movetime", "nodes", "depth", "mate")


def _parse_go(parts: list[str]) -> dict:
    """解析 `go` 的参数。无法识别的 token 跳过，不因此拒绝整条命令。"""
    out: dict = {"infinite": False, "ponder": False}
    i = 1
    while i < len(parts):
        tok = parts[i]
        if tok in ("infinite", "ponder"):
            out[tok] = True
            i += 1
        elif tok == "searchmoves":
            break              # 不支持限定候选着法；其后全是着法，解析到此为止
        elif tok in _GO_INTS and i + 1 < len(parts):
            try:
                out[tok] = int(parts[i + 1])
            except ValueError:
                pass
            i += 2
        else:
            i += 1
    return out


def _score_cp(q: float) -> int:
    """Q（= P胜 - P负，取值 [-1, 1]）折算成 UCI 的 cp 分值。

    用 LC0 的标定曲线而不是原先的线性 `300 * q`：线性缩放下 Q=0.90 和
    Q=0.99 只差 27cp，而这两个局面一个是优势一个是已经赢定，裁判程序的
    认输/判和阈值根本分不开它们。tan 曲线在两端会迅速拉开。
    """
    q = max(-0.9999, min(0.9999, q))
    cp = 111.714640912 * math.tan(1.5620688421 * q)
    return int(max(-12000.0, min(12000.0, round(cp))))


class UciLoop:
    def __init__(self, args):
        self.args = args
        self.engine: UniChessEngine | None = None
        self.board = chess.Board()
        self.nps = DEFAULT_NPS          # 实测的每秒模拟数，每步更新
        self._nps_seen = False          # 是否已有实测值（决定首次要不要平滑）

    def _ensure(self) -> UniChessEngine:
        if self.engine is None:
            self.engine = UniChessEngine(
                self.args.ckpt, device=self.args.device,
                syzygy_path=self.args.syzygy, book_path=self.args.book,
                temperature=self.args.temperature,
                mcts_sims=self.args.mcts_sims, mcts_batch=self.args.mcts_batch)
        return self.engine

    def _position(self, parts: list[str]) -> None:
        if len(parts) < 2:
            return
        if parts[1] == "startpos":
            self.board = chess.Board()
            rest = parts[2:]
        elif parts[1] == "fen":
            # fen 后面是 6 个字段，再往后可能跟 moves
            idx = parts.index("moves") if "moves" in parts else len(parts)
            self.board = chess.Board(" ".join(parts[2:idx]))
            rest = parts[idx:]
        else:
            return
        if rest and rest[0] == "moves":
            for u in rest[1:]:
                try:
                    self.board.push_uci(u)
                except ValueError:
                    print(f"info string 忽略非法走法 {u}", flush=True)

    # ---------- 时间管理 ----------

    def _budget(self, go: dict) -> tuple[int, float | None, str]:
        """把 go 的参数折成 (模拟次数上限, 墙钟预算毫秒, 理由)。

        **计时赛以墙钟为准，模拟次数只是兜底上限。** 反过来做过一版，
        实测立刻暴露了问题：把毫秒换算成模拟次数要乘一个 nps，而 nps 在
        远端 CPU 上是 11 而初值猜的是 150——第一步就搜了 15.9 秒，
        而分给它的时间只有 2.8 秒。nps 靠实测平滑要好几步才收敛，第一步
        永远没有实测值可用，所以单靠它封顶在任何设备上都不安全。
        给出的 sims 因此故意放宽到估计值的 NPS_SAFETY 倍，让 deadline 去咬。

        `go nodes` / `go infinite` 不返回墙钟：前者 UCI 要求精确按节点数算，
        后者本来就没有时限。
        """
        base = self.args.mcts_sims
        if base <= 0:
            return 0, None, "network"   # 没开 MCTS，时间参数无从施力

        if go.get("nodes"):
            return max(MIN_SIMS, int(go["nodes"])), None, "nodes"
        if go.get("infinite") or go.get("ponder"):
            # 搜索是同步的，没法被 stop 打断，所以 infinite 不能真的无限，
            # 否则这个进程再也不会回话。给一个大但有限的预算。
            return max(base * 8, MIN_SIMS), None, "infinite"

        ms = go.get("movetime")
        if ms is None:
            white = self.board.turn == chess.WHITE
            remain = go.get("wtime" if white else "btime")
            if remain is None:
                return base, None, "fixed"   # 没给任何时间信息 -> 沿用 --mcts-sims
            inc = go.get("winc" if white else "binc") or 0
            mtg = go.get("movestogo") or 0
            horizon = float(mtg) if mtg > 0 else MOVES_TO_GO
            ms = remain / horizon + inc * INC_FRACTION
            # 上限是硬的：估计错了也只是这一步想得浅，超时则是直接判负。
            ms = min(ms, remain * MAX_TIME_FRACTION)
        ms = max(float(ms) - MOVE_OVERHEAD_MS, 1.0)
        if self._nps_seen and ms / 1000.0 * self.nps < 1.0:
            # 实测速率说这点时间连一次模拟都跑不完。硬搜只会超时，
            # 不如直接走网络直出——它是一次前向，快一个数量级。
            return 0, ms, "too-fast"
        sims = max(MIN_SIMS, int(ms / 1000.0 * self.nps * NPS_SAFETY))
        return sims, ms, "time"

    def _report(self, eng: UniChessEngine, sims: int, budget_ms: float | None,
                why: str, elapsed: float) -> None:
        """打一行 info，顺便用这一步的实测速率校准 nps。"""
        info = eng.search_info()
        done = info["sims"]
        if done > 0 and elapsed > 0.02:
            obs = done / elapsed
            # 第一次实测直接取代初值，不做平滑：DEFAULT_NPS 只是个猜测，
            # 拿它跟实测加权平均等于让一个纯猜的数拖着估计走好几步才收敛。
            # 之后才用指数平滑压掉单步抖动（缓存冷、对手一手把树打空）。
            self.nps = obs if not self._nps_seen else \
                (1 - NPS_SMOOTH) * self.nps + NPS_SMOOTH * obs
            self._nps_seen = True

        extra = ""
        q = info["q"]
        if q is None:
            # 残局表 / 开局书 / 网络直出：没有搜索出来的 Q，退回网络的 WDL
            _, _, wdl = eng.evaluate(self.board)
            q = float(wdl[0]) - float(wdl[2])
            nodes = 1
            extra = f" wdl {int(wdl[0]*1000)} {int(wdl[1]*1000)} {int(wdl[2]*1000)}"
        else:
            nodes = max(info["nodes"], 1)

        ms = max(int(elapsed * 1000), 1)
        line = (f"info depth {max(info['depth'], 1)} score cp {_score_cp(q)}"
                f"{extra} nodes {nodes} nps {int(nodes / max(elapsed, 1e-6))} "
                f"time {ms}")
        pv = " ".join(m.uci() for m in info["pv"])
        if pv:
            line += f" pv {pv}"
        print(line, flush=True)
        cap = "-" if budget_ms is None else f"{budget_ms:.0f}ms"
        print(f"info string source={eng.last_source} why={why} "
              f"sims={done}/{sims} wall={elapsed*1000:.0f}/{cap} "
              f"reused={int(info['reused'])} cut={int(info['stopped_early'])} "
              f"nps_est={self.nps:.0f}", flush=True)

    def _go(self, parts: list[str]) -> None:
        eng = self._ensure()
        if self.board.is_game_over(claim_draw=False):
            print("bestmove 0000", flush=True)
            return
        go = _parse_go(parts)
        sims, budget_ms, why = self._budget(go)

        t0 = time.perf_counter()
        deadline = None if budget_ms is None else t0 + budget_ms / 1000.0
        mv = eng.play(self.board, sims=sims, deadline=deadline)
        elapsed = time.perf_counter() - t0

        # info 放在搜索之后：分值取搜索后的根节点 Q，比搜索前的网络直出准，
        # 而且省掉旧代码里那次纯为了打印而跑的额外前向。
        self._report(eng, sims, budget_ms, why, elapsed)
        print(f"bestmove {mv.uci()}", flush=True)

    def run(self) -> int:
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            cmd = parts[0]

            if cmd == "uci":
                print(f"id name {NAME}")
                print(f"id author {AUTHOR}")
                print("option name Temperature type string default 0.0")
                print("uciok", flush=True)
            elif cmd == "isready":
                self._ensure()
                print("readyok", flush=True)
            elif cmd == "ucinewgame":
                self.board = chess.Board()
                # 不清树的话，上一局的搜索树会被接到新局面上（engine 那边有
                # EPD 校验兜底，但没理由把兜底当常规路径用）。
                if self.engine is not None:
                    self.engine.reset_search()
                self.nps = DEFAULT_NPS
                self._nps_seen = False
            elif cmd == "position":
                self._position(parts)
            elif cmd == "go":
                self._go(parts)
            elif cmd == "setoption":
                if "Temperature" in parts:
                    try:
                        self.args.temperature = float(parts[-1])
                        if self.engine is not None:
                            self.engine.temperature = self.args.temperature
                    except ValueError:
                        pass
            elif cmd in ("quit", "stop"):
                if cmd == "quit":
                    return 0
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--syzygy", default=None)
    ap.add_argument("--book", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--mcts-sims", type=int, default=0,
                    help=">0 时启用 MCTS。这是「没有时间信息时」的每步模拟次数；"
                         "go 带了 wtime/btime/movetime/nodes 时以那边为准")
    ap.add_argument("--mcts-batch", type=int, default=128)
    return UciLoop(ap.parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(main())
