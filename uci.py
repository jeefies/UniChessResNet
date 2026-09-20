"""UCI 协议适配器。

不实现它就没法用 cutechess-cli 跟 Stockfish 对打，也就测不出 Elo——
而分阶段推进的前提是每阶段都能被量化，所以这个不是可选项。

用法（cutechess-cli）：
    cutechess-cli -engine cmd=./unichess.sh -engine cmd=stockfish ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine.engine import UniChessEngine

NAME = "UniChess"
AUTHOR = "jeefy"


class UciLoop:
    def __init__(self, args):
        self.args = args
        self.engine: UniChessEngine | None = None
        self.board = chess.Board()

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

    def _go(self) -> None:
        eng = self._ensure()
        if self.board.is_game_over(claim_draw=False):
            print("bestmove 0000", flush=True)
            return
        _, _, wdl = eng.evaluate(self.board)
        # UCI 的 cp 分值：用 WDL 折算成一个可读的评估
        score = int(round(300.0 * (float(wdl[0]) - float(wdl[2]))))
        depth = self.args.mcts_sims if self.args.mcts_sims > 0 else 1
        print(f"info depth {depth} score cp {score} "
              f"wdl {int(wdl[0]*1000)} {int(wdl[1]*1000)} {int(wdl[2]*1000)}",
              flush=True)
        mv = eng.play(self.board)
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
            elif cmd == "position":
                self._position(parts)
            elif cmd == "go":
                self._go()
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
                    help=">0 时启用 MCTS，每步模拟次数")
    ap.add_argument("--mcts-batch", type=int, default=128)
    return UciLoop(ap.parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(main())
