"""从 Lichess 月度 PGN 生成训练分片，**只带价值标签**。

和 evals 分片的分工：

    evals 库  ->  Stockfish 的 MultiPV 策略软标签 + 评分折算的 WDL
                  但 FEN 只有 4 个字段，五十步计数和重复次数恒为 0
    PGN       ->  真实对局结果的 WDL + **真实的五十步计数/重复次数**
                  不提供策略标签

**为什么不给策略标签。** 人类实走只能做成 one-hot，而且带着人类的错误。
把它混进策略头会把网络往人类风格拉，和「棋力最强」的目标相悖
（那是 Maia 的目标，不是我们的）。所以这批数据只喂价值头。

实现上不改 96 字节格式：策略概率全部留 0，训练时按「该样本策略标签是否全零」
屏蔽策略损失即可。`--with-policy` 可以打开 one-hot 策略标签用于对照实验。

过滤规则（方案里定的，避开「只筛 2200+」的坑）：
  - 时限为 rapid/classical（Lichess 绝大多数是 bullet/blitz，高分快棋质量未必好）
  - 双方 Elo 都在 [800, 2800] 且**差距 <= 300**
  - Termination 为 Normal，剔除超时/掉线
  - 总步数 >= 20
  - 跳过每局前 8 步（开局重复度过高）
"""
from __future__ import annotations

import argparse
import io
import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import zstandard as zstd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unichess_r.core.encoding import orient_move
from unichess_r.core.moves import move_to_index, move_to_promo_index
from data.record import RECORD_DTYPE, board_to_record

_TC = re.compile(r"^(\d+)\+(\d+)$")


def time_control_ok(tc: str) -> bool:
    """只要 rapid / classical：初始 >= 600 秒，或 >= 180 秒且有加秒。"""
    m = _TC.match((tc or "").strip())
    if not m:
        return False
    base, inc = int(m.group(1)), int(m.group(2))
    return base >= 600 or (base >= 180 and inc >= 2)


def outcome_wdl(result: str, turn: bool) -> tuple[float, float, float] | None:
    """对局结果 -> 当前行棋方视角的 (胜, 和, 负)。"""
    if result == "1/2-1/2":
        return (0.0, 1.0, 0.0)
    if result == "1-0":
        white_won = True
    elif result == "0-1":
        white_won = False
    else:
        return None
    win = (turn == chess.WHITE) == white_won
    return (1.0, 0.0, 0.0) if win else (0.0, 0.0, 1.0)


def process_chunk(args) -> bytes:
    """一批 PGN 文本 -> 记录字节串。在 worker 进程里跑。"""
    text, skip_plies, min_plies, with_policy = args
    out = np.zeros(4096, dtype=RECORD_DTYPE)
    n = 0
    stream = io.StringIO(text)

    while True:
        try:
            game = chess.pgn.read_game(stream)
        except Exception:
            break
        if game is None:
            break
        h = game.headers
        if h.get("Termination") != "Normal":
            continue
        if not time_control_ok(h.get("TimeControl", "")):
            continue
        try:
            we, be = int(h.get("WhiteElo", 0)), int(h.get("BlackElo", 0))
        except ValueError:
            continue
        if not (800 <= we <= 2800 and 800 <= be <= 2800):
            continue
        if abs(we - be) > 300:
            continue
        result = h.get("Result", "*")
        if result not in ("1-0", "0-1", "1/2-1/2"):
            continue

        moves = list(game.mainline_moves())
        if len(moves) < min_plies:
            continue

        board = game.board()
        # 重复次数需要走子历史，所以不能用 copy(stack=False)
        for ply, mv in enumerate(moves):
            if ply >= skip_plies:
                wdl = outcome_wdl(result, board.turn)
                if wdl is not None and n < len(out):
                    rep = 2 if board.is_repetition(3) else (
                        1 if board.is_repetition(2) else 0)
                    policy = None
                    promo = None
                    if with_policy:
                        om = orient_move(mv, board.turn)
                        policy = [(move_to_index(om), 1.0)]
                        promo = move_to_promo_index(om)
                    out[n] = board_to_record(board, policy=policy, wdl=wdl,
                                             promo=promo, rep=rep)
                    n += 1
            board.push(mv)
            if n >= len(out):
                break
        if n >= len(out):
            break

    return out[:n].tobytes()


def iter_chunks(path: Path, games_per_chunk: int, limit_games: int):
    """流式解压 pgn.zst，按「若干局」切块（以空行分隔的 PGN 记录为单位）。"""
    with open(path, "rb") as f:
        reader = zstd.ZstdDecompressor().stream_reader(f)
        text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        buf: list[str] = []
        games = 0
        in_moves = False
        for line in text:
            buf.append(line)
            if line.startswith("1. ") or line.startswith("1."):
                in_moves = True
            elif in_moves and not line.strip():
                games += 1
                in_moves = False
                if games >= games_per_chunk:
                    yield "".join(buf)
                    buf, games = [], 0
                    if limit_games and games >= limit_games:
                        return
        if buf:
            yield "".join(buf)


def main() -> int:
    ap = argparse.ArgumentParser(description="Lichess PGN -> 价值标签分片")
    ap.add_argument("--pgn", default="data/raw/lichess_db_standard_rated_2026-07.pgn.zst")
    ap.add_argument("--out", default="data/shards_pgn")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--games-per-chunk", type=int, default=200)
    ap.add_argument("--shard-records", type=int, default=2_000_000)
    ap.add_argument("--skip-plies", type=int, default=8)
    ap.add_argument("--min-plies", type=int, default=20)
    ap.add_argument("--with-policy", action="store_true",
                    help="额外写入人类实走的 one-hot 策略标签（默认不写，见模块注释）")
    ap.add_argument("--max-records", type=int, default=0, help="产出上限，0 = 不限")
    a = ap.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    itemsize = RECORD_DTYPE.itemsize

    t0 = time.time()
    kept = 0
    shard_id = 0
    shard_bytes = 0
    fh = open(out_dir / f"pgn_{shard_id:04d}.bin", "wb")

    tasks = ((c, a.skip_plies, a.min_plies, a.with_policy)
             for c in iter_chunks(Path(a.pgn), a.games_per_chunk, 0))

    try:
        with mp.Pool(a.workers) as pool:
            for i, blob in enumerate(pool.imap(process_chunk, tasks, chunksize=4), 1):
                if blob:
                    fh.write(blob)
                    shard_bytes += len(blob)
                    kept += len(blob) // itemsize
                if shard_bytes >= a.shard_records * itemsize:
                    fh.close()
                    shard_id += 1
                    shard_bytes = 0
                    fh = open(out_dir / f"pgn_{shard_id:04d}.bin", "wb")
                if i % 500 == 0:
                    el = time.time() - t0
                    print(f"[{el/60:6.1f} 分] 处理 {i*a.games_per_chunk:>10,} 局 | "
                          f"保留 {kept:>12,} 条 | {kept/max(el,1):>8,.0f} 条/秒 | "
                          f"{kept*itemsize/1e9:5.2f} GB | 分片 {shard_id}", flush=True)
                if a.max_records and kept >= a.max_records:
                    print("达到产出上限，停止")
                    break
    finally:
        fh.close()

    last = out_dir / f"pgn_{shard_id:04d}.bin"
    if last.exists() and last.stat().st_size == 0:
        last.unlink()

    el = time.time() - t0
    print(f"\n完成：保留 {kept:,} 条，{kept*itemsize/1e9:.2f} GB，用时 {el/60:.1f} 分")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
