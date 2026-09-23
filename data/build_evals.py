"""从 Lichess evals 库生成 Stage 1 训练分片。

evals 库（394,669,566 条）每条自带 Stockfish 的多条 PV，所以它同时提供了
value 标签和 policy 软标签——本机再跑 MultiPV 标注不再是关键路径。

只保留 **PV >= 2** 的条目（约占 42%，≈1.65 亿）：单条 PV 只能当 one-hot，
策略信号太弱，而策略头的训练质量比样本数量更要紧。

两个已实测确认的格式要点：
  1. **cp 是白方视角**（黑方行棋 99.8% 升序、白方行棋 100% 降序）
     => 黑方行棋时必须把 cp 取负才能得到「行棋方视角」。
        符号弄反会让网络学着走最差的一手，且要等到测 Elo 才会发现。
  2. FEN 只有 4 个字段（无五十步计数、无回合数），需补 " 0 1"。

已知局限：evals 库不含五十步计数与重复次数，故这两个平面在本批数据里恒为 0。
等 PGN 数据到位后会有真实值。残局阶段有 Syzygy 兜底，影响可控。
"""
from __future__ import annotations

import argparse
import io
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import chess
import numpy as np
import zstandard as zstd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unichess_r.core.encoding import orient_move
from unichess_r.core.moves import move_to_index, move_to_promo_index
from data.annotate import score_to_wdl
from data.record import RECORD_DTYPE, board_to_record

MATE_CP = 30000.0


def _pv_score_cp(pv: dict, white_pov_to_stm: int) -> float | None:
    """取出一条 PV 的分值，转成**行棋方视角**的厘兵。"""
    if "mate" in pv and pv["mate"] is not None:
        m = int(pv["mate"]) * white_pov_to_stm
        return MATE_CP if m > 0 else -MATE_CP
    if "cp" in pv and pv["cp"] is not None:
        return float(pv["cp"]) * white_pov_to_stm
    return None


def process_lines(args) -> bytes:
    """一批 jsonl 文本行 -> 打包好的记录字节串。在 worker 进程里跑。"""
    lines, temperature, min_pv, min_depth = args
    out = np.zeros(len(lines), dtype=RECORD_DTYPE)
    n = 0

    for line in lines:
        try:
            d = json.loads(line)
        except Exception:
            continue
        evals = d.get("evals") or []
        if not evals:
            continue
        best = max(evals, key=lambda e: e.get("depth", 0))
        if best.get("depth", 0) < min_depth:
            continue
        pvs = best.get("pvs") or []
        if len(pvs) < min_pv:
            continue

        fen = d.get("fen")
        if not fen:
            continue
        parts = fen.split()
        if len(parts) == 4:            # evals 库只有 4 个字段
            fen = fen + " 0 1"
        try:
            board = chess.Board(fen)
        except Exception:
            continue
        if not board.is_valid() or board.is_game_over(claim_draw=False):
            continue

        # cp 是白方视角；黑方行棋时取负
        sign = 1 if board.turn == chess.WHITE else -1

        cands: list[tuple[int, int | None, float]] = []
        seen: set[int] = set()
        for pv in pvs:
            uci = (pv.get("line") or "").split(" ", 1)[0]
            if not uci:
                continue
            score = _pv_score_cp(pv, sign)
            if score is None:
                continue
            try:
                # **必须用 board.parse_uci，不能用 chess.Move.from_uci。**
                # 评估库（Lichess）把易位写成「王吃己车」的 e1h1 / e1a1
                # （Chess960 写法）。from_uci 原样返回 Move(e1, h1)，而
                # `mv in board.legal_moves` 对它返回 **True**——python-chess 的
                # is_pseudo_legal 内部会先 _from_chess960 归一化再判定。于是
                # 这个非规范形式一路通过校验，move_to_index 拿到的 to_square
                # 却还是 h1，标签就写到了索引 4*64+7 上。
                # 而推理端（search/mcts.py:priors_from_policy）是对
                # board.legal_moves 里的 e1g1 算索引 4*64+6 去查表的，
                # **那份易位概率永远读不到**。实测已建好的 shards_evals 里
                # 2587 个易位标签有 2585 个是错的形式，占全部策略质量的 1.99%。
                # parse_uci 会先归一化再校验合法性，非法则抛异常。
                mv = board.parse_uci(uci)
            except Exception:
                continue
            om = orient_move(mv, board.turn)
            idx = move_to_index(om)
            if idx in seen:
                continue
            seen.add(idx)
            cands.append((idx, move_to_promo_index(om), score))

        if len(cands) < min_pv:
            continue

        best_cp = max(c[2] for c in cands)
        weights = [math.exp((c[2] - best_cp) / temperature) for c in cands]
        total = sum(weights)
        if total <= 0:
            continue
        # 先截断到前 5 条再归一化。若在 softmax 之后才截断，
        # PV 有 6~10 条时存下来的概率和会小于 1。
        pairs = list(zip(cands, weights))[:5]
        sub = sum(w for _, w in pairs)
        policy = [(c[0], w / sub) for c, w in pairs]

        top = max(cands, key=lambda c: c[2])
        if abs(top[2]) >= MATE_CP:
            wdl = score_to_wdl(None, 1 if top[2] > 0 else -1)
        else:
            wdl = score_to_wdl(top[2], None)

        out[n] = board_to_record(board, policy=policy, wdl=wdl,
                                 promo=top[1], rep=0)
        n += 1

    return out[:n].tobytes()


def iter_batches(path: Path, batch: int, limit: int):
    """流式解压 jsonl.zst，按批产出文本行。"""
    with open(path, "rb") as f:
        reader = zstd.ZstdDecompressor().stream_reader(f)
        text = io.TextIOWrapper(reader, encoding="utf-8")
        buf, total = [], 0
        for line in text:
            buf.append(line)
            total += 1
            if len(buf) >= batch:
                yield buf
                buf = []
            if limit and total >= limit:
                break
        if buf:
            yield buf


def main() -> int:
    ap = argparse.ArgumentParser(description="evals 库 -> 训练分片")
    ap.add_argument("--evals", default="data/raw/lichess_db_eval.jsonl.zst")
    ap.add_argument("--out", default="data/shards_evals")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--batch", type=int, default=4000, help="每个任务的行数")
    ap.add_argument("--shard-records", type=int, default=2_000_000,
                    help="每个分片文件的记录数")
    ap.add_argument("--temperature", type=float, default=90.0,
                    help="策略 softmax 温度，单位厘兵")
    ap.add_argument("--min-pv", type=int, default=2, help="最少 PV 条数")
    ap.add_argument("--min-depth", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="只读前 N 行，0=全部")
    a = ap.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    itemsize = RECORD_DTYPE.itemsize

    t0 = time.time()
    lines_in = kept = 0
    shard_id = 0
    shard_bytes = 0
    fh = open(out_dir / f"evals_{shard_id:04d}.bin", "wb")

    tasks = ((b, a.temperature, a.min_pv, a.min_depth)
             for b in iter_batches(Path(a.evals), a.batch, a.limit))

    with mp.Pool(a.workers) as pool:
        for blob in pool.imap(process_lines, tasks, chunksize=2):
            lines_in += a.batch
            if blob:
                fh.write(blob)
                shard_bytes += len(blob)
                kept += len(blob) // itemsize
            if shard_bytes >= a.shard_records * itemsize:
                fh.close()
                shard_id += 1
                shard_bytes = 0
                fh = open(out_dir / f"evals_{shard_id:04d}.bin", "wb")
            if lines_in % (a.batch * 250) == 0:
                el = time.time() - t0
                print(f"[{el/60:6.1f} 分] 读入 {lines_in:>12,} 行 | "
                      f"保留 {kept:>12,} ({100*kept/max(lines_in,1):4.1f}%) | "
                      f"{lines_in/el:>8,.0f} 行/秒 | "
                      f"{kept*itemsize/1e9:5.2f} GB | 分片 {shard_id}", flush=True)
    fh.close()

    # 删掉可能为空的最后一个分片
    last = out_dir / f"evals_{shard_id:04d}.bin"
    if last.exists() and last.stat().st_size == 0:
        last.unlink()

    el = time.time() - t0
    print(f"\n完成：读入 {lines_in:,} 行，保留 {kept:,} 条 "
          f"({100*kept/max(lines_in,1):.1f}%)，{kept*itemsize/1e9:.2f} GB，"
          f"用时 {el/60:.1f} 分")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
