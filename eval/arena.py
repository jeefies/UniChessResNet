"""对局评测台：两个 UCI 引擎对打，算 Elo 差和 SPRT。

为什么自己写而不用 cutechess-cli：本机没有免密 sudo，装不了。
python-chess 的 UCI 封装已经验证可用（Phase A 里跟 Stockfish 打过完整对局），
自己实现 Elo/SPRT 的数学部分也就几十行。

三条评测纪律（方案里定的）：
  1. **同一开局双方各执一次先手**，否则先手优势会污染结果
  2. **用 SPRT 判显著性**——200 局的 55% 胜率在统计上什么都说明不了
  3. 报 Elo 时必须带误差棒
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import chess
import chess.engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------- 统计

def score_to_elo(score: float) -> float:
    """胜率 -> Elo 差。"""
    score = min(max(score, 1e-9), 1 - 1e-9)
    return -400.0 * math.log10(1.0 / score - 1.0)


def elo_to_score(elo: float) -> float:
    return 1.0 / (1.0 + 10 ** (-elo / 400.0))


ELO_INF = float("inf")


def elo_with_error(w: int, d: int, l: int, confidence: float = 0.95
                   ) -> tuple[float, float, float]:
    """返回 (Elo 差, 下界, 上界)。误差来自每局得分的样本方差。

    全胜/全负时 Elo 在数学上是无穷，返回 inf 而不是一个看着像真值的大数——
    「+3600」会让人误以为测出了具体差距，实际上只说明差距大到无法估计。
    """
    n = w + d + l
    if n == 0:
        return 0.0, -ELO_INF, ELO_INF
    s = (w + 0.5 * d) / n
    var = (w * (1 - s) ** 2 + d * (0.5 - s) ** 2 + l * s ** 2) / n
    se = math.sqrt(var / n)
    z = 1.959963985 if abs(confidence - 0.95) < 1e-6 else 2.575829
    lo_s, hi_s = s - z * se, s + z * se
    elo = ELO_INF if s >= 1.0 else (-ELO_INF if s <= 0.0 else score_to_elo(s))
    lo = -ELO_INF if lo_s <= 0.0 else score_to_elo(lo_s)
    hi = ELO_INF if hi_s >= 1.0 else score_to_elo(hi_s)
    return elo, lo, hi


def fmt_elo(x: float) -> str:
    if x == ELO_INF:
        return "+inf"
    if x == -ELO_INF:
        return "-inf"
    return f"{x:+.1f}"


def sprt_llr(w: int, d: int, l: int, elo0: float, elo1: float) -> float:
    """广义 SPRT 的对数似然比（正态近似）。

    H0: Elo 差 = elo0    H1: Elo 差 = elo1
    """
    n = w + d + l
    if n == 0:
        return 0.0
    s = (w + 0.5 * d) / n
    var = (w * (1 - s) ** 2 + d * (0.5 - s) ** 2 + l * s ** 2) / n
    if var <= 0:
        return 0.0
    s0, s1 = elo_to_score(elo0), elo_to_score(elo1)
    return n * (s1 - s0) * (2 * s - s0 - s1) / (2 * var)


def sprt_verdict(llr: float, alpha: float = 0.05, beta: float = 0.05) -> str:
    upper = math.log((1 - beta) / alpha)
    lower = math.log(beta / (1 - alpha))
    if llr >= upper:
        return "接受 H1（确实更强）"
    if llr <= lower:
        return "接受 H0（没有变强）"
    return "继续（尚不显著）"


def los(w: int, d: int, l: int) -> float:
    """Likelihood of superiority：A 真的比 B 强的概率。"""
    if w + l == 0:
        return 0.5
    return 0.5 * (1 + math.erf((w - l) / math.sqrt(2.0 * (w + l))))


# ---------------------------------------------------------------- 开局集

def make_openings(stockfish: str, count: int, plies: int = 8,
                  max_cp: int = 60, seed: int = 20260908) -> list[str]:
    """生成一批**均势**开局局面，保证对局公平。

    随机走 plies 步，再用 Stockfish 确认评分在 ±max_cp 以内；
    否则这个开局本身就偏向一方，会污染 Elo。
    """
    rng = random.Random(seed)
    eng = chess.engine.SimpleEngine.popen_uci(stockfish)
    eng.configure({"Threads": 2, "Hash": 128})
    out: list[str] = []
    tried = 0
    try:
        while len(out) < count and tried < count * 60:
            tried += 1
            b = chess.Board()
            ok = True
            for _ in range(plies):
                moves = list(b.legal_moves)
                if not moves:
                    ok = False
                    break
                b.push(rng.choice(moves))
            if not ok or b.is_game_over(claim_draw=False):
                continue
            info = eng.analyse(b, chess.engine.Limit(nodes=60_000))
            sc = info["score"].white()
            if sc.mate() is not None or abs(sc.score()) > max_cp:
                continue
            out.append(b.fen())
    finally:
        eng.quit()
    return out


# ---------------------------------------------------------------- 对局

@dataclass
class EngineSpec:
    cmd: list[str]
    name: str
    options: dict = field(default_factory=dict)
    nodes: int | None = None
    movetime: float | None = 1.0

    def limit(self) -> chess.engine.Limit:
        if self.nodes:
            return chess.engine.Limit(nodes=self.nodes)
        return chess.engine.Limit(time=self.movetime or 1.0)


def _play_pair(args) -> tuple[int, int, int, list[str]]:
    """同一开局打两局（双方各执一次白），返回 A 视角的 (胜, 和, 负)。"""
    fen, a_spec, b_spec, max_plies = args
    w = d = l = 0
    notes: list[str] = []
    ea = chess.engine.SimpleEngine.popen_uci(a_spec.cmd)
    eb = chess.engine.SimpleEngine.popen_uci(b_spec.cmd)
    try:
        if a_spec.options:
            ea.configure(a_spec.options)
        if b_spec.options:
            eb.configure(b_spec.options)
        for a_is_white in (True, False):
            board = chess.Board(fen)
            plies = 0
            while not board.is_game_over(claim_draw=False) and plies < max_plies:
                a_turn = (board.turn == chess.WHITE) == a_is_white
                eng, spec = (ea, a_spec) if a_turn else (eb, b_spec)
                try:
                    res = eng.play(board, spec.limit())
                except Exception as e:
                    notes.append(f"{spec.name} 异常: {type(e).__name__}")
                    res = None
                if res is None or res.move is None or res.move not in board.legal_moves:
                    notes.append(f"{spec.name} 走出非法/空走法 @ {board.fen()}")
                    # 判该方负
                    if a_turn:
                        l += 1
                    else:
                        w += 1
                    break
                board.push(res.move)
                plies += 1
            else:
                r = board.result(claim_draw=False)
                if r == "1/2-1/2" or r == "*":
                    d += 1
                elif (r == "1-0") == a_is_white:
                    w += 1
                else:
                    l += 1
    finally:
        ea.quit()
        eb.quit()
    return w, d, l, notes


def run_match(a: EngineSpec, b: EngineSpec, openings: list[str], *,
              workers: int = 4, max_plies: int = 300,
              elo0: float = 0.0, elo1: float = 25.0) -> dict:
    tasks = [(fen, a, b, max_plies) for fen in openings]
    W = D = L = 0
    t0 = time.time()
    all_notes: list[str] = []

    with mp.Pool(workers) as pool:
        for i, (w, d, l, notes) in enumerate(pool.imap_unordered(_play_pair, tasks), 1):
            W += w; D += d; L += l
            all_notes.extend(notes)
            n = W + D + L
            if i % 5 == 0 or i == len(tasks):
                elo, lo, hi = elo_with_error(W, D, L)
                llr = sprt_llr(W, D, L, elo0, elo1)
                print(f"[{i}/{len(tasks)} 组] {n} 局  "
                      f"+{W} ={D} -{L}  "
                      f"Elo {fmt_elo(elo)} [{fmt_elo(lo)}, {fmt_elo(hi)}]  "
                      f"LLR {llr:+.2f}  {time.time()-t0:.0f}s", flush=True)

    elo, lo, hi = elo_with_error(W, D, L)
    llr = sprt_llr(W, D, L, elo0, elo1)
    return {
        "a": a.name, "b": b.name,
        "games": W + D + L, "w": W, "d": D, "l": L,
        "score": (W + 0.5 * D) / max(W + D + L, 1),
        "elo": elo, "elo_lo": lo, "elo_hi": hi,
        "llr": llr, "sprt": sprt_verdict(llr), "los": los(W, D, L),
        "notes": all_notes[:20],
        "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="两个 UCI 引擎对打并统计 Elo/SPRT")
    ap.add_argument("--engine-a", required=True, help="引擎 A 命令")
    ap.add_argument("--engine-b", required=True, help="引擎 B 命令")
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--nodes-a", type=int, default=0)
    ap.add_argument("--nodes-b", type=int, default=0)
    ap.add_argument("--movetime", type=float, default=1.0)
    ap.add_argument("--pairs", type=int, default=100, help="开局组数（每组 2 局）")
    ap.add_argument("--openings", default="eval/openings.txt")
    ap.add_argument("--stockfish", default="tools/stockfish")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--elo0", type=float, default=0.0)
    ap.add_argument("--elo1", type=float, default=25.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    op = Path(a.openings)
    if not op.exists():
        print(f"生成 {a.pairs} 个均势开局 ...")
        fens = make_openings(a.stockfish, a.pairs)
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text("\n".join(fens) + "\n")
        print(f"  -> {op}")
    fens = [x for x in op.read_text().splitlines() if x.strip()][:a.pairs]
    print(f"用 {len(fens)} 个开局，每个开局双方各执一次白，共 {2*len(fens)} 局")

    spec_a = EngineSpec(a.engine_a.split(), a.name_a,
                        nodes=a.nodes_a or None, movetime=a.movetime)
    spec_b = EngineSpec(a.engine_b.split(), a.name_b,
                        nodes=a.nodes_b or None, movetime=a.movetime)

    res = run_match(spec_a, spec_b, fens, workers=a.workers,
                    elo0=a.elo0, elo1=a.elo1)
    print()
    print(f"{res['a']} vs {res['b']}")
    print(f"  {res['games']} 局  +{res['w']} ={res['d']} -{res['l']}  "
          f"得分率 {res['score']:.3f}")
    print(f"  Elo 差 {fmt_elo(res['elo'])}  "
          f"95% 区间 [{fmt_elo(res['elo_lo'])}, {fmt_elo(res['elo_hi'])}]")
    if res["elo"] in (ELO_INF, -ELO_INF):
        print("  （一方全胜，Elo 差无法估计——说明对手强度选得不合适，应换更接近的档位）")
    print(f"  LLR {res['llr']:+.2f}  ->  {res['sprt']}")
    print(f"  LOS {100*res['los']:.1f}%")
    if res["notes"]:
        print("  异常:", *res["notes"][:5], sep="\n    ")
    if a.out:
        safe = {k: (None if isinstance(v, float) and math.isinf(v) else v)
                for k, v in res.items()}
        Path(a.out).write_text(json.dumps(safe, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
