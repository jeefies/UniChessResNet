from __future__ import annotations

import math
from collections import defaultdict


def wilson_lower(successes: int, trials: int, z: float = 1.96) -> float:
    if trials <= 0:
        return 0.0
    p = successes / trials
    d = 1.0 + z * z / trials
    centre = p + z * z / (2.0 * trials)
    spread = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))
    return max(0.0, (centre - spread) / d)


def paired_sign_test(scores: list[float]) -> dict[str, float | int]:
    positive = sum(score > 0.5 for score in scores)
    negative = sum(score < 0.5 for score in scores)
    decisive = positive + negative
    tail = sum(math.comb(decisive, k) for k in range(positive, decisive + 1)) if decisive else 0
    mean = sum(scores) / len(scores) if scores else 0.0
    return {"pairs": len(scores), "score": mean,
            "wilson_lower95": wilson_lower(sum(score > 0.5 for score in scores), len(scores)),
            "positive_pairs": positive, "negative_pairs": negative,
            "decisive_pairs": decisive, "paired_sign_p": tail / (2.0 ** decisive) if decisive else 1.0}


def paired_game_scores(games: list[dict]) -> list[float]:
    """Convert mirrored games into one score per starting position."""
    grouped: dict[object, list[dict]] = defaultdict(list)
    for game in games:
        if game.get("result") in ("1-0", "0-1", "1/2-1/2"):
            grouped[game.get("pair")].append(game)
    scores = []
    for group in grouped.values():
        if len(group) != 2 or len({bool(g.get("candidate_white")) for g in group}) != 2:
            continue
        values = []
        for game in group:
            white_score = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[game["result"]]
            values.append(white_score if game.get("candidate_white") else 1.0 - white_score)
        scores.append(sum(values) / len(values))
    return scores


def reliability(games: list[dict]) -> dict[str, float | int]:
    total = len(games)
    completed = sum(game.get("result") in ("1-0", "0-1", "1/2-1/2") for game in games)
    unknown = total - completed
    illegal = sum(game.get("termination") == "ILLEGAL_MOVE" for game in games)
    interrupted = sum(game.get("termination") in ("INTERRUPTED", "PLY_LIMIT") for game in games)
    return {"games": total, "completed_games": completed, "unknown_games": unknown,
            "illegal_games": illegal, "interrupted_or_cutoff": interrupted,
            "completion_rate": completed / total if total else 0.0,
            "reliable": bool(total and unknown == 0 and illegal == 0)}
