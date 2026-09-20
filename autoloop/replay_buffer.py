"""Persistent, stratified experience replay for the autonomous learners.

The buffer keeps a bounded long-term reservoir while always admitting a
fresh-window slice.  Selection is deterministic for a row key, weighted by
difficulty, and stratified by the same piece-count stages used by the layered
models.  It stores complete teacher-verified rows so the source bundle may
later expire without destroying the learning sample.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import time
from pathlib import Path

import chess

from autoloop.common import STATE, atomic_json, event, read_bundle, read_json

REGRESSION = '2kq4/3np3/4p3/rP2P3/4Q3/2p1B3/p1B2PP1/3RK3 w - - 0 30'
STAGE_RANGES = ((25, 32), (19, 24), (13, 18), (7, 12), (2, 6))
STAGE_NAMES = ('opening', 'middlegame', 'transition', 'endgame', 'tablebase')
STAGE_QUOTAS = (0.20, 0.20, 0.20, 0.20, 0.20)
SCHEMA = 1
DEFAULT_CAPACITY = 12_000
DEFAULT_FRESH_BUDGET = 4_000
DEFAULT_RESERVOIR_BUDGET = 8_000
MAX_SOURCE_FILES = 128


def position_key(fen: str) -> str:
    return hashlib.sha256(' '.join(fen.split()[:4]).encode()).hexdigest()


def sample_key(row: dict) -> str:
    fen = row['fen']
    return position_key(fen) + ':' + fen.split()[4] + ':' + str(int(row.get('repetition', False)))


def is_holdout(fen: str) -> bool:
    return int(position_key(fen)[:8], 16) % 20 == 0 or fen == REGRESSION


def stage_for_row(row: dict) -> str:
    try:
        pieces = int(row.get('pieces', len(chess.Board(row['fen']).piece_map())))
    except (KeyError, ValueError, TypeError):
        return 'unknown'
    for name, (minimum, maximum) in zip(STAGE_NAMES, STAGE_RANGES):
        if minimum <= pieces <= maximum:
            return name
    return 'unknown'


def _finite(value, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _top_move(row: dict, field: str) -> str | None:
    values = row.get(field) or []
    if not values or not isinstance(values[0], (list, tuple)):
        return None
    return str(values[0][0]) if values[0] else None


def _distribution_gap(left, right) -> float:
    """Jensen-Shannon distance for sparse legal-move distributions."""
    a = {str(move): max(_finite(prob), 0.0) for move, prob in (left or []) if isinstance(move, str)}
    b = {str(move): max(_finite(prob), 0.0) for move, prob in (right or []) if isinstance(move, str)}
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    av = sum(a.values()) or 1.0
    bv = sum(b.values()) or 1.0
    js = 0.0
    for key in keys:
        x, y = a.get(key, 0.0) / av, b.get(key, 0.0) / bv
        m = (x + y) / 2.0
        if x > 0:
            js += 0.5 * x * math.log(x / m)
        if y > 0:
            js += 0.5 * y * math.log(y / m)
    return min(max(js / math.log(2.0), 0.0), 1.0)


def difficulty(row: dict) -> float:
    """Return a bounded priority weight, not a training target."""
    cp_loss = min(max(_finite(row.get('cp_loss_estimate')), 0.0), 800.0) / 800.0
    search_teacher_gap = _distribution_gap(row.get('policy'), row.get('teacher_policy'))
    model_gap = max(_distribution_gap(row.get('policy'), row.get('big_policy')),
                    _distribution_gap(row.get('policy'), row.get('opponent_policy')))
    disagreement = float(bool(_top_move(row, 'policy') and
                              _top_move(row, 'teacher_policy') and
                              _top_move(row, 'policy') != _top_move(row, 'teacher_policy')))
    threat = float(bool(row.get('promotion_threat')))
    result_gap = 0.0
    teacher_wdl = row.get('teacher_wdl')
    result_wdl = row.get('result_wdl')
    if isinstance(teacher_wdl, list) and isinstance(result_wdl, list) and len(teacher_wdl) == len(result_wdl):
        result_gap = min(sum(abs(_finite(a) - _finite(b)) for a, b in zip(teacher_wdl, result_wdl)), 1.0)
    entropy = min(max(_finite(row.get('root_entropy')), 0.0) / 5.0, 1.0)
    explicit = min(max(_finite(row.get('importance'), 1.0), 0.5), 2.0)
    source = str(row.get('source', ''))
    source_bonus = {'model_stockfish': 0.28, 'model_crossplay': 0.16,
                    'big_selfplay': 0.10, 'small_selfplay': 0.04}.get(source, 0.0)
    # Uncertainty alone is deliberately capped; deterministic hashing supplies diversity.
    base = (1.0 + 1.5 * cp_loss + 0.55 * disagreement + 0.70 * search_teacher_gap
            + 0.55 * model_gap + 0.25 * threat + 0.45 * result_gap
            + 0.12 * entropy + source_bonus)
    return min(max(base * explicit, 1.0), 5.0)


def _rank(key: str, weight: float, namespace: str) -> float:
    digest = hashlib.sha256((namespace + ':' + key).encode()).hexdigest()[:16]
    uniform = int(digest, 16) / float(1 << 64)
    return weight * (0.55 + 0.90 * uniform)


def _row_signature(row: dict) -> str:
    fields = ['fen', 'policy', 'teacher_policy', 'teacher_wdl', 'result_wdl',
              'big_policy', 'big_wdl', 'big_verified', 'teacher_cp', 'cp_loss_estimate']
    payload = {key: row[key] for key in fields if key in row}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def replay_version(rows: list[dict]) -> str:
    signatures = sorted(sample_key(row) + ':' + _row_signature(row) for row in rows)
    return hashlib.sha256('\n'.join(signatures).encode()).hexdigest()


class ExperienceReplay:
    """Merge recent bundles into a persistent, stage-balanced reservoir."""

    def __init__(self, role: str, *, state_dir: Path | None = None,
                 capacity: int = DEFAULT_CAPACITY,
                 fresh_budget: int = DEFAULT_FRESH_BUDGET,
                 reservoir_budget: int = DEFAULT_RESERVOIR_BUDGET):
        if role not in ('small', 'big', 'layered'):
            raise ValueError('experience replay role must be small, big, or layered')
        self.role = role
        self.base = Path(state_dir or STATE)
        self.capacity = int(capacity)
        self.fresh_budget = int(fresh_budget)
        self.reservoir_budget = int(reservoir_budget)
        self.state_path = self.base / f'{role}-replay-reservoir.json'

    @staticmethod
    def _valid(row: object) -> bool:
        if not isinstance(row, dict):
            return False
        required = ('fen', 'start', 'history', 'policy', 'teacher_policy', 'teacher_wdl')
        return all(key in row for key in required) and isinstance(row['history'], list)

    def _fresh(self, source_rows: list[dict] | None = None) -> dict[str, tuple[dict, float]]:
        latest: dict[str, tuple[dict, float]] = {}
        if source_rows is not None:
            now = time.time()
            for row in source_rows:
                if not self._valid(row) or is_holdout(row['fen']):
                    continue
                key = sample_key(row)
                # The caller has already applied any role-specific feedback.
                latest.setdefault(key, (row, now))
            return latest
        paths = sorted(list((self.base / 'replay').glob('*.gz')) +
                       list((self.base / 'population').glob('*.gz')) +
                       list((self.base / 'inbox').glob('*.gz')),
                       key=lambda path: path.stat().st_mtime, reverse=True)
        for path in paths[:MAX_SOURCE_FILES]:
            try:
                batch = read_bundle(path)
                mtime = path.stat().st_mtime
            except (FileNotFoundError, OSError, EOFError, ValueError, TypeError):
                continue
            for row in batch:
                if not self._valid(row) or is_holdout(row['fen']):
                    continue
                key = sample_key(row)
                if key not in latest:
                    latest[key] = (row, mtime)
        return latest

    def _load(self) -> tuple[int, dict[str, dict]]:
        try:
            raw = read_json(self.state_path, {})
        except (OSError, ValueError, TypeError):
            raw = {}
        if not isinstance(raw, dict) or raw.get('schema') != SCHEMA:
            return 0, {}
        entries = {}
        for entry in raw.get('entries', []):
            if not isinstance(entry, dict) or not isinstance(entry.get('row'), dict):
                continue
            row = entry['row']
            if not self._valid(row):
                continue
            try:
                key = sample_key(row)
            except (KeyError, IndexError):
                continue
            if is_holdout(row['fen']):
                continue
            entry = dict(entry, key=key, stage=stage_for_row(row),
                         difficulty=max(_finite(entry.get('difficulty'), 1.0), 1.0),
                         rank=max(_finite(entry.get('rank'), 1.0), 1e-9))
            entries[key] = entry
        return max(0, int(raw.get('seen', 0))), entries

    @staticmethod
    def _select(entries: list[dict], budget: int) -> list[dict]:
        if budget <= 0 or not entries:
            return []
        groups = collections.defaultdict(list)
        for entry in entries:
            groups[entry.get('stage', 'unknown')].append(entry)
        selected: list[dict] = []
        selected_keys = set()
        for stage, quota in zip(STAGE_NAMES, STAGE_QUOTAS):
            group = sorted(groups.get(stage, []), key=lambda item: (-item['rank'], item['key']))
            take = min(len(group), max(1, round(budget * quota)))
            selected.extend(group[:take])
            selected_keys.update(item['key'] for item in group[:take])
        if len(selected) < budget:
            remaining = sorted((item for item in entries if item['key'] not in selected_keys),
                               key=lambda item: (-item['rank'], item['key']))
            selected.extend(remaining[:budget - len(selected)])
        return selected[:budget]

    def refresh(self, source_rows: list[dict] | None = None) -> list[dict]:
        now = time.time()
        seen, entries = self._load()
        fresh = self._fresh(source_rows)
        for key, (row, mtime) in fresh.items():
            entry = entries.get(key)
            if entry is None:
                seen += 1
                weight = difficulty(row)
                entries[key] = {'key': key, 'row': row, 'stage': stage_for_row(row),
                                'difficulty': weight, 'rank': _rank(key, weight, self.role),
                                'first_seen': now, 'last_seen': now, 'source_mtime': mtime}
                continue
            entry['last_seen'] = now
            if mtime >= _finite(entry.get('source_mtime'), 0.0):
                entry['row'] = row
                entry['source_mtime'] = mtime
                weight = difficulty(row)
                entry['difficulty'] = weight
                entry['rank'] = max(_finite(entry.get('rank'), 0.0), _rank(key, weight, self.role))
                entry['stage'] = stage_for_row(row)

        reservoir = self._select(list(entries.values()), self.capacity)
        fresh_entries = [entries[key] for key in fresh if key in entries]
        long_term = self._select(reservoir, self.reservoir_budget)
        recent = self._select(fresh_entries, self.fresh_budget)
        selected = {entry['key']: entry for entry in long_term}
        selected.update({entry['key']: entry for entry in recent})
        rows = [entry['row'] for entry in sorted(selected.values(),
                                                  key=lambda item: (item.get('stage', 'unknown'), -item['rank'], item['key']))]
        atomic_json(self.state_path, {'schema': SCHEMA, 'role': self.role, 'seen': seen,
                                      'updated_at': now, 'capacity': self.capacity,
                                      'entries': reservoir})
        counts = collections.Counter(stage_for_row(row) for row in rows)
        ages = [(now - _finite(entry.get('first_seen'), now)) / 86400 for entry in selected.values()]
        event(self.role, 'replay_buffer', fresh_rows=len(fresh), reservoir_rows=len(reservoir),
              training_rows=len(rows), unique_seen=seen, stage_counts=dict(counts),
              difficult_rows=sum(difficulty(row) > 1.5 for row in rows),
              oldest_days=round(max(ages, default=0.0), 3))
        return rows
