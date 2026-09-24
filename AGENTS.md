# AGENTS.md — UniChess ResNet

Chess engine: Stockfish-distilled ResNet + MCTS (maintenance mode). Library code lives in the `unichess_r` package (`core/ engine/ model/ search/`); batch arenas and self-play go through UniChessKit via `unichess_r/kit_adapter.py` (`evaluate_batch` for boards, `evaluate_planes` for pre-encoded planes, which the kit C++ PUCT uses by default). The old autonomous loop (`autoloop/`, `selfplay/`, `tools/`) and its systemd units were removed on 2026-09-23. `README.md` (Chinese) holds measured Elo/throughput numbers and a bug-history log worth reading before changing encoding, MCTS, or data labeling.

## This checkout is source-only — most of the repo cannot run here

`.gitignore` excludes `data/`, `runs/`, `logs/`, `.venv/`, `*.pt`. Two consequences that bite immediately:

- **`data/record.py` is not in the repo** (all of `data/` is gitignored, including its Python source). It defines `RECORD_DTYPE`, `NO_PROMO`, `board_to_record`, `record_to_board` and is imported by `unichess_r/model/dataset.py:17`, `unichess_r/model/train*.py`, `unichess_r/model/test_dataset.py`. All of those fail at **import time** here. Never reconstruct `RECORD_DTYPE` from guesswork: the on-disk shards were written by the missing module, and a wrong layout silently mis-trains instead of erroring.
- Same for `data/build_evals.py`, `data/build_pgn.py`, `data/test_record.py`, `data/test_build_evals.py`, `data/download_mt.sh`, `data/ship_shards.sh` — referenced by `run_tests.sh` and the README but absent.

All root `*.sh` are bash with Linux-absolute paths; none run on Windows without WSL.

## Tests

Plain scripts with `if __name__ == "__main__": raise SystemExit(main())`, returning 0/1. **Not pytest** — there is no pytest/tox/pyproject config anywhere, and the checks are named `check_*`, so a pytest run collects nothing and "passes" vacuously. Run from the repo root: every test does `sys.path.insert(0, parent)` and resolves `tools/stockfish` / `data/raw/syzygy345` relative to cwd.

```
python unichess_r/core/test_roundtrip.py      # 100k positions, no external deps
python unichess_r/model/test_dataset.py       # needs data/record.py + shards
python unichess_r/engine/test_tablebase.py    # needs Syzygy + runs/smoke/ckpt_00000200.pt
python unichess_r/search/test_mcts.py         # needs tools/stockfish for the mate-in-2 check
```

`run_tests.sh` hardcodes `PY=.venv/bin/python` and, under `set -e`, aborts at line 8 on the missing `data/test_record.py`, so the later tests never run. It also does not cover `unichess_r/search/test_mcts.py`.

Skip-vs-fail behavior differs and matters: `unichess_r/search/test_mcts.py:124-127` skips the tablebase check gracefully (returns True) when fewer than 100 `.rtbw` files are present, but a missing `tools/stockfish` raises out of `main()` and aborts the whole run. `unichess_r/engine/test_tablebase.py` has no guard and exits 1.

GPU-mandatory (hardcoded `cuda`, no fallback): `gpubench.py`, `prec_bench.py`, `split_bench.py`, `unichess_r/search/bench_mcts.py`.

## Encoding contract — changing it invalidates every trained checkpoint

- 19 planes, `(19,8,8)` float32 (`unichess_r/core/encoding.py:25-27`). Plane order is fixed by `_PIECE_ORDER` (`unichess_r/core/encoding.py:30`); reordering breaks existing weights. `NUM_PLANES` is duplicated at `unichess_r/model/net.py:21` — update both.
- **Always from the side-to-move's perspective**, so there is no side-to-move plane. `orient()` returns the board for White and `board.mirror()` for Black (`unichess_r/core/encoding.py:40`), which flips vertically *and* swaps colors, making planes 0-5 always "mine". Callers must `orient_move` before indexing policy (`unichess_r/engine/engine.py:158-162`).
- Policy is `POLICY_SIZE = 4096`, index `from_square*64 + to_square` (`unichess_r/core/moves.py:15`, `:25`). Promotions are **not** in that index; they use a separate 4-way head with fixed order `(QUEEN, ROOK, BISHOP, KNIGHT)` (`unichess_r/core/moves.py:19-20`). The head's 1x1 conv then `reshape(n, nb, 4096)` makes plane=from, spatial=to (`unichess_r/model/net.py:91-92`, `:120`); flatten+FC would be ~50M params.
- `unichess_r/core/test_roundtrip.py:45-51` asserts that no two legal moves in one position collide on `(index, promo)`. That collision-freedom is what makes the 4096+4 factorization valid; any move-encoding change must keep it green.
- Planes 17/18 (halfmove, repetitions) need real board history. `repetitions=None` infers via `board.is_repetition()` and silently yields 0 for a bare FEN (`unichess_r/core/encoding.py:68-69`).

## Record format and labels

- The record is documented as 96 bytes, but the authoritative `RECORD_DTYPE` lives in the missing `data/record.py`, so treat the layout as unverified. Files are headerless: count is `file_size // itemsize` (`unichess_r/model/dataset.py:115`).
- **Top-k is hardcoded to 5** in the decoder (`np.repeat(np.arange(n), 5)`, `unichess_r/model/dataset.py:80`). There is no k field and no assertion, so a shard written with a different k silently mis-decodes.
- Sentinels: ep `255` means none (`unichess_r/model/dataset.py:56-57`); promo target `-100` is the loss `ignore_index` (`unichess_r/model/dataset.py:86`, `unichess_r/model/train.py:114`).
- Castling bits are own/opp-swapped when Black is to move (`unichess_r/model/dataset.py:48-52`) — the most error-prone spot in the decoder.
- **Everything is mover-relative.** WDL is stored in the mover's frame,. The evals cp→WDL negation for Black lives in the missing `data/build_evals.py`; per README:94-96 a sign error there only surfaces at Elo measurement.
- All-zero policy rows are legal (PGN shards are value-only) and stay zero by design (`unichess_r/model/dataset.py:81-83`). The loss masks them via `has_policy = p_t.sum(1) > 1e-6` (`unichess_r/model/train.py:111-112`), and `soft_cross_entropy` returns a hard 0 when the mask is empty (`unichess_r/model/train.py:38-45`).

## Training entrypoints

`unichess_r/model/train_iteration_fast.py` is the canonical trainer (run from the repo root: `python -u unichess_r/model/train_iteration_fast.py --out runs/<name> ...`).

- `unichess_r/model/train.py` is the earlier Stage-1 distillation script (the one in README:129). Nothing imports or launches it; it has `--preset`, OneCycleLR, and no resume, no channels-last, no cuDNN autotune.
- It refuses to start if `latest.pt` exists without `--resume`; on resume only `batch*accum` must match, so it may be re-sharded across micro-batch sizes.
- Presets: `tiny 6x64`, `small 10x128`, `medium 15x192`, `large 20x256` (`unichess_r/model/net.py:146-151`). `--buckets > 1` rebuilds `NetConfig` and silently reverts `se_ratio`, `value_channels`, `value_hidden` to defaults (`unichess_r/model/train.py:56-63`).

Performance invariants that are easy to break:

- Fused AdamW is re-stamped on resume (`group['fused']=True; group['foreach']=None`, `train_iteration_fast.py:126-127`) because loading a non-fused state dict would otherwise drop it.
- channels-last applies to model, inputs, and 4-D optimizer state (`train_iteration_fast.py:118`, `:82`, `:132`). Training uses bf16 autocast; inference uses **fp16 on purpose** — `prec_bench.py` measured fp16 deviation 1.75e-3 vs bf16 1.28e-2 at equal speed (`unichess_r/engine/engine.py:57-58`).
- Batch-level decoding is a requirement, not an optimization: per-item `__getitem__` costs roughly 6x (`unichess_r/model/dataset.py:139-143`).
- `torch.compile` is not used anywhere.

## Engine / UCI

**No Python file in the engine, MCTS, or eval layer reads any environment variable.** `UNICHESS_CKPT`, `UNICHESS_MCTS`, `UNICHESS_DEVICE`, `UNICHESS_SYZYGY` are consumed only by the shell wrappers, which translate them into CLI flags. Exporting them around `python uci.py` does nothing.

Wrapper differences are not cosmetic:

| | interpreter | env | cwd | device | Syzygy default |
|---|---|---|---|---|---|
| `unichess.sh` | `.venv/bin/python` | venv, no activate | script dir | cpu | **always on**, cannot be disabled (`:8` uses `:-`) |
| `unichess_gpu.sh` | conda PATH python | `conda activate unichess` | script dir | cuda | on unless explicitly empty (`:17-21`) |
| `unichess_pro.sh` | `/root/autodl-tmp/conda_envs/hybridflow/bin/python` | `PYTHONPATH=$BASE/pylibs` | **`cd /root/autodl-tmp/fwj/UniChess`** | cuda | **off** unless set (`:15`) |

PRO 6000 is a shared machine: dependencies go into a project-local `pylibs/` via `pip --target` and enter through `PYTHONPATH`; never install into the shared conda env (README:17-20, `unichess_pro.sh:10`, `run_pro6000.sh:12`).

External deps, all gitignored: `tools/stockfish` (hard failure if missing), `data/raw/syzygy345` (graceful in `unichess_r/engine/engine.py:67-72`, hard-fails `unichess_r/engine/test_tablebase.py`), `eval/openings.txt` (auto-generated when absent, which itself needs Stockfish).

`eval/openings.txt` has **40 lines** and `eval/arena.py:266` slices `[:pairs]`, so the README's `--pairs 100` silently yields 40 pairs / 80 games. Games = 2 x pairs, colors reversed. `eval/calibrate.py` has no `--nodes` for our own engine; our side is always time-limited.

## MCTS invariants (hard-won; keep them)

Each corresponds to a documented bug in README:83-118.

- Root visits every legal move at least once: `root_min_visits=1` (`unichess_r/search/mcts.py:45`), enforced at `:212-214`. Best-effort only — with `simulations < n_legal` some moves still get zero.
- Terminal-node hits count against the simulation budget: `terminal_sims` at `unichess_r/search/mcts.py:230`, `:237`, `:246`, with the loop condition at `:199`. Without this a solved position spins (observed 7661 visits at sims=200).
- `board.is_valid()` guards tablebase probes (`unichess_r/engine/engine.py:122-125`, `unichess_r/search/mcts.py:164`) because python-chess generates king-capture moves from illegal positions and then probes a nonexistent table.
- Syzygy move ordering is `(opponent WDL, is-non-zeroing, DTZ)`, not DTZ alone (`unichess_r/engine/engine.py:147-150`). The variable is named `zeroing` but holds `0` *for* zeroing moves so they sort first under `min`. DTZ-only ordering made KPvK unwinnable.
- **Known inconsistency:** `unichess_r/search/mcts.py:359-373`'s root-tablebase fallback ranks by `(-value, signed dtz)` under `max` and ignores zeroing entirely — the exact bug the fix above removed. It is reached only when the root resolves to an exact value, so pure-MCTS pawn endgames do not get the fix; `engine.play`'s tablebase-first path does.
- Virtual loss is subtracted from W, not merely counted as a visit (`unichess_r/search/mcts.py:94`, `:99`), otherwise pending paths look neutral and collide repeatedly.
- Collisions deliberately do **not** flush the GPU batch (`unichess_r/search/mcts.py:250-259`); flushing fragments an 800-sim search into many tiny batches.
- `unichess_r/search/mcts.py:170-176` refuses a tablebase win when `halfmove_clock + dtz >= 100`. `unichess_r/engine/engine.py` has no equivalent.
- Dead config fields: `MCTSConfig.temp_moves` and `c_puct` are never read (`unichess_r/search/mcts.py:40`, `:33`); `best_child` uses `c_puct_base` / `c_puct_init`.

## Dangerous commands

- `scripts_gpu_attrib.sh:31` SIGSTOPs live processes; the resume path has no robust trap, so a killed shell leaves the target suspended indefinitely.
- Weights under `runs/` are the only copy (not in git). Smoke tests must write to an isolated `runs/<smoke-name>/`.
