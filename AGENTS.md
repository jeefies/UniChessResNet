# AGENTS.md — UniChess ResNet

Chess engine: Stockfish-distilled ResNet + MCTS, plus an autonomous self-play loop (`autoloop/`). `README.md` (Chinese) holds measured Elo/throughput numbers and a bug-history log worth reading before changing encoding, MCTS, or data labeling.

## This checkout is source-only — most of the repo cannot run here

`.gitignore` excludes `data/`, `runs/`, `logs/`, `.venv/`, `*.pt`. Two consequences that bite immediately:

- **`data/record.py` is not in the repo** (all of `data/` is gitignored, including its Python source). It defines `RECORD_DTYPE`, `NO_PROMO`, `board_to_record`, `record_to_board` and is imported by `model/dataset.py:17`, `model/train*.py`, `model/test_dataset.py`, `autoloop/worker.py`, `autoloop/layered.py`. All of those fail at **import time** here. Never reconstruct `RECORD_DTYPE` from guesswork: the on-disk shards were written by the missing module, and a wrong layout silently mis-trains instead of erroring.
- Same for `data/build_evals.py`, `data/build_pgn.py`, `data/test_record.py`, `data/test_build_evals.py`, `data/download_mt.sh`, `data/ship_shards.sh` — referenced by `run_tests.sh` and the README but absent.

Windows: `autoloop/common.py:2` imports `fcntl` at module scope, so `common`, `worker`, `layered`, `governor`, `relay`, `alphazero`, `replay_buffer`, `dashboard_export`, `test_system` are all unimportable here. `autoloop/evaluation.py` is the only autoloop module that imports cleanly. `autoloop/common.py:5-6` additionally raises `ValueError` at import time if `UNICHESS_LOOP_RUN` contains a slash or backslash, or does not start with `autoloop`.

All root `*.sh` are bash with Linux-absolute paths; none run on Windows without WSL.

## Tests

Plain scripts with `if __name__ == "__main__": raise SystemExit(main())`, returning 0/1. **Not pytest** — there is no pytest/tox/pyproject config anywhere, and the checks are named `check_*`, so a pytest run collects nothing and "passes" vacuously. Run from the repo root: every test does `sys.path.insert(0, parent)` and resolves `tools/stockfish` / `data/raw/syzygy345` relative to cwd.

```
python core/test_roundtrip.py      # 100k positions, no external deps
python model/test_dataset.py       # needs data/record.py + shards
python engine/test_tablebase.py    # needs Syzygy + runs/smoke/ckpt_00000200.pt
python search/test_mcts.py         # needs tools/stockfish for the mate-in-2 check
python -m unittest autoloop.test_system -v   # POSIX only
```

`run_tests.sh` hardcodes `PY=.venv/bin/python` and, under `set -e`, aborts at line 8 on the missing `data/test_record.py`, so the later tests never run. It also does not cover `search/test_mcts.py` or `autoloop/test_system.py`.

Skip-vs-fail behavior differs and matters: `search/test_mcts.py:124-127` skips the tablebase check gracefully (returns True) when fewer than 100 `.rtbw` files are present, but a missing `tools/stockfish` raises out of `main()` and aborts the whole run. `engine/test_tablebase.py` has no guard and exits 1.

GPU-mandatory (hardcoded `cuda`, no fallback): `gpubench.py`, `prec_bench.py`, `split_bench.py`, `search/bench_mcts.py`, `tools/verify_fast.py`.

## Encoding contract — changing it invalidates every trained checkpoint

- 19 planes, `(19,8,8)` float32 (`core/encoding.py:25-27`). Plane order is fixed by `_PIECE_ORDER` (`core/encoding.py:30`); reordering breaks existing weights. `NUM_PLANES` is duplicated at `model/net.py:21` — update both.
- **Always from the side-to-move's perspective**, so there is no side-to-move plane. `orient()` returns the board for White and `board.mirror()` for Black (`core/encoding.py:40`), which flips vertically *and* swaps colors, making planes 0-5 always "mine". Callers must `orient_move` before indexing policy (`engine/engine.py:158-162`, `autoloop/worker.py:286-288`).
- Policy is `POLICY_SIZE = 4096`, index `from_square*64 + to_square` (`core/moves.py:15`, `:25`). Promotions are **not** in that index; they use a separate 4-way head with fixed order `(QUEEN, ROOK, BISHOP, KNIGHT)` (`core/moves.py:19-20`). The head's 1x1 conv then `reshape(n, nb, 4096)` makes plane=from, spatial=to (`model/net.py:91-92`, `:120`); flatten+FC would be ~50M params.
- `core/test_roundtrip.py:45-51` asserts that no two legal moves in one position collide on `(index, promo)`. That collision-freedom is what makes the 4096+4 factorization valid; any move-encoding change must keep it green.
- Planes 17/18 (halfmove, repetitions) need real board history. `repetitions=None` infers via `board.is_repetition()` and silently yields 0 for a bare FEN (`core/encoding.py:68-69`).

## Record format and labels

- The record is documented as 96 bytes, but the authoritative `RECORD_DTYPE` lives in the missing `data/record.py`, so treat the layout as unverified. Files are headerless: count is `file_size // itemsize` (`model/dataset.py:115`).
- **Top-k is hardcoded to 5** in the decoder (`np.repeat(np.arange(n), 5)`, `model/dataset.py:80`). There is no k field and no assertion, so a shard written with a different k silently mis-decodes.
- Sentinels: ep `255` means none (`model/dataset.py:56-57`); promo target `-100` is the loss `ignore_index` (`model/dataset.py:86`, `model/train.py:114`).
- Castling bits are own/opp-swapped when Black is to move (`model/dataset.py:48-52`) — the most error-prone spot in the decoder.
- **Everything is mover-relative.** WDL is stored in the mover's frame, and self-play labeling uses `score.pov(b.turn)` (`autoloop/worker.py:257-264`) rather than manual negation. The evals cp→WDL negation for Black lives in the missing `data/build_evals.py`; per README:94-96 a sign error there only surfaces at Elo measurement.
- All-zero policy rows are legal (PGN shards are value-only) and stay zero by design (`model/dataset.py:81-83`). The loss masks them via `has_policy = p_t.sum(1) > 1e-6` (`model/train.py:111-112`), and `soft_cross_entropy` returns a hard 0 when the mask is empty (`model/train.py:38-45`).

## Training entrypoints

`model/train_iteration_fast.py` is canonical: `autoloop/governor.py:25` identifies the live trainer by that exact path and `:76` launches it, and `autoloop/worker.py:14` / `autoloop/layered.py:31` import `Pool`, `loss_for`, `save_atomic` from it.

- `model/train_iteration.py` is the superseded **correctness oracle**, kept so `tools/verify_fast.py` and `tools/bench_iteration.py` can diff value and gradients against the fast path. Do not delete it, and keep the two `loss_for` implementations equivalent. Note `train_iteration.py:87-90` does not upcast logits to fp32 while the fast path does (`train_iteration_fast.py:91-94`).
- `model/train.py` is the earlier Stage-1 distillation script (the one in README:129). Nothing imports or launches it; it has `--preset`, OneCycleLR, and no resume, no channels-last, no cuDNN autotune.
- Both iteration scripts refuse to start if `latest.pt` exists without `--resume`. On resume `train_iteration.py:120` requires `batch` and `accum` to match individually, while `train_iteration_fast.py:124` only requires the product, so the fast path may be re-sharded across micro-batch sizes.
- Presets: `tiny 6x64`, `small 10x128`, `medium 15x192`, `large 20x256` (`model/net.py:146-151`). `--buckets > 1` rebuilds `NetConfig` and silently reverts `se_ratio`, `value_channels`, `value_hidden` to defaults (`model/train.py:56-63`).

Performance invariants that are easy to break:

- **Frozen BatchNorm is order-dependent.** `autoloop/worker.py:330-331` and `autoloop/layered.py:324-327` set every `BatchNorm2d` to `.eval()` *after* `model.train()`, so tiny fresh replay batches cannot overwrite pretrained running stats. Any later `model.train()` call silently unfreezes them.
- Fused AdamW is re-stamped on resume (`group['fused']=True; group['foreach']=None`, `train_iteration_fast.py:126-127`) because loading a non-fused state dict would otherwise drop it.
- channels-last applies to model, inputs, and 4-D optimizer state (`train_iteration_fast.py:118`, `:82`, `:132`). Training uses bf16 autocast; inference uses **fp16 on purpose** — `prec_bench.py` measured fp16 deviation 1.75e-3 vs bf16 1.28e-2 at equal speed (`engine/engine.py:57-58`).
- Batch-level decoding is a requirement, not an optimization: per-item `__getitem__` costs roughly 6x (`model/dataset.py:139-143`).
- `torch.compile` is not used anywhere. TF32 is only ever explicitly disabled, in `tools/verify_fast.py:9-10`.

## Engine / UCI

**No Python file in the engine, MCTS, or eval layer reads any environment variable.** `UNICHESS_CKPT`, `UNICHESS_MCTS`, `UNICHESS_DEVICE`, `UNICHESS_SYZYGY` are consumed only by the shell wrappers, which translate them into CLI flags. Exporting them around `python uci.py` does nothing.

Wrapper differences are not cosmetic:

| | interpreter | env | cwd | device | Syzygy default |
|---|---|---|---|---|---|
| `unichess.sh` | `.venv/bin/python` | venv, no activate | script dir | cpu | **always on**, cannot be disabled (`:8` uses `:-`) |
| `unichess_gpu.sh` | conda PATH python | `conda activate unichess` | script dir | cuda | on unless explicitly empty (`:17-21`) |
| `unichess_pro.sh` | `/root/autodl-tmp/conda_envs/hybridflow/bin/python` | `PYTHONPATH=$BASE/pylibs` | **`cd /root/autodl-tmp/fwj/UniChess`** | cuda | **off** unless set (`:15`) |

PRO 6000 is a shared machine: dependencies go into a project-local `pylibs/` via `pip --target` and enter through `PYTHONPATH`; never install into the shared conda env (README:17-20, `unichess_pro.sh:10`, `run_pro6000.sh:12`).

External deps, all gitignored: `tools/stockfish` (hard failure if missing), `data/raw/syzygy345` (graceful in `engine/engine.py:67-72`, hard-fails `engine/test_tablebase.py`), `eval/openings.txt` (auto-generated when absent, which itself needs Stockfish).

`eval/openings.txt` has **40 lines** and `eval/arena.py:266` slices `[:pairs]`, so the README's `--pairs 100` silently yields 40 pairs / 80 games. Games = 2 x pairs, colors reversed. `eval/calibrate.py` has no `--nodes` for our own engine; our side is always time-limited.

## MCTS invariants (hard-won; keep them)

Each corresponds to a documented bug in README:83-118.

- Root visits every legal move at least once: `root_min_visits=1` (`search/mcts.py:45`), enforced at `:212-214`. Best-effort only — with `simulations < n_legal` some moves still get zero.
- Terminal-node hits count against the simulation budget: `terminal_sims` at `search/mcts.py:230`, `:237`, `:246`, with the loop condition at `:199`. Without this a solved position spins (observed 7661 visits at sims=200).
- `board.is_valid()` guards tablebase probes (`engine/engine.py:122-125`, `search/mcts.py:164`) because python-chess generates king-capture moves from illegal positions and then probes a nonexistent table.
- Syzygy move ordering is `(opponent WDL, is-non-zeroing, DTZ)`, not DTZ alone (`engine/engine.py:147-150`). The variable is named `zeroing` but holds `0` *for* zeroing moves so they sort first under `min`. DTZ-only ordering made KPvK unwinnable.
- **Known inconsistency:** `search/mcts.py:359-373`'s root-tablebase fallback ranks by `(-value, signed dtz)` under `max` and ignores zeroing entirely — the exact bug the fix above removed. It is reached only when the root resolves to an exact value, so pure-MCTS pawn endgames do not get the fix; `engine.play`'s tablebase-first path does.
- Virtual loss is subtracted from W, not merely counted as a visit (`search/mcts.py:94`, `:99`), otherwise pending paths look neutral and collide repeatedly.
- Collisions deliberately do **not** flush the GPU batch (`search/mcts.py:250-259`); flushing fragments an 800-sim search into many tiny batches.
- `search/mcts.py:170-176` refuses a tablebase win when `halfmove_clock + dtz >= 100`. `engine/engine.py` has no equivalent.
- Dead config fields: `MCTSConfig.temp_moves` and `c_puct` are never read (`search/mcts.py:40`, `:33`); `best_child` uses `c_puct_base` / `c_puct_init`.

## autoloop

Roles: `small` (batch 512, <=200 updates/generation) and `big` (batch 1024, <=400), both sized from their seed checkpoint's `cfg` rather than literals (`autoloop/worker.py:305`, `:310`, `:334`, `:537`); and `layered`, five independent 16x256 (~19.9M) nets bucketed by piece count, inclusive: 25-32, 19-24, 13-18, 7-12, 2-6 (`autoloop/layered.py:43-50`). Out-of-range counts raise in `layered.py:55-59` but map to `'unknown'` in `replay_buffer.py:54`.

State root is `runs/$UNICHESS_LOOP_RUN` (default `autoloop`; `autoloop/common.py:5-7`). `tools/check_autoloop.py:3` and `tools/cleanup_autoloop_artifacts.py:5` hardcode `runs/autoloop` and therefore inspect the wrong directory under an override.

Promotion gate (`autoloop/worker.py:422`) requires all of: every pair complete, mean paired score `> 0.52`, sign-test `p < 0.05`, validation non-regression (loss <= 1.05x, Brier <= 1.10x, value MAE <= 1.10x per category, checked *before* the arena), and the 3200-sim regression position. `score_lower95_hoeffding` is reported but is not part of the decision. Layered stages have no arena at all — validation-only plus an `improved` check (`autoloop/layered.py:376-390`).

Atomic writes are mandatory for anything the supervisor reads: `atomic_json` does tmp + `fsync` + `replace` with `allow_nan=False` (`autoloop/common.py:15-20`); `save_atomic` does tmp + `replace` without fsync (`model/train_iteration_fast.py:22-25`). Promotion is `os.replace`, never copy (`autoloop/worker.py:555`). `write_bundle` uses `with_suffix`, so `x.json.gz` becomes `x.json.tmp`, which is why stale `.tmp` files are swept after 24h (`autoloop/common.py:55-59`, `:89-92`).

Control plane: create `runs/<run>/PAUSE` for a graceful pause (`autoloop/worker.py:466`) and remove it to allow idle-gated resume. **Never SIGSTOP a worker** — a suspended process keeps its VRAM; the governor's design goal is that competing jobs see zero training intensity instead of a stalled process holding memory. The governor re-validates PID and process start time before signalling and only ever signals its own verified worker (`autoloop/governor.py:62`, `:68`).

`autoloop/README.md` documents the loop, storage quotas, and the full observability layout. Read it before touching anything under `autoloop/` or `runs/autoloop/`.

## Dangerous commands

- `scripts_stopall.sh` kills **all** `rsync`, `curl`, and `scp` processes on the box via `pgrep -x`, not just this project's (`:17-19`). Its own header warns that the invoking command line must not contain `ship` or `download`, or `pgrep -f` matches the caller.
- `scripts_gpu_attrib.sh:31` and `tools/run_speed_check.sh:6` SIGSTOP live processes; the resume path has no robust trap, so a killed shell leaves training suspended indefinitely.
- `tools/install_autoloop_service.py <role>` writes and `enable --now`s a `Restart=always` systemd user unit: GPU work starts immediately and survives logout.
- `tools/switch_training_fast.py` takes no arguments and stops the live trainer with no dry-run.
- `tools/cleanup_autoloop_artifacts.py --apply` runs `shutil.rmtree` (guarded: dry-run by default, asserts the >100MiB bootstrap checkpoints exist, refuses symlinks and paths outside `runs/`).
- `tools/launch_iteration46.sh` starts a 125,000-step GPU run.
- `scripts_uptest.sh` does `rm -rf /tmp/uptest` on a remote host and hardcodes `jeefy@172.16.2.12`.
- `tools/bench_iteration.py` allocates a 24x320 net on CUDA and will contend with live training.

Hardcoded to `/home/jeefy/UniChess` or `/root/autodl-tmp/...`: everything in `tools/*.sh`, `tools/switch_training_fast.py`, `tools/prepare_fast_smoke.py`, `tools/prepare_migration_parts.py`, plus run-directory literals such as `iteration46_20260909`.
