# UniChess autonomous iteration

This system runs in the ORIGINAL projects, never a second Windows source checkout.

## Machines and services

- 5070 Ti: `/home/jeefy/UniChess`, `/home/jeefy/miniconda3/envs/unichess/bin/python`.
  User service `unichess-autoloop-actors`: four CPU search processes share one batched CUDA evaluator. Each completed game is teacher-reviewed and published immediately. Independent service `unichess-autoloop-small` runs the learner/arena so evaluation does not stop data production.
  Actor 并发默认 8，可用 `UNICHESS_ACTOR_COUNT`、`UNICHESS_INFERENCE_BATCH` 和 `UNICHESS_INFERENCE_WAIT_MS` 调节；默认聚合上限为 256 个局面，参数会记录到 `actor_batch_config` 指标。
- PRO 6000: `/root/autodl-tmp/fwj/UniChess`, existing `run_pro6000.sh` environment.
  `tools/run_autoloop_pro.sh` supervises `autoloop.governor`. No shared environment or other users' services are changed.
- WSL Ubuntu: `/home/jeefy/UniChess`, user service `unichess-autoloop-relay`.
  Reuses existing SSH configuration to relay checksummed compressed bundles in both directions. **WSL and its network must remain running for cross-machine synchronization.** Both remote workers survive a relay outage; PRO waits when no new replay is available. No private keys were copied to either remote host.

## Resource ownership

PRO samples GPU processes, attributed/unattributed memory, utilization, temperature, power and free disk every 5 seconds. ANY foreign CUDA process (even low utilization), >768 MiB unattributed VRAM, <8 GiB free VRAM, a telemetry failure or storage pressure requests graceful shutdown of OUR verified worker. PID, command, project directory and process start time are checked before signals. Other jobs are never signalled.

Training saves model + optimizer and exits at an update boundary; arena progress persists every 8 plies. GPU memory is released when the worker exits. A hung worker may be killed after 180 seconds, falling back to the last atomic checkpoint (normally every 50 updates). Resume requires 120 seconds of continuous idle. This is deliberately conservative: competing workloads cause zero training intensity rather than retaining a low-duty process's VRAM. Checkpoint I/O can briefly overlap the competing job during shutdown.

The bootstrap completed 125,000 updates. Its original `latest.pt` and `best.pt` are retained. The big loop starts from that best-loss model as an INITIAL BASELINE, not as an asserted stronger champion.

## Learning loop

1. Freeze the current small champion for four self-play games. 800 MCTS simulations/move, up to 256 plies, exploration in the first 16 plies. Search starts: 25% openings, 50% actual <=10-piece positions, 25% advanced-pawn positions from existing labelled shards.
2. Preserve every move, FEN/history, root priors, visit distribution, edge Q, search value, entropy and result/termination. A ply-limit cutoff has UNKNOWN result, never an invented draw. Threefold/50-move claims are explicit termination reasons.
3. Review selected positions with Stockfish: 200k nodes / top 5 moves, plus 100k nodes for the played move. These are finite-search estimates, not proof of optimal play. Local Syzygy tables are enabled if available; probe-hit counts are logged. Threat and repetition positions receive denser sampling.
4. Train on 75% original supervised anchors + 25% recent teacher-reviewed replay. Replay policy: 75% Stockfish / 25% MCTS visits. WDL: Stockfish alone for unfinished games, or 75% teacher / 25% actual outcome for completed games. Keep the original general/endgame/promotion anchor mixture.
   Each role now maintains a persistent bounded reservoir (`small`, `big`, and an independent `layered` buffer), with five equal piece-count quotas, deterministic difficulty-weighted admission, a fresh-window slice, holdout exclusion and target-content data-versioning. A generation is therefore reproducible while old difficult positions remain available after source bundles expire.

5. Small: up to 200 updates/generation, batch 512. Big: up to 400 updates/generation, batch 1024. Fresh generations are limited to approximately four replay passes to avoid repeatedly fitting a tiny initial dataset. BF16, channels-last, fused AdamW, LR 1e-5, clipped gradients, compiled replay in RAM, and frozen pretrained BatchNorm running statistics to avoid calibration drift on tiny fresh replay sets. Immutable generation snapshots make resource-interrupted training resumable. A promising but unpromoted challenger can continue across generations; champion status is separate.
6. Gate against the champion using 24 color-swapped opening/position pairs, 400 simulations/move, max 320 plies. Gate starts rotate by generation and come from held-out shards; the original datasets are NOT globally FEN-deduplicated. Unknown games prevent promotion. Require all pairs complete, score >52%, paired exact sign-test p<0.05, no >5% validation-loss or >10% Brier/value-error regression in any category (checked before the arena), and no regression in the reported promotion-blocking test at 3200 simulations. These are per-candidate checks, not a lifetime statistical guarantee or a claim about 3200-simulation match strength.
7. PRO sends legal-move policy feedback to the small learner; the 16 largest recorded mistakes also receive 3200-simulation big-model reanalysis. Feedback only receives 10% weight when its best move agrees with Stockfish and its value is close. Two networks agreeing alone is not treated as ground truth.

8. The layered family uses five independent 19.9M-parameter models: 25-32, 19-24, 13-18, 7-12 and 2-6 pieces. Each stage has its own replay slice, held-out validation, recovery checkpoint and champion/previous slot. MCTS routes mixed batches by piece count; Syzygy remains authoritative for supported <=5-piece positions.
9. After each big generation, PRO trains the layered stages sequentially, then runs mirrored cross-play for small-vs-layered, layered-vs-big and small-vs-big. The 5070 Ti snapshot must be available before this three-family evaluation is considered complete.

True progress is tracked separately from promotion: the first complete three-family snapshot is frozen as a reference, and every fifth generation replays fixed starts, fixed colors and fixed search settings against it. A progress signal requires complete games, zero illegal moves, score above 50% and paired sign-test p<0.05 for a family. Unknown/cutoff games are reported and never scored as draws.
No automated replacement of the public website model is performed. Accepted internal champions and their rollback weights are under `runs/autoloop/models/`.

## Observability

`runs/autoloop/` contains:

- `small-status.json`, `big-status.json`: current phase/generation/model hash.
- `governor-status.json`: PRO process ownership, live GPU metrics, yield reason and idle timer.
- `cluster-status.json` (WSL): both workers and GPU summaries; each has timestamps, so stalled or disconnected workers are visible.
- `metrics/*.jsonl`: per-move search visits/Q/entropy/depth/collisions/network work; batched inference timing; game lengths, promotions, termination/cutoffs; teacher depth/node budget/estimated CP losses/threat/endgame counts; loss components, WDL Brier score, entropy, gradient norm, LR, training throughput and CUDA memory; validation by category, regression move, paired arena outcomes/statistics; queue counts, hashes/transfers, errors, resource transitions and promotions.
- `games/*.json.gz`: full game trajectories and arena games (start FEN + UCI history).
- `replay/`, `inbox/`, `feedback/`: compressed versioned training data and teacher provenance.
- `small-replay-reservoir.json`, `big-replay-reservoir.json`, `layered-replay-reservoir.json`: bounded persistent replay state; each `replay_buffer` event reports fresh/reservoir/training counts, stage distribution, difficult-row count and oldest sample age.
- `*-last-arena.json`, `*-arena-progress.json`: reviewable gate decision and resumable evaluation.
- `storage.json`, `cleanup.json`: quota status and audited one-time cleanup.
- `verification-evidence/`: retained smoke-test metrics/logs after temporary models are removed.

Raw metrics are intentionally extensible. They cover the main failure modes above; no finite metric list can guarantee detection of every chess or infrastructure defect.

## Bounded storage

Per machine: 24 GiB managed-storage high-water mark and 20 GiB free-space reserve. These are application watermarks, not filesystem-enforced hard quotas; atomic writes need temporary headroom. Pressure pauses new work without deleting user datasets or protected model files.

Replay and inbox: each <=4 GiB / 7 days. Feedback: <=2 GiB / 7 days. Games: <=1 GiB / 30 days. Jobs: <=1 GiB / 7 days, excluding active immutable snapshots. Each metric stream rotates at 16 MiB with four archives. Abandoned temporary transfers older than one day are removable.

Model slots are fixed: champion, previous champion, candidate, optional continuing learner, one frozen actor model, and one optimizer recovery checkpoint. There is no checkpoint per generation. Only named, project-owned expendable artifacts are cleaned; source code, original labelled shards and bootstrap recovery weights are protected.

## Operations

On 5070 Ti:

    systemctl --user status unichess-autoloop-small unichess-autoloop-actors
    systemctl --user stop unichess-autoloop-small
    systemctl --user start unichess-autoloop-small

On WSL:

    systemctl --user status unichess-autoloop-relay
    cat /home/jeefy/UniChess/runs/autoloop/cluster-status.json

On PRO, create `runs/autoloop/PAUSE` to request a persistent graceful pause. Remove that file to allow idle-gated recovery. Do not use SIGSTOP: it retains GPU memory. The WSL relay ensures the project supervisor stays running; it honors PAUSE.

Tests: `python -m unittest autoloop.test_system -v` and `python search/test_mcts.py`. GPU smoke tests must use an isolated `UNICHESS_LOOP_RUN=autoloop_smoke` directory. The temporary validation/transfer tests are not chess-strength evidence.
