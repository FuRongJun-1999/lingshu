# World-model benchmark

`bench_world.py` uses the existing simulator and observation-based models. It
requires only the Python standard library. Run from the repository root:

```powershell
python -X utf8 -B bench/bench_world.py --output artifacts/world_before.json
# Apply the model patch; keep this benchmark unchanged.
python -X utf8 -B bench/bench_world.py --output artifacts/world_after.json
python -X utf8 -B bench/bench_world.py --compare artifacts/world_before.json artifacts/world_after.json
```

The default is ten seeds (0–9). There are nine controlled cases: full-observation
wandering, pursuit, and patrol; alternating observations of constant motion;
pursuit with round-robin, random, or curiosity observation budgets; and unified
model occlusion and relation reversal. Learners receive 12 full-observation
warmup steps, then 40 evaluation steps; unified cases use 24 evaluation steps
after warmup. The linear and unified cases use explicitly prescribed positions.
They test known mechanisms, not unseen real-world tasks.

Predictions are scored **before** the new observations are fed back into the
model. The learner then updates its statistics online. This benchmark is not a
frozen-parameter held-out evaluation; `WorldLearner.eval_phase()` provides that
separate protocol. Sparse cases use budget 1 (linear) or 2 (pursuit); their scorer
can see the whole simulated world, while the learner receives only its selected
observations. All cases start with known entity IDs. The curiosity case measures
the existing selection rule via `_select`, not the entire `explore_tick` loop.

Each output records per-seed results, mean summaries, Python version, Git HEAD,
four original model source hashes, and hashes of the actual world trajectories
with random entity IDs removed. The comparison rejects changes to the recorded
configuration or trajectories. Source hashes identify modifications; Git HEAD
alone does not identify uncommitted source. Keep the benchmark script fixed too.

| Metric | Meaning |
|---|---|
| `mean_distance`, `rmse_distance` | Euclidean point prediction error |
| `point_hit_rate` | Error strictly below the same fixed 0.5 threshold |
| `naive_*` | Previous physical position as an external audit baseline |
| `bound_coverage` | Actual position inside the model's own reported radius |
| `mean_bound` | Mean radius; report alongside coverage to expose widening |
| `prediction_coverage` | Fraction of actual entities for which a prediction exists |
| `samples` | Number of scored entity/tick predictions |

Coverage alone cannot establish better point predictions. The radii are
heuristic reachable-region allowances, not calibrated probabilities. Elapsed
time is informational; this benchmark is not a performance microbenchmark.
The contract probes also check anonymous association, no-sample verification,
repeated evidence verification, and horizon metadata. Moving-entity multi-step
behavior is covered by the upstream horizon regression tests.

## Changed contracts

**Callers must migrate their rate thresholds and empty-result handling.** See
the [v2 migration guide](../docs/world_model_metrics_migration.md) for old-range
field mapping, version checks, and `None`-safe examples. Learning curves and
policy comparisons also identify their metric version.

- `WorldLearner.eval_phase()` and `CuriosityExplorer.probe()` return
  `metric_version=2`. Learned, naive, and oracle rates now use the same point
  threshold; `bound_coverage` is separate. Oracle gap uses common entity/tick
  outcomes only.
- Unverified rates and losses return `None` (JSON `null`) when there are no
  samples. Unified/seven-layer `hit_rate` remains the legacy **range coverage**
  field; their separate `point_hit_rate` measures point hits. Rolling range rates
  are not directly comparable with learner point rates.
- Spatial `predict`/`generate` retain upstream multi-step extrapolation and carry
  the requested `target_tick`. `masked_loss` rejects `mask_last != 1`; it is a
  time-aware linear diagnostic, not a gradient update.
- Anchor `verify_anchor()` is an idempotent read. Submit new qualified evidence
  to advance stability, subject to the existing minimum channel count and time
  interval. Use `observation_tick` for same-time channel grouping
  and `evidence_id` for same-channel deduplication. Without these optional
  arguments, each submission is a new evidence event; the API cannot infer that
  an upstream sender resent identical physical evidence.

Regression tests: `tests/test_world_observation_contracts.py` and
`tests/test_pursuit_prediction.py`. See the
[two-stage analysis](../docs/world_model_optimization.md) and
[recorded evidence](results/world_model/20261010/README.md).
