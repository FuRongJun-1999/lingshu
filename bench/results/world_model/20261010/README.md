# World-model evidence (2026-10-10)

See the [analysis and aggregate results](../../../../docs/world_model_optimization.md), [metric migration](../../../../docs/world_model_metrics_migration.md), and [benchmark protocol](../../../README.md).

The large per-seed JSON files are removed from the current PR diff. Their original publication remains accessible in the PR's initial commit; this does not claim to erase already published Git objects. Local raw records and a ZIP archive are kept under untracked `artifacts/`. No new release, external storage service, or dependency is introduced.

The summaries remain in the analysis; the small `provenance.json` keeps source fingerprints, benchmark and integration bases, and test scope. The original per-seed evidence is pinned to initial commit `593ba9441dc81c73dd8894f1d914ee64a174950c`, with SHA-256 below over those Git blob bytes:

| File | SHA-256 |
|---|---|
| [original.json](https://github.com/rinDBeans/lingshu/blob/593ba9441dc81c73dd8894f1d914ee64a174950c/bench/results/world_model/20261010/original.json) | `2854761b32947c8c88c250e8a6a4cbe9f0e2e04d2942acea47621258d0a5ebb3` |
| [observation_fix.json](https://github.com/rinDBeans/lingshu/blob/593ba9441dc81c73dd8894f1d914ee64a174950c/bench/results/world_model/20261010/observation_fix.json) | `a8186feee4f0c3a3ab0e716a27c951af9ac10c33535423e3aa343c1a36d76a68` |
| [pursuit_fix.json](https://github.com/rinDBeans/lingshu/blob/593ba9441dc81c73dd8894f1d914ee64a174950c/bench/results/world_model/20261010/pursuit_fix.json) | `4650c80b19824f36e3b80bebf713f90696d014b1cd09ef3604eabd194bad0370` |
| [additional_seeds.json](https://github.com/rinDBeans/lingshu/blob/593ba9441dc81c73dd8894f1d914ee64a174950c/bench/results/world_model/20261010/additional_seeds.json) | `52e20b45353ac86047623e2440e1b79edbe7e7317428cbe072163cbf04c84af0` |
| [upstream_latest.json](https://github.com/rinDBeans/lingshu/blob/593ba9441dc81c73dd8894f1d914ee64a174950c/bench/results/world_model/20261010/upstream_latest.json) | `e8af8e5565bf2a26bbfe8cd1e1d86134e30d06a4b7b58d9a746fccfd05aa6022` |
| [integrated.json](https://github.com/rinDBeans/lingshu/blob/593ba9441dc81c73dd8894f1d914ee64a174950c/bench/results/world_model/20261010/integrated.json) | `9e07ad14956c6d6cb173fa6f006f720d3dca88cfeee876888bc14db53c5c13ea` |
| [integrated_additional_seeds.json](https://github.com/rinDBeans/lingshu/blob/593ba9441dc81c73dd8894f1d914ee64a174950c/bench/results/world_model/20261010/integrated_additional_seeds.json) | `3fbe1dbf450029cac7e578371fd2eb2a05f97b773d7f0228fdf7a080fa83184f` |

`original` is the original `abb52d0` baseline; `observation_fix` and `pursuit_fix` are the historical first and second patches. `additional_seeds` compares those two historical patches. `upstream_latest` is pristine `c22385c`; `integrated` is the integrated predictor; `integrated_additional_seeds` compares `c22385c` with that predictor. Historical 11/50 random-observation regressions belong to first→second; the integrated predictor improved all 50 random-observation seeds against `c22385c`. These are different baselines. Unified direction reversal and individual curiosity/close-target failures remain disclosed.

Each paired experiment has matching actual trajectory hashes. Historical phases were measured as uncommitted sources: Git HEAD alone cannot identify them. Historical `before_runtime_source_sha256` identifies injected frozen sources. Final integration also includes `777e18f`; its default benchmark was rechecked with identical metrics. Scope is controlled simulators, known entity IDs, and online model updates; no probability calibration, significance test, or real-world generalization is claimed.

## Reproduce current versus committed baseline

Run from a checkout of this PR (PowerShell):

```powershell
git worktree add --detach ../lingshu-world-bench-base c22385cb2d75f36bc12fd1de4449edbf92be8698
New-Item -ItemType Directory -Force ../lingshu-world-bench-base/bench | Out-Null
Copy-Item -LiteralPath bench/bench_world.py -Destination ../lingshu-world-bench-base/bench/bench_world.py
python -X utf8 -B ../lingshu-world-bench-base/bench/bench_world.py --output artifacts/world_baseline.json
python -X utf8 -B bench/bench_world.py --output artifacts/world_current.json
python -X utf8 -B bench/bench_world.py --compare artifacts/world_baseline.json artifacts/world_current.json
```

Use a new destination path if that worktree already exists. The same frozen benchmark script must run in both checkouts. Add `--seeds 60` to both runs for seeds 0–59; compare seeds 10–59 separately when checking the additional-seed claims. Exact point motion across horizons is guarded by tests; differences in full dictionaries alone can reflect target-tick metadata.
