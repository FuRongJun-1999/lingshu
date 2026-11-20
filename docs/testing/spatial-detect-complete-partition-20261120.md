# Complete spatial-detection partitions

Baseline: `d74848fb33dbff1d719959df19d1266a1ac60f44`.

`spatial_detect` repeated a width of `axis_length // 3` three times. An axis
not divisible by three therefore lost its final one or two cells from both
energy shares and classification. The supported `image_to_grid` default
test configuration (48-pixel image, 16 lattice columns) produces an 18×16
lattice: its final column was never classified. This is independent of the
earlier grid-to-pixel sampling fix (#116), count constraint (#860), and the
pending M5.5 bridge addition (#1439).

Each boundary now floors the scaled axis length (`i * axis_length // 3`).
Both stages use the same boundaries; empty quadrants of axes shorter than
three cells are skipped. Divisible dimensions keep their original slices.
Confidence thresholds, model weights, training, and foreground policy are
unchanged. No original image assets are required for these tests.

## Reproduction and regression

```sh
python -X utf8 -m pytest tests/test_spatial_detect_complete_partition.py -q
python -X utf8 -m pytest tests/test_hex_text.py tests/test_hex_text_thresholds.py tests/test_bounty1800_nn_boundaries.py tests/test_issue236_fullwidth_semicolon.py tests/test_issue1957_single_attr_align_guard.py tests/test_issue194_make_multimodal_scene_capacity.py -q
python -X utf8 tests/test_hex_text_contracts.py
python -X utf8 tools/verify_manifest.py --strict
```

The 19 new tests give 16 failures / 3 passes on the untouched baseline and
19 passes on the repair. The independent crop recorder counts unique input
cell IDs actually passed to classification: all 400 shapes with axes 1..20,
44,100 cells in total, are covered exactly once. The baseline has only 36
complete shapes, 76 small-axis errors, and 8,379 omitted cells. The repair
has 400 complete shapes, zero errors, zero omissions, and zero repetitions.

A real `HexHierNet` with transparent center-only/color weights also verifies
the public pipeline. A 48×48 RGB image whose last column is blue genuinely
samples into the final lattice column. The baseline reports only
`r0/circle|red/share=0`; the repair reports blue in `r2/r5/r8`. Consequently
`multimodal_check(..., "右中有蓝色圆形", n_expected=3)` changes from
`REJECT/align=0/count_match=False` to `ACCEPT/align=1/count_match=True`.
The four tested lattice widths are 16, 17, 19, and 23.

Five named mutations are recorded in
`tests/fixtures/spatial_detect_complete_partition_mutations.json`: drop
remainder rows, drop remainder columns, truncate energy only, truncate
classification only, and classify an empty quadrant. Each is killed by its
named business assertion (one failed test, zero errors/skips).

The combined new/adjacent suite passes 65 tests in 22.15 seconds on Windows,
Python 3.12 and NumPy 2.5.5; the script contracts pass 14/14. The manifest
reports 64/64 matches; time-core lint keeps 23 existing findings and adds none.
The unchanged gate/collection/manifest/import/encoding/lint guards pass all
128 tests in 276.56 seconds (13 existing syntax-warning reports).
Full-repository and Linux CI verification remain separate release gates.

## Public-path benchmark

`bench/bench_spatial_detect_partition.py` uses an unchanged seeded
`HexHierNet`, three warmup samples and nine retained latency samples, plus
three separate `tracemalloc` runs. It checks deterministic results and
unchanged input/model arrays. Run the same script against both source roots
in fresh processes. Compare input/weight hashes before reading the output
hashes or timings. The divisible-grid output is byte-identical; uneven grids
intentionally process cells absent from the original crops.

The initial Windows A/B run (one numerical-library thread) measured:

| Batch × lattice | Baseline ms/call | Repair ms/call | Peak bytes, both |
| --- | ---: | ---: | ---: |
| 2 × 18×18 | 0.517 | 0.462 | 179,696 |
| 2 × 18×16 | 0.379 | 0.759 | 160,496 |
| 2 × 65×67 | 2.263 | 2.101 | 2,317,280 |

The 18×16 case now classifies three regions per sample instead of one; its
2.00× latency is a visible cost of recovering the missing evidence. These
are descriptive shared-host observations, not evidence of statistically
significant speed changes. Traced allocations do not measure total RSS.

The fix does not add validation for empty spatial dimensions, nonfinite
inputs, or arbitrary network-protocol violations. The wider architecture
still applies its existing threshold and class-selection policies.
