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
three cells cannot become dominant. Divisible dimensions keep their original
slices. Because complete uneven partitions have unequal areas, the extra-
region gate uses each region's area fraction times the unchanged `exist_mult`
instead of a fixed `1/9`; the divisible-grid gate remains exactly `1/9`.
Reported shares remain true total-energy fractions, and the dominant region
still maximizes total energy. Model weights and training are unchanged.
No original image assets are required for these tests.

## Reproduction and regression

```sh
python -X utf8 -m pytest tests/test_spatial_detect_complete_partition.py -q
python -X utf8 -m pytest tests/test_hex_text.py tests/test_hex_text_thresholds.py tests/test_bounty1800_nn_boundaries.py tests/test_issue236_fullwidth_semicolon.py tests/test_issue1957_single_attr_align_guard.py tests/test_issue194_make_multimodal_scene_capacity.py -q
python -X utf8 tests/test_hex_text_contracts.py
python -X utf8 tools/verify_manifest.py --strict
python -X utf8 -B tools/testing/replay_save_guards.py --spec tests/fixtures/spatial_detect_complete_partition_mutations.json --baseline d74848fb33dbff1d719959df19d1266a1ac60f44 --mutate --output artifacts/spatial-partition-replay
```

The 70 new tests give 59 failures / 11 passes on the untouched baseline and
70 passes on the repair. The independent crop recorder counts unique input
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
The four tested lattice widths are 16, 17, 19, and 23. Uniform red and zero
fields at 1/2-cell axes, 4×4, 5×5, 8×8, and 18×16 are tested through both the
default and transparent real networks: area differences must not create extra
detections, and confidence/share values stay finite. The uniform-field
positive-value shares are also checked against independent per-cell area counts.

Six named mutations are recorded in
`tests/fixtures/spatial_detect_complete_partition_mutations.json`: drop
remainder rows, drop remainder columns, truncate energy only, truncate
classification only, allow an empty dominant region, and restore the fixed
`1/9` gate. Each is killed by its
named business assertion (one failed test, zero errors/skips).

The final combined new/adjacent suite passes 116 tests in 22.13 seconds on Windows,
Python 3.12 and NumPy 2.5.5; the script contracts pass 14/14. The manifest
reports 64/64 matches; time-core lint keeps 23 existing findings and adds none.
Adding the unchanged encoding, document-count and manifest guards gives
143 passing tests in 33.65 seconds (nine existing syntax-warning reports).
Before the area-baseline correction, the unchanged gate/collection/manifest/
import/encoding/lint guards passed all 128 tests in 276.56 seconds (13 existing
syntax-warning reports); final delivery rechecks remain separate below.
Full-repository and Linux CI verification remain separate release gates.

## Public-path benchmark

`bench/bench_spatial_detect_partition.py` uses an unchanged seeded
`HexHierNet`, three warmup samples and nine retained latency samples, plus
three separate `tracemalloc` runs. It checks deterministic results and
unchanged input/model arrays. Run the same script against both source roots
in fresh processes. Compare input/weight hashes before reading the output
hashes or timings. The divisible-grid output is byte-identical; uneven grids
intentionally process cells absent from the original crops.

The final Windows A/B run (one numerical-library thread) measured:

| Batch × lattice | Baseline ms/call | Repair ms/call | Baseline peak bytes | Repair peak bytes |
| --- | ---: | ---: | ---: | ---: |
| 2 × 18×18 | 0.548 | 0.467 | 179,696 | 179,736 |
| 2 × 18×16 | 0.362 | 0.350 | 160,496 | 160,536 |
| 2 × 65×67 | 2.033 | 2.119 | 2,317,280 | 2,317,320 |

Both final source variants classify the same number of regions in these
workloads (18×18: 2/1; the two uneven cases: 1/1 per batch). Uneven-grid output
values intentionally change because crops now include the missing cells.
The earlier partition-only R1 performance evidence is retained separately;
the final R2 also corrects its uniform-area gate artifact. These are
descriptive shared-host observations, not evidence of statistically
significant speed changes. Traced allocations do not measure total RSS.

Empty spatial extents return no regions; this is not a new validation API.
The fix does not add validation for nonfinite inputs or arbitrary
network-protocol violations. The wider architecture
still applies its existing threshold and class-selection policies.
