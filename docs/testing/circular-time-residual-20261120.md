# Circular fitting: local time and complete residuals

Baseline: `d74848fb33dbff1d719959df19d1266a1ac60f44`.

`fit_circular` documents a horizontal circle plus uniform vertical drift. Its
dictionary is consumed at `t_ref`, but the old vertical regression used absolute
time. Moving an otherwise identical helix from time 0 to time 10 changed its
initial height from 3 to -17. At Unix epoch 1700000000, the angle regression's
normal equations became singular. The repair fits both angle and height against
`t - t_ref`. The model schema, axis order, unwrapping and circle estimator remain
the same.

The old metric also discarded the vertical residual and divided by `2*n`, while
other candidates use three-coordinate RMS. A circle with vertical acceleration
20 reported `5.32e-16` despite an independent coordinate RMS of `4.18563` and
maximum position error `14.375`. `classify` consequently selected a supposedly
perfect helix over the accelerated model. The residual now evaluates the actual
three-coordinate predictions and divides by `3*n`.

These are distinct from the world-axis ordering fix in upstream #786 / PR #829.
The previous axis repair is retained. Searches of 1307 upstream issues/PRs and
2194 bounty issues, plus exact circular searches and #786/#817 comments, found
the axis-order, damped-sign and truth-helper reports, not these two symptoms.
This is one cohesive repair and one contribution intake, not separate claims
for every affected parameter or guard.

## Reproduction and guards

```sh
python -X utf8 -m pytest tests/test_circular_time_residual_guard.py -q
python -X utf8 tools/testing/replay_save_guards.py \
  --spec tests/fixtures/circular_time_residual_mutations.json \
  --baseline d74848fb33dbff1d719959df19d1266a1ac60f44 --mutate \
  --output artifacts/circular-mutations
python -X utf8 bench/circular_time_residual.py \
  --baseline d74848fb33dbff1d719959df19d1266a1ac60f44 \
  --output artifacts/circular-benchmark.json
```

The six new guard groups all pass; the unchanged baseline fails all six through
semantic assertions. Six named mutants each fail the exact selected assertion,
with no import errors or skips. Coverage includes all up axes, positive and
negative time origins, Unix epochs, irregular sampling, negative angular speed,
vertical regression against an independent centered-covariance oracle, planar
noise, classification and extrapolated position/velocity. Related physics
tests: **145 passed, 4 existing skips**. The skips are separate from this new
guard and are not counted as exercised scenarios.

## Accuracy and descriptive cost

The standalone benchmark covers 666 analytic cases. Fixed: **7146 assertions,
zero failures**, maximum sampled position error `1.484e-13` and velocity error
`2.046e-13`. Baseline: 4986 executed assertions, 1530 failed, including 216 fit
exceptions. Those exceptions prevent 2160 further prediction checks; they are
explicitly recorded as unexecuted, not counted as passes. Both variants have
7146 planned checks.

Two executions on the same shared Windows host retained nine interleaved timing
samples per size, using identical zero-epoch inputs that both variants can fit.
The complete raw samples are evidence, not a timing threshold or a speed claim.

| Samples per fit | Fixed / baseline median latency, two runs | Traced peak bytes, baseline -> fixed |
| --- | --- | --- |
| 25 | 1.003–1.032 | 5168 -> 6136 |
| 100 | 1.017–1.037 | 10008 -> 13384 |
| 1000 | 1.000–1.036 | 71316 -> 104228 |

The two retained local-time arrays increase peak Python allocations. This is a
correctness repair, with these costs disclosed. Spatial Kasa conditioning,
tilted circles and angle aliasing are outside this repair and outside the
accuracy claim. Existing serialized models are not rewritten; an already stored
incorrect intercept needs refitting from its observations.

The time-core baseline changes only by the four added source lines below this
function: all 23 existing findings remain, with zero new or stale entries. No
new lint exemptions or test deselections are introduced.
