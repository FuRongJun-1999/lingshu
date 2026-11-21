# M5.5 shared quadrant integration

The main detector already covers every lattice cell with scaled thirds and
compares each region's evidence against its actual area. The pending M5.5 bridge
had copied the older fixed-width crops and uniform one-ninth gate. Its new
ingestion path could lose evidence in the final rows or columns even while
`spatial_detect` returned the correct detection.

Both callers now use the same private quadrant iterator. Their distinct output
fields, confidence precision and calibration rules remain at their original
entrypoints. The existing spatial mutation fixtures target that iterator after
the code move; their assertions are unchanged.

```sh
python -X utf8 -m pytest tests/test_m55_perception_knowledge_bridge.py \
  tests/test_m55_partition_consistency.py \
  tests/test_spatial_detect_complete_partition.py -q
python -X utf8 tests/test_m55_perception_knowledge_bridge.py --mutate
python -X utf8 tools/testing/replay_save_guards.py \
  --spec tests/fixtures/spatial_detect_complete_partition_mutations.json \
  --mutate --output OUTPUT
```

The additional route checks include an independent cell-ownership calculation
over 324 row/column combinations, final-cell evidence, area weighting, and the
full card-ingestion-to-knowledge-query path. Before sharing the crops, 15 of
these 20 cases failed; all pass after the integration correction.
The original detector's outputs also matched exact main in 72 controlled
random, uniform, boundary, empty and tiny lattice cases.

This correction is part of PR #1439's integration with current main. Its existing
bounty intake #2727 was accepted and settled; no new intake is associated with
the integration.