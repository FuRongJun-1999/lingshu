# World-Atta static replay provider

`world_atta.py` is a read-only external actual provider for the synthetic
World-Atta handoff. It consumes only `oracle.jsonl` through the existing
`wm_bridge.oracle.read_oracle()` and `frame_slice` contract. It does not import
Atta, Blender, NumPy, or lingshu core, and it never writes memory/state.

```python
from wm_bridge.world_atta import load_scene, actual_at_tick, as_observations

scene = load_scene("/path/to/dataset-final", "s01_flange")
actual = actual_at_tick(scene, tick=0)
observations = as_observations(scene, tick=0)
```

The provider is deliberately **static**: only `export_frame=0` and `tick=0`
are accepted. A non-zero tick raises `WorldAttaProviderError`; it does not
return an empty frame. Rows are sorted by entity id for deterministic replay,
and the original oracle row is retained in each observation under `oracle`.
`kind=synthetic_ground_truth` and `provenance.synthetic=true` are explicit: the
fixture is an E1 synthetic actual source, not real-camera evidence, a dynamic
world, a prediction, or an independent reconstruction benchmark.

No World-Atta dataset is vendored in this repository. Tests construct a tiny
JSONL fixture in a temporary directory; a local checkout can be replayed by
pointing `load_scene()` at its dataset root.
