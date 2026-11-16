# Atta delivery manifest adapter

`atta_receipt.py` is a read-only, pure-standard-library verifier for the public
Atta `atta.deliver/v3` JSON manifest. It is intentionally placed in the
out-of-wheel `wm_bridge` adapter: lingshu core does not import `atta` or `bpy`,
and no Atta source is copied here. The field mapping follows the documented
Atta interfaces and `atta/engine.py` delivery output (baseline observed at
Atta commit `916c4a04`); it is not an Atta runtime API.

```sh
python -m wm_bridge.cli validate-atta /path/to/delivery-directory
python -m wm_bridge.cli validate-atta /path/to/delivery-directory \
  --manifest model_manifest.json
python -m wm_bridge.cli validate-atta /path/to/delivery-directory \
  --allow-degraded  # explicit opt-in; degraded is not pass
```

Exit status is `0` only when the manifest is structurally valid, its verdict is
accepted, and all listed output files match the declared byte count and SHA-256;
otherwise it is `2`. Unknown schemas and methods fail closed. The scene root is
a path-containment boundary. Relative paths are preferred. Atta may write
absolute paths; these are accepted only when they resolve inside the explicitly
provided root. For portable handoffs, place the manifest and all referenced
files below one directory and use relative paths.

The summary preserves `model_id`, `spec_hash`, `mesh_hash`, verification verdict,
file hashes, and an explicit synthetic-ground-truth marker. This adapter does
**not** open the saved verification receipt path, reconstruct the Atta mesh,
prove an RGB-to-3D reconstruction, or promote a tessellated GLB/OBJ/STL to exact
CAD evidence. `exact: false` on a delivery manifest means exported mesh files
are derivatives; the geometry verdict remains Atta's verification receipt.
A passing manifest is not independent evidence if its producer is also the
only source of the claimed ground truth.

It does not call the existing `wm_bridge.manifest.verify`: Atta file records
use `{path, bytes, sha256}`, while wm_bridge's own manifest uses raw/LF hashes.
These are distinct file protocols.
