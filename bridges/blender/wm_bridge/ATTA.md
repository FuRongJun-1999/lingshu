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
python -m wm_bridge.cli validate-atta /path/to/delivery-directory \
  --allow-partial   # explicit opt-in; partial delivery is not complete evidence
```

Exit status is `0` only when the manifest is structurally valid, its verdict is
accepted, and every successful file row matches its byte count and SHA-256.
By default the delivery must also be complete: no export errors or failed rows.
Explicit `--allow-partial` accepts declared failures without hashing those rows;
it does not waive validation of successful rows or malformed JSON (including
non-finite exponent values such as `1e999` at any nesting depth).
Otherwise exit status is `2`. Unknown schemas and methods fail closed. The scene root is
a path-containment boundary. Relative paths are preferred. Atta may write
absolute paths; these are accepted only when they resolve inside the explicitly
provided root. For portable handoffs, place the manifest and all referenced
files below one directory and use relative paths.

## `files[]` is one row per file, not per format

Atta exports a separate mesh file for every assembly part
(`atta/assembly.py:1534-1544` writes `<stem>_<part>.obj` / `.stl`), so an
eight-part assembly produces eight rows with `format: "obj"`. The verifier
therefore keys uniqueness on `(format, part)`, not on `format` alone, and
propagates `part` into the normalized summary so provenance survives. A row
with `ok: false` is a **declared export failure**: it may omit `path`, `bytes`
and `sha256`, is never hash-verified, and is reported as such instead of being
misdiagnosed as a missing or corrupt file.

## Adapter `status` includes export errors and declared failed rows

Atta computes `status = "ok" if files and not errors else ("partial" if files
else "failed")` (`atta/engine.py:2024`, `:2169`). A manifest can therefore carry
`verify.verdict: "pass"` while `errors` still lists omitted exports — for
example a STEP part skipped because its B-rep backend was unavailable. Such a
delivery is **partial**, and the adapter rejects it by default rather than
silently promoting it to complete evidence. Pass `allow_partial=True` (or
`--allow-partial`) to accept it deliberately; the returned summary then carries
`status`, `errors`, `file_count`, `verified_file_count` and
`failed_file_count` so a caller cannot mistake a partial asset for a whole one.
The adapter also reports `partial` whenever any file row has `ok: false`, even
if `errors` is empty, absent or null. This is an adapter completeness check,
not a change to Atta's producer formula. Absent/null `errors` remain normalized
to `[]`; nonempty `errors` still make an otherwise successful delivery partial.
Empty `files` remain invalid even with `allow_partial=True`.

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
