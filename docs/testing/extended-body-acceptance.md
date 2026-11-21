# Extended body acceptance

The existing full gate installs `.[full,dev]`. Its green result may include
skipped sparse-backend tests and physical comparisons that need a historical
source tree. It also runs from an editable source checkout. The separate
`extended-body-acceptance.yml` workflow exercises those delivery forms without
changing the dependency combinations of the three existing full-gate lanes.

The historical/sparse job runs on Ubuntu with Python 3.12 and installs the
declared `dev,core-sparse` extras. It authenticates commit
`124ea433fe9a2ad2fd66a7360df85f6d61bab793`, checks its full tree identity
`fb2cefe048c0c4798c9016afd7256d86f0e664b8`, then exports its actual `lingshu/`
files. Every exported file must match the Git blob from that tree. The four
existing physical baseline/parity/mutation tests must execute and pass.
The three existing activation files run with real SciPy; their tests assert
that the sparse path builds real CSR matrices. All 16 sparse-specific contract
cases must appear in the result, and no test in either leg may skip.

The installed-wheel job runs on Ubuntu and Windows with Python 3.12. It builds
a wheel in a disposable source copy and creates two fresh virtual environments
outside the checkout. The light environment installs the wheel without extras
and must have neither NumPy nor Pillow. The full environment installs the same
wheel's declared full extra. Each runs the actual scenario CLI in isolated Python
from outside source, for `all` and each of its five views. The JSON checks retain
scope, provenance, qualification, world state and causal ordering. Full acceptance
also checks the 500 × 500 PNG pixels and consumes the installed NN asset through
`load_consolidated` and a constant-field convolution oracle. Product imports must
come from the new environment's site-packages. Each fresh environment must also
pass `pip check`. No editable install or inherited
`PYTHONPATH` is accepted for either wheel environment.

## Local reproduction

Use Python 3.12 or above. Create an isolated environment for the contracts leg
so the optional installation does not change an existing application environment:

```sh
python -m venv /path/outside/source/contract-venv
# Use that environment's Python below (Scripts/python.exe on Windows).
/path/outside/source/contract-venv/bin/python -m pip install -e ".[dev,core-sparse]"
/path/outside/source/contract-venv/bin/python -X utf8 tools/testing/extended_body_acceptance.py contracts --fetch-baseline --output /path/outside/source/contract-evidence

python -X utf8 tools/testing/extended_body_acceptance.py selftest
python -X utf8 tools/testing/extended_body_acceptance.py wheel --output /path/outside/source/wheel-evidence
```

Output directories must be new or empty and outside the repository. Existing
evidence is preserved. The driver retains raw command stdout/stderr, exact source
head, authenticated baseline records, JUnit, wheel identity, CLI JSON and PNG.
It fails on missing/empty results, skipped or missing required cases, changed
baseline files, failed commands or unverified installed-package output. Its
selftest exercises eight rejected evidence cases and three positive controls,
including a clean source commit change. Both the Git commit/tree and working-tree
status must agree before and after acceptance.

These are synthetic acceptance journeys, not accuracy measurements on real camera
assets or production memories. The workflow needs public package indexes and,
when the historical object is not locally present, access to the pinned Git
commit. An unavailable prerequisite is a failure of this required extended leg;
it is not reported as a successful skipped acceptance.
