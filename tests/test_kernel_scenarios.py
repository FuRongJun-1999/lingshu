"""A trained visual kernel remains usable after its JSON handoff."""

import json

import numpy as np

from lingshu.nn.hex_cnn import (
    DEFAULT_KERNELS,
    hex_conv,
    image_to_grid,
    load_consolidated,
    save_consolidated,
    selfsup_finetune,
)


def test_trained_kernel_can_be_consolidated_and_reused(tmp_path):
    # A public, synthetic sensor frame: alternating light and dark bands.
    image = np.full((64, 64, 3), 40, dtype=np.uint8)
    image[:, 16:32] = 220
    image[:, 48:64] = 220
    training = selfsup_finetune(
        image, cells_across=16, epochs=3, lr=0.05, seed=7
    )
    learned = training["kernel"]
    assert not np.allclose(learned, DEFAULT_KERNELS["center_surround"])

    path = tmp_path / "sensor-kernels.json"
    save_consolidated(
        {"learned": learned},
        {"source": "synthetic-bands", "epochs": 3},
        str(path),
    )
    # Inspect the artifact a downstream consumer actually receives.
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["source"] == "synthetic-bands"
    assert payload["epochs"] == 3
    assert payload["consolidated"] is True

    restored = load_consolidated(str(path))["learned"]
    np.testing.assert_allclose(restored, learned, atol=5.1e-7, rtol=0)

    # Reuse on a different frame; the JSON handoff should preserve responses.
    held_out = np.full((64, 64, 3), 80, dtype=np.uint8)
    held_out[16:48, 16:48] = 180
    features, _ = image_to_grid(held_out, cells_across=16)
    features = features / 255.0
    before = hex_conv(features, learned)
    after = hex_conv(features, restored)
    # Seven weights rounded to six decimals, input bounded by one.
    np.testing.assert_allclose(after, before, atol=4e-6, rtol=0)

    # A uniform sensor field has an analytic response. This also checks that
    # equal pre/post outputs did not merely come from a nonfunctional operator.
    uniform = np.full((4, 4, 3), 0.5, dtype=np.float32)
    np.testing.assert_allclose(
        hex_conv(uniform, restored), 0.5 * float(learned.sum()), atol=4e-6, rtol=0
    )
