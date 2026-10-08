"""Visual scenarios use visible pixels, not the renderer's success log, as evidence."""

import numpy as np
import pytest

from lingshu.gen.hexgen_c1_real import measure_item
from lingshu.gen.hexgen_self_source import render_prompt_v3
from lingshu.nn.hex_composite import extract_attributes
from lingshu.nn.hex_gen import ATTR_CALIB_96, generate_from_text


@pytest.mark.parametrize(
    ("transform", "red_cell", "green_cell"),
    [(None, (0, 0), (2, 2)), ("rot90cw", (0, 2), (2, 0))],
    ids=["requested-layout", "clockwise-layout"],
)
def test_text_layout_and_sizes_survive_pixel_readback(
    transform, red_cell, green_cell
):
    # The expected cells come from the sentence and the meaning of a clockwise
    # turn. Do not derive them from parts, logs, or the production zone mapping.
    generated = generate_from_text(
        "左上有红色实心小圆,右下有绿色实心大圆",
        size=96,
        seed=3,
        transform=transform,
    )
    image = generated["image"].astype(int)
    areas = []
    for color, channel, cell, expected_size in (
        ("red", 0, red_cell, "small"),
        ("green", 1, green_cell, "large"),
    ):
        # Coarse channel dominance separates these saturated objects from the
        # dark background without reusing the renderer's palette or its log.
        other_channels = [c for c in range(3) if c != channel]
        mask = image[..., channel] > image[..., other_channels].max(axis=2) + 100
        rows, columns = np.nonzero(mask)
        assert rows.size > 0, f"The requested {color} circle is absent"
        assert set(zip(rows // 32, columns // 32)) == {cell}
        areas.append(rows.size)

        zone = f"r{cell[0] * 3 + cell[1]}"
        attributes = extract_attributes(
            generated["image"], zone, color, ATTR_CALIB_96, shape_hint="circle"
        )
        assert attributes["pattern"] == "solid"
        assert attributes["size"] == expected_size

    assert areas[1] > areas[0], "The green circle should visibly exceed the red one"


def test_generated_subjects_are_counted_and_located_from_pixels():
    # This public synthetic palette keeps the scenario independent of the
    # author's private calibration corpus. Placement and painting are real.
    image = render_prompt_v3(
        (2, "circle", "blue", "plain", ("r2", "r6")),
        size=128,
        pal={"blue": (40, 60, 220)},
    )

    # Give the reader only image pixels: no labels, intended positions, masks,
    # or renderer diagnostics can supply the answer to this readback.
    objects, _ = measure_item({"img": image.astype(np.float64)})

    assert len(objects) == 2
    assert {obj["cell"] for obj in objects} == {"r2", "r6"}
    for obj in objects:
        red, green, blue = obj["core_rgb"]
        assert blue > max(red, green) + 100
        assert not obj["clipped"], "Both requested subjects should fit on the canvas"
