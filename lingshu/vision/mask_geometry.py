"""Geometry of connected regions in a caller-supplied boolean foreground mask."""
from __future__ import annotations

from collections import deque
from heapq import heappush, heapreplace

__all__ = ["analyze_binary_components"]

_MAX_PIXELS = 262_144
_MAX_COMPONENTS = 128
_NEIGHBORS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_NEIGHBORS_8 = _NEIGHBORS_4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _mask_dimensions(mask) -> tuple[int, int, int]:
    if not isinstance(mask, (list, tuple)):
        raise TypeError("mask must be a list or tuple of rows")
    height = len(mask)
    if height == 0:
        raise ValueError("mask must have positive height")

    width = None
    for row in mask:
        if not isinstance(row, (list, tuple)):
            raise TypeError("mask rows must be lists or tuples")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("mask must be rectangular")
    if not width:
        raise ValueError("mask must have positive width")

    pixel_count = height * width
    if pixel_count > _MAX_PIXELS:
        raise ValueError(f"mask exceeds the {_MAX_PIXELS} pixel budget")
    return width, height, pixel_count


def analyze_binary_components(
    mask, *, connectivity=4, min_area=1, max_components=128
) -> dict:
    """Summarize the True regions of a nonempty rectangular bool mask.

    Use 4- or 8-neighbor connectivity without filling holes. Bounding boxes
    are half-open [left, top, right, bottom], centroids use pixel centers, and
    touches_boundary refers to the outermost row or column of the mask.
    Report at most max_components regions with area >= min_area, ordered by
    descending area, then ascending bbox top and left. Filtering and truncation
    do not change the total foreground pixel or unfiltered region counts.

    Raise TypeError for invalid mask containers or cells, and ValueError for
    invalid dimensions, pixel budget, or keyword arguments.
    """
    width, height, pixel_count = _mask_dimensions(mask)
    if type(connectivity) is not int or connectivity not in (4, 8):
        raise ValueError("connectivity must be the integer 4 or 8")
    if type(min_area) is not int or not 1 <= min_area <= _MAX_PIXELS:
        raise ValueError(f"min_area must be an integer in 1..{_MAX_PIXELS}")
    if type(max_components) is not int or not 1 <= max_components <= _MAX_COMPONENTS:
        raise ValueError(f"max_components must be an integer in 1..{_MAX_COMPONENTS}")

    for row in mask:
        for cell in row:
            if type(cell) is not bool:
                raise TypeError("mask cells must be bool")

    visited = bytearray(pixel_count)
    queue = deque()
    neighbors = _NEIGHBORS_4 if connectivity == 4 else _NEIGHBORS_8
    foreground_pixels = component_count = filtered_count = 0
    best = []

    for start_y, row in enumerate(mask):
        for start_x, foreground in enumerate(row):
            seed = start_y * width + start_x
            if not foreground or visited[seed]:
                continue

            visited[seed] = 1
            queue.append(seed)
            left = right = start_x
            top = bottom = start_y
            area = sum_x = sum_y = 0
            while queue:
                y, x = divmod(queue.popleft(), width)
                area += 1
                sum_x += x
                sum_y += y
                left = min(left, x)
                right = max(right, x)
                bottom = max(bottom, y)
                # The row-major seed already gives the minimum y.
                for dx, dy in neighbors:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        index = ny * width + nx
                        if not visited[index] and mask[ny][nx]:
                            visited[index] = 1
                            queue.append(index)

            foreground_pixels += area
            component_count += 1
            if area < min_area:
                continue
            filtered_count += 1

            # The weakest retained entry is at the root. Discovery order
            # breaks otherwise identical ranks without comparing dicts.
            rank = (area, -top, -left, -component_count)
            if len(best) == max_components and rank <= best[0][0]:
                continue
            component = {
                "area": area,
                "bbox": [left, top, right + 1, bottom + 1],
                "centroid": [
                    (2 * sum_x + area) / (2 * area),
                    (2 * sum_y + area) / (2 * area),
                ],
                "touches_boundary": (
                    left == 0 or top == 0 or right == width - 1 or bottom == height - 1
                ),
            }
            if len(best) < max_components:
                heappush(best, (rank, component))
            else:
                heapreplace(best, (rank, component))

    components = [component for _, component in sorted(best, reverse=True)]
    reported_count = len(components)
    return {
        "width": width,
        "height": height,
        "pixel_count": pixel_count,
        "foreground_pixels": foreground_pixels,
        "connectivity": connectivity,
        "min_area": min_area,
        "component_count": component_count,
        "filtered_component_count": filtered_count,
        "reported_component_count": reported_count,
        "omitted_component_count": filtered_count - reported_count,
        "components": components,
    }
