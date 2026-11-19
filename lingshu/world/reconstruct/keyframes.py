from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def select_depth_frames(poses: dict, tracks: list[dict], *, all_views: bool = False) -> tuple[list[str], dict]:
    """Keep distinct views and protect local and repeat sparse-track coverage."""
    names = sorted(poses, key=lambda name: poses[name]["source_index"])
    settings = {
        "min_views": 4,
        "min_unique_track_ratio": 0.98,
        "min_repeated_track_ratio": 0.85,
        "min_local_track_ratio": 0.85,
        "min_local_tile_ratio": 0.75,
        "max_baseline_to_depth": 0.25,
        "max_view_angle_deg": 20.0,
    }
    if all_views or len(names) <= settings["min_views"]:
        return names, {"mode": "all" if all_views else "coverage", "thresholds": settings}

    index = {name: i for i, name in enumerate(names)}
    ids, tile_ids, tiles, depths = ([[] for _ in names] for _ in range(4))
    support = np.zeros(len(tracks), dtype=np.int32)
    for track_id, track in enumerate(tracks):
        seen_views, seen_tiles, seen_depths = set(), set(), set()
        for obs in track["observations"]:
            name = obs["image"]
            if name not in index:
                continue
            i, pose = index[name], poses[name]
            x, y = obs["pixel_xy"]
            col = min(3, max(0, int(4 * x / pose["width"])))
            row = min(3, max(0, int(4 * y / pose["height"])))
            # Independent views and image footprints have different units.
            if i not in seen_views:
                ids[i].append(track_id)
                support[track_id] += 1
                seen_views.add(i)
            tile = 4 * row + col
            if (i, tile) not in seen_tiles:
                tile_ids[i].append(track_id)
                tiles[i].append(tile)
                seen_tiles.add((i, tile))
            if i not in seen_depths and np.isfinite(obs["camera_z"]) and obs["camera_z"] > 0:
                depths[i].append(obs["camera_z"])
                seen_depths.add(i)

    ids = [np.asarray(a, dtype=np.intp) for a in ids]
    tile_ids = [np.asarray(a, dtype=np.intp) for a in tile_ids]
    tiles = [np.asarray(a, dtype=np.intp) for a in tiles]
    tile_counts = [np.bincount(a, minlength=16) for a in tiles]
    start_unique = int(np.count_nonzero(support))
    start_repeated = int(np.count_nonzero(support >= 2))
    if not start_repeated:
        return names, {"mode": "coverage", "thresholds": settings, "reason": "no repeat tracks"}
    unique, repeated = start_unique, start_repeated
    extrinsics = np.asarray([poses[name]["cam_from_world"] for name in names])
    centers = -np.einsum("nji,nj->ni", extrinsics[:, :, :3], extrinsics[:, :, 3])
    forward = extrinsics[:, 2, :3]
    tree = cKDTree(centers)
    min_cos = np.cos(np.radians(settings["max_view_angle_deg"]))
    neighbors = [set() for _ in names]
    dependents = [set() for _ in names]
    for i, values in enumerate(depths):
        if len(values) < 30:
            continue
        radius = settings["max_baseline_to_depth"] * np.median(values)
        for j in tree.query_ball_point(centers[i], radius):
            if j != i and forward[i] @ forward[j] >= min_cos:
                neighbors[i].add(j)
                dependents[j].add(i)

    selected = set(range(len(names)))
    omitted: set[int] = set()
    local = {}

    def local_ratio(i: int) -> float:
        nearby = neighbors[i] & selected
        if not nearby or not len(ids[i]):
            return 0.0
        nearby_ids = np.concatenate([ids[j] for j in sorted(nearby)])
        shared = np.isin(ids[i], nearby_ids)
        shared_tiles = np.isin(tile_ids[i], nearby_ids)
        tile_hits = np.bincount(tiles[i][shared_tiles], minlength=16)
        populated = tile_counts[i] >= 8
        if np.any(tile_hits[populated] < settings["min_local_tile_ratio"] * tile_counts[i][populated]):
            return 0.0
        return float(shared.mean())

    # Remove low-support intermediate frames first; keep both ends of the capture.
    for i in sorted(range(1, len(names) - 1), key=lambda j: (len(ids[j]), j)):
        if len(selected) <= settings["min_views"]:
            break
        if len(ids[i]) < 30:
            continue
        current = support[ids[i]]
        next_unique = unique - int(np.count_nonzero(current == 1))
        next_repeated = repeated - int(np.count_nonzero(current == 2))
        if (next_unique < settings["min_unique_track_ratio"] * start_unique
                or next_repeated < settings["min_repeated_track_ratio"] * start_repeated):
            continue
        selected.remove(i)
        affected = (omitted & dependents[i]) | {i}
        ratios = {j: local_ratio(j) for j in affected}
        if min(ratios.values()) < settings["min_local_track_ratio"]:
            selected.add(i)
            continue
        omitted.add(i)
        local.update(ratios)
        support[ids[i]] -= 1
        unique, repeated = next_unique, next_repeated

    return [names[i] for i in sorted(selected)], {
        "mode": "coverage",
        "thresholds": settings,
        "unique_track_ratio": unique / start_unique,
        "repeated_track_ratio": repeated / start_repeated,
        "omitted_view_local_track_ratio": {names[i]: local[i] for i in sorted(omitted)},
    }
