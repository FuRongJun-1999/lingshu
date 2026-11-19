"""Auto depth coverage counts distinct views, tracks and image tiles."""

from copy import deepcopy

import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")

from lingshu.world.reconstruct.keyframes import select_depth_frames


def _poses(count, *, distant_ends=False):
    centers = [index / 10 for index in range(count)]
    if distant_ends:
        centers[0], centers[-1] = -20.0, 20.0
    return {
        f"view-{index}": {
            "source_index": index,
            "width": 100,
            "height": 100,
            "cam_from_world": [[1, 0, 0, -center], [0, 1, 0, 0], [0, 0, 1, 0]],
        }
        for index, center in enumerate(centers)
    }


def _track(views, pixel=(10.0, 10.0)):
    return {"observations": [
        {"image": f"view-{index}", "pixel_xy": list(pixel), "camera_z": 10.0}
        for index in views
    ]}


def test_repeating_same_view_features_preserves_selection_ratios_and_raw_tracks():
    poses = _poses(5)
    tracks = [_track(range(5)) for _ in range(60)]
    expected = select_depth_frames(poses, tracks)
    assert expected[0] == ["view-0", "view-2", "view-3", "view-4"]
    assert expected[1]["unique_track_ratio"] == 1.0
    assert expected[1]["repeated_track_ratio"] == 1.0

    repeated = deepcopy(tracks)
    for track in repeated:
        track["observations"].extend([deepcopy(track["observations"][1])] * 3)
    untouched = deepcopy(repeated)
    assert select_depth_frames(poses, repeated) == expected
    assert repeated == untouched


@pytest.mark.parametrize("duplicate_pixel", [(10.0, 10.0), (90.0, 90.0)])
def test_repeated_same_view_does_not_create_independent_repeat_support(duplicate_pixel):
    poses = _poses(5)
    tracks = ([_track(range(5)) for _ in range(80)]
              + [_track([1, 2]) for _ in range(20)]
              + [_track([3, 4]) for _ in range(20)])
    expected = select_depth_frames(poses, tracks)
    assert expected[0] == list(poses)

    repeated = deepcopy(tracks)
    for track in repeated[80:100]:
        duplicate = deepcopy(track["observations"][0])
        duplicate["pixel_xy"] = list(duplicate_pixel)
        track["observations"].append(duplicate)
    assert select_depth_frames(poses, repeated) == expected


def test_local_track_ratio_is_not_weighted_by_repeated_features():
    poses = _poses(6, distant_ends=True)
    tracks = ([_track(range(6)) for _ in range(80)]
              + [_track([0, 1, 5]) for _ in range(20)]
              + [_track([0, 2, 3, 4, 5]) for _ in range(200)])
    expected = select_depth_frames(poses, tracks)
    assert "view-1" in expected[0]  # Only 80/100 tracks recur in nearby views.

    repeated = deepcopy(tracks)
    for track in repeated[:80]:
        track["observations"].append(deepcopy(track["observations"][1]))
    assert select_depth_frames(poses, repeated) == expected


def test_cross_tile_features_retain_each_track_tile_without_an_arbitrary_pixel():
    poses = _poses(6, distant_ends=True)
    tracks = ([_track(range(6)) for _ in range(90)]
              + [_track(range(6), pixel=(90.0, 90.0)) for _ in range(10)]
              + [_track([0, 1, 5]) for _ in range(10)]
              + [_track([0, 2, 3, 4, 5]) for _ in range(200)])
    for track in tracks[100:110]:
        duplicate = deepcopy(track["observations"][1])
        duplicate["pixel_xy"] = [90.0, 90.0]
        track["observations"].append(duplicate)

    selected = select_depth_frames(poses, tracks)
    # Overall nearby overlap is 100/110, but tile 15 only has 10/20 tracks.
    assert "view-1" in selected[0]
    reversed_observations = deepcopy(tracks)
    for track in reversed_observations:
        track["observations"].reverse()
    assert select_depth_frames(poses, reversed_observations) == selected


def test_repeated_features_do_not_reweight_the_camera_depth_median():
    poses = _poses(6, distant_ends=True)
    for index, center in ((1, 0.0), (2, 1.0), (3, 1.1), (4, 1.2)):
        poses[f"view-{index}"]["cam_from_world"][0][3] = -center
    tracks = ([_track(range(6)) for _ in range(60)]
              + [_track([0, 2, 3, 4, 5]) for _ in range(200)])
    for track in tracks[:40]:
        for observation in track["observations"]:
            observation["camera_z"] = 1.0
    expected = select_depth_frames(poses, tracks)
    assert "view-1" in expected[0]  # Median depth 1 gives no nearby frame.

    repeated = deepcopy(tracks)
    for track in repeated[40:60]:
        track["observations"].extend([deepcopy(track["observations"][1])] * 3)
    assert select_depth_frames(poses, repeated) == expected


@pytest.mark.parametrize(("count", "all_views", "mode"), [(5, True, "all"), (4, False, "coverage")])
def test_all_and_small_inputs_return_before_inspecting_tracks(count, all_views, mode):
    class UnreadableTracks:
        def __iter__(self):
            raise AssertionError("early return must not inspect tracks")

        def __len__(self):
            raise AssertionError("early return must not inspect tracks")

    poses = _poses(count)
    selected, report = select_depth_frames(poses, UnreadableTracks(), all_views=all_views)
    assert selected == list(poses)
    assert report["mode"] == mode
    assert set(report) == {"mode", "thresholds"}
