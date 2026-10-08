"""Regression coverage for JSON query reports (issue #46)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lingshu.nn.hex_query import ConnectionLibrary, query_match, save_report


def make_library():
    lib = ConnectionLibrary()
    lib.add_record("pair", [
        {"type": "circle", "zone": "top-left", "color": "red"},
        {"type": "circle", "zone": "bottom-right", "color": "blue"},
    ])
    lib.add_record("triangle", [
        {"type": "triangle", "zone": "*", "color": "green"},
    ])
    return lib


@pytest.mark.parametrize("detections,score,verdict", [
    ([{"obj": "circle|red", "pos": "r0"},
      {"obj": "circle|blue", "pos": "r8"}], 1.0, "ACCEPT"),
    ([{"obj": "circle|red", "pos": "r0"}], 0.5, "DEFER"),
    ([{"obj": "stripe|red", "pos": "r4"}], 0.0, "BLINDSPOT"),
])
def test_query_report_json_roundtrip(detections, score, verdict, tmp_path):
    result = query_match(detections, make_library())
    expected = [{
        "name": "pair", "score": score, "id": 0,
        "verdict": verdict, "margin": score,
        "all_scores": [
            {"name": "pair", "score": score, "id": 0},
            {"name": "triangle", "score": 0.0, "id": 1},
        ],
    }]
    assert json.loads(json.dumps(result, ensure_ascii=False)) == expected
    report = {"matches": result}
    path = tmp_path / "nested" / "report.json"
    assert save_report(report, str(path)) == str(path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"matches": expected}
    # Editing the top-level result must not alter its ranking snapshot.
    result[0]["name"] = "edited"
    assert result[0]["all_scores"][0]["name"] == "pair"


def test_single_record_report():
    lib = ConnectionLibrary()
    lib.add_record("circle", [{"type": "circle", "zone": "*", "color": "red"}])
    result = query_match([{"obj": "circle|red", "pos": "r0"}], lib)
    restored = json.loads(json.dumps(result))
    assert restored[0]["margin"] == 1.0
    assert restored[0]["all_scores"] == [{"name": "circle", "score": 1.0, "id": 0}]


def test_ranking_remains_sorted_and_limited_to_five():
    lib = ConnectionLibrary()
    for i in range(7):
        lib.add_record(str(i), [{"type": "circle", "zone": "*", "color": "blue"}])
    lib.add_record("winner", [{"type": "circle", "zone": "*", "color": "red"}])
    result = json.loads(json.dumps(query_match(
        [{"obj": "circle|red", "pos": "r0"}], lib)))
    assert result[0]["name"] == "winner"
    assert result[0]["margin"] == 1.0
    assert [item["id"] for item in result[0]["all_scores"]] == [7, 0, 1, 2, 3]


def test_no_detections_remains_json_serializable():
    assert json.loads(json.dumps(query_match([], make_library()))) == [
        {"verdict": "BLINDSPOT", "note": "无检出"},
    ]
