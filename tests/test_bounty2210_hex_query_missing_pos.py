import pytest
from lingshu.nn.hex_query import (
    query_match,
    build_default_library,
    ConnectionLibrary,
    extract_relations,
    _conn_matches,
)

def test_conn_matches_missing_pos_key_with_wildcard_zone():
    """dsh-bounty #2210: zone='*' never requires pos, should match without KeyError."""
    conn = {"role": "p", "type": "circle", "zone": "*", "color": "*"}
    det_without_pos = {"obj": "circle|red"}
    assert _conn_matches(conn, det_without_pos) is True

def test_conn_matches_missing_pos_key_with_specific_zone():
    """dsh-bounty #2210: zone!='*' fails closed when pos is missing (does not raise KeyError)."""
    conn = {"role": "p", "type": "circle", "zone": "top", "color": "*"}
    det_without_pos = {"obj": "circle|red"}
    assert _conn_matches(conn, det_without_pos) is False

def test_query_match_missing_pos_end_to_end():
    """dsh-bounty #2210: query_match end-to-end should never raise KeyError on missing pos."""
    lib = ConnectionLibrary()
    lib.add_record("single_wildcard", [{"role": "p", "type": "circle", "zone": "*", "color": "*"}], {})
    
    # Missing pos detection
    res = query_match([{"obj": "circle|red"}], lib)
    assert len(res) == 1
    assert res[0]["name"] == "single_wildcard"
    assert res[0]["verdict"] == "ACCEPT"
    assert res[0]["score"] == 1.0

def test_query_match_default_library_missing_pos():
    """dsh-bounty #2210: default library contains recipes with zones; missing pos should safely evaluate without KeyError."""
    lib = build_default_library()
    # Missing pos in input
    res = query_match([{"obj": "circle|red"}], lib)
    assert len(res) == 1
    assert "verdict" in res[0]
    # Should not crash and return valid verdict (e.g. BLINDSPOT)
    assert res[0]["verdict"] in {"ACCEPT", "DEFER", "REJECT", "BLINDSPOT"}

def test_extract_relations_missing_pos():
    """dsh-bounty #2210: extract_relations skips detections with missing or unparseable pos instead of crashing."""
    dets = [
        {"obj": "circle|red"},  # missing pos
        {"obj": "square|blue", "pos": "r4"},
        {"obj": "triangle|green", "pos": "r1"},
    ]
    rels = extract_relations(dets)
    # Only pair with both valid pos (r4 and r1) will produce a relation
    assert len(rels) == 1
    assert rels[0]["type_a"] == "square"
    assert rels[0]["type_b"] == "triangle"
