"""Persistence-path regressions for #1065, already fixed by PR #1070."""

import json

import pytest

from lingshu.world.channel_credibility import ChannelCredibilityRegistry


@pytest.mark.parametrize(
    ("relative_path", "existing_parent"),
    [
        pytest.param("cred.json", True, id="bare-filename"),
        pytest.param("new/nested/cred.json", False, id="missing-nested-parents"),
        pytest.param("existing/cred.json", True, id="existing-parent"),
    ],
)
def test_persistence_paths_write_json_and_reload(
    relative_path, existing_parent, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / relative_path
    if existing_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        assert not path.parent.exists()
    assert not path.exists()

    # Pass the relative spelling unchanged: "cred.json" has no dirname.
    reg = ChannelCredibilityRegistry(persist_path=relative_path)
    reg.record_hit("visual", conf=1.0)
    reg.record_miss("visual", conf=0.5)

    assert path.is_file(), "Public updates must persist even without existing parents"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["visual"] == pytest.approx(
        {"a": 11.5, "b": 7.0, "hits": 1.0, "misses": 1.0}
    )

    reloaded = ChannelCredibilityRegistry(persist_path=relative_path)
    assert reloaded.registry() == reg.registry()
    assert reloaded.credibility("visual") == pytest.approx(11.5 / 18.5)
