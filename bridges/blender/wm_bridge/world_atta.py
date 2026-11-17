# -*- coding: utf-8 -*-
"""Read-only World-Atta oracle replay provider.

World-Atta is a synthetic, static fixture rather than an online camera or a
prediction source.  This adapter keeps it on the external bridge boundary and
exposes the existing ``wm_bridge.oracle/1`` records as an ``actual`` provider
without modifying ``lingshu.world`` or any memory store.

The provider intentionally supports only the dataset's declared static
snapshot (tick 0).  A non-zero tick is an explicit error, never an empty frame
that could be mistaken for a valid observation.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

from . import oracle as O


class WorldAttaProviderError(ValueError):
    """World-Atta fixture is absent, malformed, or requested out of contract."""


def _scene_path(root: os.PathLike[str] | str, scene_id: str | None = None) -> Path:
    base = Path(root).expanduser().resolve()
    if scene_id is not None:
        if not isinstance(scene_id, str) or not scene_id.strip():
            raise WorldAttaProviderError("scene_id 必须是非空字符串")
        if Path(scene_id).name != scene_id or scene_id in {".", ".."}:
            raise WorldAttaProviderError("scene_id 禁止路径穿越：%r" % scene_id)
        base = (base / scene_id).resolve()
        try:
            base.relative_to(Path(root).expanduser().resolve())
        except ValueError as exc:
            raise WorldAttaProviderError("scene_id 越出 dataset root") from exc
    if not base.is_dir():
        raise WorldAttaProviderError("World-Atta scene 目录不存在：%s" % base)
    return base


def _read(scene_dir: Path) -> Dict[str, Any]:
    path = scene_dir / "oracle.jsonl"
    if not path.is_file():
        raise WorldAttaProviderError("缺 oracle.jsonl：%s" % path)
    try:
        doc = O.read_oracle(str(path))
    except (O.OracleContractError, OSError, ValueError) as exc:
        raise WorldAttaProviderError("oracle 契约不通过：%s" % exc) from exc
    scene = doc.get("scene") or {}
    if scene.get("motion") not in (None, "static_snapshot"):
        raise WorldAttaProviderError(
            "只读 replay 仅接受 static_snapshot，收到 %r" % scene.get("motion"))
    if scene.get("export_frame") != 0:
        raise WorldAttaProviderError(
            "World-Atta 静态 provider 要求 export_frame=0，收到 %r"
            % scene.get("export_frame"))
    if any(record.get("tick") != 0 for record in doc.get("tracks", [])):
        raise WorldAttaProviderError("静态 World-Atta oracle 含非零 tick")
    return doc


def load_scene(scene_dir: os.PathLike[str] | str, scene_id: str | None = None) -> Dict[str, Any]:
    """Load and validate one scene, returning an immutable-by-convention copy."""
    path = _scene_path(scene_dir, scene_id)
    doc = _read(path)
    entities = sorted({str(row["entity"]) for row in doc["tracks"]})
    return {
        "schema": doc["schema"],
        "scene": copy.deepcopy(doc["scene"]),
        "tracks": copy.deepcopy(doc["tracks"]),
        "scene_dir": str(path),
        "oracle_path": str(path / "oracle.jsonl"),
        "entities": entities,
        "provenance": {
            "source": "world-atta",
            "kind": "synthetic_ground_truth",
            "synthetic": True,
            "motion": "static_snapshot",
            "export_frame": 0,
        },
    }


def metadata(scene: Dict[str, Any]) -> Dict[str, Any]:
    """Return provenance metadata without exposing the provider's mutable state."""
    if not isinstance(scene, dict) or "scene" not in scene:
        raise WorldAttaProviderError("scene 必须来自 load_scene()")
    out = copy.deepcopy(scene.get("provenance") or {})
    out.update({
        "scene_id": (scene.get("scene") or {}).get("name"),
        "oracle_path": scene.get("oracle_path"),
        "view_count": 0,
        "tick": 0,
    })
    return out


def iter_actual(scene: Dict[str, Any], tick: int = 0) -> Iterator[Dict[str, Any]]:
    """Yield canonical oracle records for the one supported static tick."""
    if isinstance(tick, bool) or not isinstance(tick, int):
        raise WorldAttaProviderError("tick 必须是整数：%r" % tick)
    if tick != 0:
        raise WorldAttaProviderError(
            "static_snapshot 只提供 tick=0，不把缺帧伪装成空 actual")
    if not isinstance(scene, dict) or not isinstance(scene.get("tracks"), list):
        raise WorldAttaProviderError("scene 必须来自 load_scene()")
    for row in sorted(scene["tracks"], key=lambda item: str(item["entity"])):
        yield copy.deepcopy(row)


def actual_at_tick(scene: Dict[str, Any], tick: int = 0) -> List[Dict[str, Any]]:
    """Materialize one deterministic actual frame for a verify shim."""
    return list(iter_actual(scene, tick=tick))


def as_observations(scene: Dict[str, Any], tick: int = 0) -> List[Dict[str, Any]]:
    """Map actual records to an explicit external-observation namespace.

    The original oracle row is retained under ``oracle``.  This makes it hard
    for a caller to mistake synthetic actual data for a prediction or to lose
    the source entity/link distinction.
    """
    out = []
    for row in iter_actual(scene, tick=tick):
        out.append({
            "eid": row["entity"],
            "category": row.get("category"),
            "pos": [row["pos"][axis] for axis in ("x", "y", "z")],
            "tick": row["tick"],
            "source": "world-atta",
            "kind": "synthetic_ground_truth",
            "provenance": {
                "scene": (scene.get("scene") or {}).get("name"),
                "oracle_path": scene.get("oracle_path"),
                "synthetic": True,
            },
            "oracle": row,
        })
    return out
