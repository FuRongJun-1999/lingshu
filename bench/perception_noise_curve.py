#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""perception_noise_curve · #172 端到端感知噪声退化曲线评测闭环（纯标准库）。

承接 PR #1406（tests/test_e2e_pixel_to_memory.py）：
    SceneSimulator 场景帧真值 → Camera3D 投影 2D 框 → 三轴可控感知噪声注入
    （错类率 misclass_rate / 漏检率 drop_rate / 框与位置抖动 bbox_jitter, pos_jitter）
    → World3D.add_vprim 反投影 → SpacetimeMemoryEngine 落盘与语义检索召回
    → 端到端命中率（检测 × 语义召回 × 3D 定位容差）退化曲线。

设计约束：
    1. 仅用 Python 标准库，不依赖 numpy / Pillow / pytest；
    2. 固定随机种子下逐位确定性可复算（使用独立 local Random，不污染全局 RNG）；
    3. 非法配置（bool / NaN / ±inf / 越界值 / 非数值）一律 fail-closed 抛 ValueError；
    4. 零噪声配置（rates=0, jitters=0）逐位等价于 clean oracle（100% e2e_hit_rate）。

用法（仓根）：
    python -X utf8 bench/perception_noise_curve.py
    python -X utf8 bench/perception_noise_curve.py --json
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import pathlib
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lingshu.core.core import ConditionSpace, STNode, SpacetimeMemoryEngine
from lingshu.world.scene_simulator import SceneSimulator
from lingshu.world.vprim import VPrim
from lingshu.world.world3d import Camera3D, DEFAULT_VISUAL_SPECS, World3D

SCREEN_W = 800
SCREEN_H = 600
DEFAULT_SPATIAL_TOL = 1.0  # 米（仅用于区分「正常重建 vs 明显漂移」，可由参数覆写）

# 选取视觉尺寸与词典先验均互异的五类地面实体，确保错类同时导致语义错配与尺度深度偏移
CATEGORY_POOL: Tuple[str, ...] = ("person", "car", "tree", "dog", "house")

# 5 条受控路径：全部保持在 Camera3D(cx=40, cy=2, cz=5, yaw=0, pitch=0, fov=60) 视锥内
_ENTITY_SPECS: Tuple[Tuple[str, str, Tuple[float, float, float], Tuple[float, float, float], float], ...] = (
    ("e_person", "person", (36.0, 0.0, 14.0), (38.0, 0.0, 16.0), 0.4),
    ("e_car", "car", (44.0, 0.0, 18.0), (42.0, 0.0, 20.0), 0.4),
    ("e_tree", "tree", (33.0, 0.0, 22.0), (34.0, 0.0, 23.0), 0.2),
    ("e_dog", "dog", (41.0, 0.0, 12.0), (40.0, 0.0, 13.5), 0.3),
    ("e_house", "house", (47.0, 0.0, 26.0), (46.0, 0.0, 27.0), 0.2),
)


def _validate_finite_float(name: str, value: float) -> float:
    """校验有限实数；拒绝 bool / NaN / ±inf / 非数值。"""
    if isinstance(value, bool):
        raise ValueError(f"{name} 不得为 bool，收到 {value!r}")
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是实数，收到 {value!r}") from exc
    if not math.isfinite(v):
        raise ValueError(f"{name} 必须是有限实数，收到 {value!r}")
    return v


def _validate_unit_rate(name: str, value: float) -> float:
    """校验 [0.0, 1.0] 概率参数。"""
    v = _validate_finite_float(name, value)
    if v < 0.0 or v > 1.0:
        raise ValueError(f"{name} 必须在 [0.0, 1.0] 范围内，收到 {v!r}")
    return v


def _validate_nonneg_finite(name: str, value: float) -> float:
    """校验非负有限参数（>= 0.0）。"""
    v = _validate_finite_float(name, value)
    if v < 0.0:
        raise ValueError(f"{name} 不得为负数，收到 {v!r}")
    return v


def _validate_pos_finite(name: str, value: float) -> float:
    """校验正有限参数（> 0.0）；拒绝 bool / NaN / ±inf / <= 0。"""
    v = _validate_nonneg_finite(name, value)
    if v <= 0.0:
        raise ValueError(f"{name} 必须大于 0，收到 {v!r}")
    return v


def _validate_int(name: str, value: int, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} 必须是整数（非 bool），收到 {value!r}")
    if value < minimum:
        raise ValueError(f"{name} 不得小于 {minimum}，收到 {value!r}")
    return value


@dataclass(frozen=True)
class NoiseConfig:
    """三轴感知噪声配置（校验失败即 fail-closed 抛 ValueError）。"""

    misclass_rate: float = 0.0
    drop_rate: float = 0.0
    bbox_jitter: float = 0.0
    pos_jitter: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "misclass_rate", _validate_unit_rate("misclass_rate", self.misclass_rate))
        object.__setattr__(self, "drop_rate", _validate_unit_rate("drop_rate", self.drop_rate))
        object.__setattr__(self, "bbox_jitter", _validate_nonneg_finite("bbox_jitter", self.bbox_jitter))
        object.__setattr__(self, "pos_jitter", _validate_nonneg_finite("pos_jitter", self.pos_jitter))

    @property
    def is_clean(self) -> bool:
        return (
            self.misclass_rate == 0.0
            and self.drop_rate == 0.0
            and self.bbox_jitter == 0.0
            and self.pos_jitter == 0.0
        )


def make_ground_truth_frames(seed: int = 0, num_ticks: int = 6) -> Tuple[Camera3D, List[Dict]]:
    """使用 SceneSimulator + Camera3D 确定性生成每帧真值 (eid, category, bbox2d, world_pos)。"""
    seed = _validate_int("seed", seed, minimum=0)
    num_ticks = _validate_int("num_ticks", num_ticks, minimum=1)

    cam = Camera3D(cx=40.0, cy=2.0, cz=5.0, yaw=0.0, pitch=0.0, fov_deg=60.0)
    world_ref = World3D(camera=cam)
    sim = SceneSimulator(size=80, seed=seed)
    focal = cam.focal(SCREEN_W)

    label_to_sim_eid: Dict[str, str] = {}
    for eid, cat, p0, p1, spd in _ENTITY_SPECS:
        sim.add_path(f"path_{eid}", [p0, p1])
        sim_eid = sim.add_entity(cat, behavior="follow", pos=p0, speed=spd, goal=f"path_{eid}")
        label_to_sim_eid[eid] = sim_eid

    frames: List[Dict] = []
    for tick in range(num_ticks):
        if tick > 0:
            sim.step()
        for eid, cat, _, _, _ in _ENTITY_SPECS:
            ent = sim.entities[label_to_sim_eid[eid]]
            spec = DEFAULT_VISUAL_SPECS[cat]
            wx, _, wz = ent.pos
            wy = world_ref._ground_y(cat, spec.size)
            proj = cam.project((wx, wy, wz), SCREEN_W, SCREEN_H)
            depth = cam.depth_of((wx, wy, wz))
            assert proj is not None and depth > 0.1, f"真值实体 {eid} 越出视锥：pos={(wx, wy, wz)}"
            px, py = proj
            w_px = focal * spec.size[0] / depth
            h_px = focal * spec.size[1] / depth
            bbox = (
                px - w_px / 2.0,
                py - h_px / 2.0,
                px + w_px / 2.0,
                py + h_px / 2.0,
            )
            frames.append({
                "tick": tick,
                "eid": eid,
                "category": cat,
                "bbox": bbox,
                "world_pos": (float(wx), float(wy), float(wz)),
            })
    return cam, frames


def inject_perception_noise(
    category: str,
    bbox: Tuple[float, float, float, float],
    cfg: NoiseConfig,
    rng: random.Random,
    screen_w: int = SCREEN_W,
    screen_h: int = SCREEN_H,
) -> Optional[Tuple[str, Tuple[float, float, float, float]]]:
    """对单帧真值观测注入三轴可控噪声；返回 None 表示漏检。"""
    if cfg.is_clean:
        return category, bbox

    if cfg.drop_rate > 0.0 and rng.random() < cfg.drop_rate:
        return None

    out_cat = category
    if cfg.misclass_rate > 0.0 and rng.random() < cfg.misclass_rate:
        others = [c for c in CATEGORY_POOL if c != category]
        out_cat = rng.choice(others)

    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w_px = max(2.0, x2 - x1)
    h_px = max(2.0, y2 - y1)

    if cfg.pos_jitter > 0.0:
        cx += rng.uniform(-cfg.pos_jitter, cfg.pos_jitter)
        cy += rng.uniform(-cfg.pos_jitter, cfg.pos_jitter)
    if cfg.bbox_jitter > 0.0:
        w_px = max(2.0, w_px + rng.uniform(-cfg.bbox_jitter, cfg.bbox_jitter))
        h_px = max(2.0, h_px + rng.uniform(-cfg.bbox_jitter, cfg.bbox_jitter))

    nx1 = max(0.0, min(float(screen_w - 2), cx - w_px / 2.0))
    ny1 = max(0.0, min(float(screen_h - 2), cy - h_px / 2.0))
    nx2 = max(nx1 + 2.0, min(float(screen_w), cx + w_px / 2.0))
    ny2 = max(ny1 + 2.0, min(float(screen_h), cy + h_px / 2.0))
    return out_cat, (nx1, ny1, nx2, ny2)


def run_episode(
    cfg: NoiseConfig,
    seed: int = 0,
    num_ticks: int = 6,
    spatial_tol: float = DEFAULT_SPATIAL_TOL,
) -> Dict:
    """运行单轮端到端评测：真值生成 → 噪声感知 → World3D → Engine 写入与召回计分。"""
    if not isinstance(cfg, NoiseConfig):
        raise ValueError(f"cfg 必须是 NoiseConfig，收到 {type(cfg).__name__}")
    seed = _validate_int("seed", seed, minimum=0)
    num_ticks = _validate_int("num_ticks", num_ticks, minimum=1)
    spatial_tol = _validate_pos_finite("spatial_tol", spatial_tol)

    cam, frames = make_ground_truth_frames(seed=seed, num_ticks=num_ticks)
    rng = random.Random(seed * 1009 + 17)

    world = World3D(camera=cam)
    engine = SpacetimeMemoryEngine(":memory:")
    cs = ConditionSpace(
        observation_position="world3d",
        observation_tool="perception_noise_curve",
        time_window=(0.0, 9999999999.0),
        existence_constraint="issue172_noise_eval",
    )

    for idx, fr in enumerate(frames):
        obs = inject_perception_noise(fr["category"], fr["bbox"], cfg, rng)
        if obs is None:
            continue
        p_cat, p_bbox = obs
        obj = world.add_vprim(
            VPrim(
                category=p_cat,
                bbox=p_bbox,
                confidence=1.0 if cfg.is_clean else 0.7,
                eid=fr["eid"],
            ),
            SCREEN_W,
            SCREEN_H,
            tick=fr["tick"],
        )
        assert obj is not None
        nid = f"obs_{fr['tick']}_{fr['eid']}"
        node = STNode(
            id=nid,
            content=(
                f"tick={fr['tick']} eid={fr['eid']} cat={p_cat} "
                f"x={obj.center[0]:.2f} z={obj.center[2]:.2f}"
            ),
            modality="text",
            spatial_coordinates={"x": obj.center[0], "y": obj.center[1], "z": obj.center[2]},
            temporal_coordinate=float(fr["tick"]),
            condition_space=cs,
        )
        engine.store.add_node(node)

    # 预先按真值类别通过 engine.store.search_content 召回一次，检验写入后可检索命中
    recalled_by_cat: Dict[str, set] = {}
    for cat in CATEGORY_POOL:
        hits = engine.store.search_content(cat, limit=-1, touch=False)
        recalled_by_cat[cat] = {n.id for n, _ in hits if f"cat={cat} " in (n.content or "")}

    total = len(frames)
    detected = 0
    semantic_hits = 0
    spatial_hits = 0
    e2e_hits = 0
    xz_errors: List[float] = []

    for fr in frames:
        nid = f"obs_{fr['tick']}_{fr['eid']}"
        node = engine.store.get_node(nid)
        if node is None:
            continue
        detected += 1
        sem_ok = nid in recalled_by_cat.get(fr["category"], set())
        if sem_ok:
            semantic_hits += 1
        sc = node.spatial_coordinates or {}
        err = math.hypot(
            float(sc.get("x", 0.0)) - fr["world_pos"][0],
            float(sc.get("z", 0.0)) - fr["world_pos"][2],
        )
        xz_errors.append(err)
        sp_ok = err <= spatial_tol
        if sp_ok:
            spatial_hits += 1
        if sem_ok and sp_ok:
            e2e_hits += 1

    return {
        "seed": seed,
        "num_ticks": num_ticks,
        "total_frames": total,
        "detected": detected,
        "detection_rate": round(detected / total, 6),
        "semantic_hits": semantic_hits,
        "semantic_accuracy": round(semantic_hits / total, 6),
        "spatial_hits": spatial_hits,
        "spatial_hit_rate": round(spatial_hits / total, 6),
        "e2e_hits": e2e_hits,
        "e2e_hit_rate": round(e2e_hits / total, 6),
        "mean_xz_error": round(sum(xz_errors) / len(xz_errors), 6) if xz_errors else None,
    }


def evaluate_noise_point(
    cfg: NoiseConfig,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    num_ticks: int = 6,
    spatial_tol: float = DEFAULT_SPATIAL_TOL,
) -> Dict:
    """跨多个随机种子聚合单一噪声配置点的均值读数。"""
    seed_list = [_validate_int("seed", s, minimum=0) for s in seeds]
    if not seed_list:
        raise ValueError("seeds 不能为空")
    episodes = [run_episode(cfg, seed=s, num_ticks=num_ticks, spatial_tol=spatial_tol) for s in seed_list]
    n = len(episodes)
    errs = [ep["mean_xz_error"] for ep in episodes if ep["mean_xz_error"] is not None]
    return {
        "config": {
            "misclass_rate": cfg.misclass_rate,
            "drop_rate": cfg.drop_rate,
            "bbox_jitter": cfg.bbox_jitter,
            "pos_jitter": cfg.pos_jitter,
        },
        "seeds": seed_list,
        "num_ticks": num_ticks,
        "spatial_tol": spatial_tol,
        "detection_rate": round(sum(ep["detection_rate"] for ep in episodes) / n, 6),
        "semantic_accuracy": round(sum(ep["semantic_accuracy"] for ep in episodes) / n, 6),
        "spatial_hit_rate": round(sum(ep["spatial_hit_rate"] for ep in episodes) / n, 6),
        "e2e_hit_rate": round(sum(ep["e2e_hit_rate"] for ep in episodes) / n, 6),
        "mean_xz_error": round(sum(errs) / len(errs), 6) if errs else None,
    }


def run_default_curves(
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    num_ticks: int = 6,
    spatial_tol: float = DEFAULT_SPATIAL_TOL,
) -> Dict:
    """产出单轴扫描曲线与组合噪声网格的完整评测报告。"""
    sweeps = {
        "misclass_rate": [
            evaluate_noise_point(NoiseConfig(misclass_rate=r), seeds, num_ticks, spatial_tol)
            for r in (0.0, 0.25, 0.50, 0.75, 1.0)
        ],
        "drop_rate": [
            evaluate_noise_point(NoiseConfig(drop_rate=r), seeds, num_ticks, spatial_tol)
            for r in (0.0, 0.25, 0.50, 0.75, 1.0)
        ],
        "bbox_jitter": [
            evaluate_noise_point(NoiseConfig(bbox_jitter=j), seeds, num_ticks, spatial_tol)
            for j in (0.0, 4.0, 8.0, 16.0, 32.0)
        ],
        "pos_jitter": [
            evaluate_noise_point(NoiseConfig(pos_jitter=j), seeds, num_ticks, spatial_tol)
            for j in (0.0, 25.0, 50.0, 100.0, 150.0)
        ],
    }
    combined_specs = (
        ("oracle_clean", NoiseConfig(0.0, 0.0, 0.0, 0.0)),
        ("mild_noise", NoiseConfig(0.10, 0.10, 4.0, 15.0)),
        ("moderate_noise", NoiseConfig(0.25, 0.25, 8.0, 30.0)),
        ("severe_noise", NoiseConfig(0.50, 0.50, 16.0, 60.0)),
        ("full_dropout", NoiseConfig(0.0, 1.0, 0.0, 0.0)),
    )
    combined = [
        {"label": label, **evaluate_noise_point(cfg, seeds, num_ticks, spatial_tol)}
        for label, cfg in combined_specs
    ]
    return {
        "seeds": list(seeds),
        "num_ticks": num_ticks,
        "spatial_tol": spatial_tol,
        "sweeps": sweeps,
        "combined": combined,
    }


def _format_markdown(report: Dict) -> str:
    lines = [
        "# #172 端到端感知噪声退化曲线（seeds=%s, ticks=%d, spatial_tol=%.2fm）"
        % (report["seeds"], report["num_ticks"], report["spatial_tol"]),
        "",
    ]
    for axis, rows in report["sweeps"].items():
        lines.append(f"## 单轴扫描：{axis}")
        lines.append("| 值 | 检测率 | 语义准确率 | 3D 定位命中率 | 端到端命中率 | 平均 XZ 误差(m) |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for row in rows:
            val = row["config"][axis]
            err = f"{row['mean_xz_error']:.3f}" if row["mean_xz_error"] is not None else "N/A"
            lines.append(
                f"| {val} | {row['detection_rate']*100:.1f}% | {row['semantic_accuracy']*100:.1f}% "
                f"| {row['spatial_hit_rate']*100:.1f}% | {row['e2e_hit_rate']*100:.1f}% | {err} |"
            )
        lines.append("")

    lines.append("## 组合噪声网格")
    lines.append("| 档位 | (misclass, drop, bbox_px, pos_px) | 检测率 | 语义准确率 | 3D 定位命中率 | 端到端命中率 | 平均 XZ 误差(m) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in report["combined"]:
        c = row["config"]
        err = f"{row['mean_xz_error']:.3f}" if row["mean_xz_error"] is not None else "N/A"
        lines.append(
            f"| {row['label']} | ({c['misclass_rate']}, {c['drop_rate']}, {c['bbox_jitter']}, {c['pos_jitter']}) "
            f"| {row['detection_rate']*100:.1f}% | {row['semantic_accuracy']*100:.1f}% "
            f"| {row['spatial_hit_rate']*100:.1f}% | {row['e2e_hit_rate']*100:.1f}% | {err} |"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="#172 端到端感知噪声退化曲线评测")
    parser.add_argument("--seeds", type=int, default=5, help="随机种子数（从 0 起）")
    parser.add_argument("--ticks", type=int, default=6, help="每轮模拟帧数")
    parser.add_argument("--spatial-tol", type=float, default=DEFAULT_SPATIAL_TOL, help="3D 定位容差（米）")
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出")
    args = parser.parse_args(argv)

    seeds = list(range(_validate_int("seeds", args.seeds, minimum=1)))
    report = run_default_curves(seeds=seeds, num_ticks=args.ticks, spatial_tol=args.spatial_tol)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(_format_markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
