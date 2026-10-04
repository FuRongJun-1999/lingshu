# Copyright 2026 灵枢 (Lingshu) · MIT
"""WorldModel — 语义时空图 ↔ 3D 场景绑定层（极简世界模型）。

理念（时空记忆图 = 智能体的世界模型，智能论 4.x）：
- **语义层**：实体/节点，有类别（person/fatfish/table/tree…）、关系边
  （left_of / right_of / near / far…）、状态（happy/tired/neutral…）。
- **空间层**：每个实体有一个 3D 锚点（世界坐标 x,y,z 米）+ 朝向。
- **呈现层**：类别 → 视觉先验（轮廓角色 / 骨架 / 盒体）+ 深度着色。
- **绑定 = 双向映射**：语义 → 空间（关系边驱动相对布局）+ 状态 → 呈现
  （颜色/姿态/亮度变化）。这就是世界模型的极简实现：语义理解的实体
  在同一个可观测、可旋转的 3D 场景里共存。

数据流：
    scene = WorldModel()
    scene.add_entity("肥鱼", category="fatfish", state="happy", pos=(0,0,5))
    scene.add_entity("桌子", category="table", pos=(1.5,0,6))
    scene.relation("桌子", "right_of", "肥鱼")     # 语义关系 → 空间约束
    img = scene.render(camera=...)                  # 3D 场景图
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .world3d import Camera3D, Object3D, World3D
from .silhouette3d import (  # noqa: E402
    Silhouette3D,
    SilhouettePart,
    fatfish_skeleton,
    fatfish_skinned,
    apply_pose,
    set_expression,
)
from .skeleton3d import Skeleton3D, human_skeleton, fatfish_bone_skeleton

# ---------------------------------------------------------------------------
# 实体（语义 + 空间 + 呈现）
# ---------------------------------------------------------------------------


@dataclass
class SceneEntity:
    """场景实体：语义类别 + 3D 锚点 + 状态 + 呈现类型。"""
    name: str
    category: str                      # person / fatfish / cat / table / tree…
    pos: Tuple[float, float, float]    # 3D 锚点（世界坐标，米）
    state: str = "neutral"             # 状态：happy/tired/angry/neutral…
    present: str = "auto"              # 呈现：silhouette/skeleton/box（auto=按类别）
    rot: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    relations: List[Tuple[str, str]] = field(default_factory=list)  # (target, kind)


# 类别 → 呈现类型
PRESENT_BY_CATEGORY = {
    "person": "skeleton",
    "fatfish": "silhouette",      # 主角：漫画轮廓
    "cat": "box",
    "girl": "skeleton",
    "table": "box",
    "tree": "box",
    "house": "box",
    "chair": "box",
    "dog": "box",
    "dragon": "box",
}


class WorldModel:
    """语义时空图 → 3D 场景。"""

    def __init__(self):
        self.entities: Dict[str, SceneEntity] = {}
        self.scene = World3D()
        self._silhouettes: Dict[str, Silhouette3D] = {}

    # ---- 语义层操作 ----

    def add_entity(self, name: str, category: str,
                   pos: Tuple[float, float, float],
                   state: str = "neutral",
                   present: str = "auto") -> "WorldModel":
        """添加语义实体（3D 锚点 + 类别 + 状态）。"""
        if present == "auto":
            present = PRESENT_BY_CATEGORY.get(category, "box")
        self.entities[name] = SceneEntity(
            name=name, category=category, pos=pos, state=state, present=present)
        return self

    def relation(self, a: str, kind: str, b: str) -> "WorldModel":
        """语义关系边：a kind b（kind ∈ left_of/right_of/near/far/in_front/behind）。"""
        if a not in self.entities or b not in self.entities:
            raise ValueError(f"relation requires both entities: {a}, {b}")
        self.entities[a].relations.append((b, kind))
        return self

    def set_state(self, name: str, state: str) -> "WorldModel":
        """状态更新（→ 呈现变化：颜色/姿态）。"""
        if name in self.entities:
            self.entities[name].state = state
        return self

    # ---- 语义关系 → 空间约束 ----

    def apply_relations(self) -> None:
        """把关系边解释为空间约束（极简：相对位置调整 + 朝向）。"""
        # 多次遍历直到稳定（简单约束求解）
        for _ in range(3):
            moved = False
            for name, e in self.entities.items():
                for target, kind in e.relations:
                    if target not in self.entities:
                        continue
                    t = self.entities[target]
                    dx = e.pos[0] - t.pos[0]
                    dz = e.pos[2] - t.pos[2]
                    dist = math.hypot(dx, dz)
                    if kind == "left_of" and dx <= 0:
                        e.pos = (t.pos[0] - 1.2, e.pos[1], t.pos[2]); moved = True
                    elif kind == "right_of" and dx >= 0:
                        e.pos = (t.pos[0] + 1.2, e.pos[1], t.pos[2]); moved = True
                    elif kind == "near" and (dist > 1.0 or dist < 0.3):
                        e.pos = (t.pos[0] + 0.8, e.pos[1], t.pos[2] + 0.3); moved = True
                    elif kind == "far" and dist < 3.0:
                        e.pos = (t.pos[0] + 3.5, e.pos[1], t.pos[2] + 1.0); moved = True
                    elif kind == "in_front" and dz >= 0:
                        e.pos = (e.pos[0], e.pos[1], t.pos[2] - 1.5); moved = True
                    elif kind == "behind" and dz <= 0:
                        e.pos = (e.pos[0], e.pos[1], t.pos[2] + 1.5); moved = True
            if not moved:
                break

    # ---- 状态 → 呈现变化 ----

    # 状态 → 轮廓角色颜色微调（肥鱼：害羞/开心变色）
    STATE_TINT = {
        "happy":   (1.05, 0.98, 0.98),   # 微亮
        "sad":     (0.90, 0.92, 1.00),   # 偏冷
        "angry":   (1.08, 0.92, 0.92),   # 偏红
        "tired":   (0.95, 0.95, 1.02),
        "shy":     (1.10, 0.94, 0.96),   # 偏粉
    }

    @staticmethod
    def _tint(color: Tuple[int, int, int], t: Tuple[float, float, float]) -> Tuple[int, int, int]:
        return tuple(max(0, min(255, int(c * f))) for c, f in zip(color, t))

    def _apply_state_to_silhouette(self, name: str, sil: Silhouette3D) -> None:
        tint = self.STATE_TINT.get(self.entities[name].state, (1.0, 1.0, 1.0))
        for p in sil.parts:
            p.color = self._tint(p.color, tint)
        # 姿态：状态 → 四肢/尾巴关节角度（肥鱼会动）
        apply_pose(sil, self.entities[name].state)
        # 表情：状态 → 眼型/嘴型/腮红（表情系统）
        set_expression(sil, self.entities[name].state)

    # ---- 构建 3D 场景 ----

    def build(self) -> "WorldModel":
        """语义实体 → 3D 场景（盒体进 World3D，轮廓/骨架单独保留）。"""
        self.apply_relations()
        self.scene = World3D()
        self._silhouettes = {}
        for name, e in self.entities.items():
            if e.present == "silhouette" and e.category == "fatfish":
                # 蒙皮版：身体轮廓从骨架关节推导（骨架=基准，轮廓=肉，永不脱节）
                sil = fatfish_skinned(center=e.pos, height=1.55)
                self._apply_state_to_silhouette(name, sil)
                self._silhouettes[name] = sil
            elif e.present == "skeleton":
                sk = human_skeleton(center=e.pos, height=1.7)
                # 状态 → 姿态（示意：happy 挥手，tired 低头）
                if e.state == "happy":
                    sk.joints["shoulder_r"].rot = (0, 0, 1.2)
                    sk.joints["elbow_r"].rot = (0, 0, -1.0)
                elif e.state == "tired":
                    sk.joints["neck"].rot = (0.5, 0, 0)
                self.scene.objects.append(self._skeleton_to_object(sk, name, e))
            else:
                # 盒体呈现
                size = {"table": (1.2, 0.9, 0.8), "tree": (1.0, 2.0, 1.0),
                        "house": (3.0, 2.5, 3.0), "cat": (0.5, 0.4, 0.35),
                        "dog": (0.8, 0.6, 0.5), "chair": (0.5, 0.9, 0.5),
                        "dragon": (4.0, 2.0, 1.5)}.get(e.category, (0.6, 0.6, 0.6))
                color = {"table": (150, 120, 90), "tree": (60, 140, 80),
                         "house": (200, 170, 120), "cat": (200, 150, 100),
                         "dog": (150, 120, 90), "chair": (140, 110, 80),
                         "dragon": (90, 160, 60)}.get(e.category, (160, 160, 160))
                self.scene.objects.append(Object3D(
                    category=e.category, center=e.pos, size=size,
                    color=color, shape="box"))
        return self

    def _skeleton_to_object(self, sk: Skeleton3D, name: str, e: SceneEntity) -> Object3D:
        """骨架 → Object3D（供 World3D 画家算法参与排序；实际渲染走骨架通道）。"""
        return Object3D(category=f"skeleton_{e.category}", center=e.pos,
                        size=(0.6, 1.7, 0.4), color=(200, 160, 140), shape="pillar")

    # ---- 渲染 ----

    def render(self, screen_w: int = 500, screen_h: int = 500,
               camera: Optional[Camera3D] = None,
               background: Tuple[int, int, int] = (250, 248, 244),
               bones_overlay: bool = False) -> "PIL.Image":
        """渲染完整场景：盒体(World3D) + 轮廓角色(深度着色) + 可选骨架线条叠加。

        bones_overlay=True 时：fatfish 轮廓上叠加 47 关节精细骨架线条
        （"骨架 + 面"完整体现——面=轮廓色块，骨架=内部线条）。
        """
        from PIL import Image

        cam = camera or Camera3D.look_at(
            eye=(0, 1.2, 1.5), target=(0, 0.9, 5.0), fov_deg=50)
        # 1. 盒体场景（含地面）
        img = self.scene.render(screen_w, screen_h, camera=cam,
                                background=background,
                                ground_color=(235, 230, 220))

        # 2. 轮廓角色（C 级合成：RGBA 层 paste，替代逐像素循环）
        for name, sil in self._silhouettes.items():
            layer = sil.render(screen_w, screen_h, camera=cam,
                               background=background, alpha=True)
            img.paste(layer, (0, 0), layer)   # 以自身 alpha 为 mask

        # 3. 骨架线条叠加（fatfish → 精细 47 关节骨架）
        if bones_overlay:
            from PIL import ImageDraw
            from .skeleton3d import fatfish_bone_skeleton
            for name, sil in self._silhouettes.items():
                e = self.entities.get(name)
                if e is None or e.category != "fatfish":
                    continue
                sk = fatfish_bone_skeleton(center=e.pos, height=1.55)
                # 骨架以深色线条叠加（半透明感：深蓝灰）
                self._draw_bones(draw := ImageDraw.Draw(img), sk, cam, screen_w, screen_h)
        return img

    def _draw_bones(self, draw, sk: Skeleton3D, cam: Camera3D,
                    screen_w: int, screen_h: int) -> None:
        """把骨架骨骼线段画到 draw（线条色深蓝灰，醒目但低于轮廓）。"""
        world = {n: sk.world_pos(n) for n in sk.joints}
        screen = {}
        for n, p in world.items():
            sp = cam.project(p, screen_w, screen_h)
            if sp is not None:
                screen[n] = sp
        for parent, child in sk.bones():
            if parent in screen and child in screen:
                draw.line([screen[parent], screen[child]],
                          fill=(40, 60, 90), width=1)

    def scene_text(self) -> str:
        """场景摘要（语义层视角）：实体 + 关系 + 状态 + 位置。"""
        lines = []
        for name, e in self.entities.items():
            rels = ", ".join(f"{kind} {t}" for t, kind in e.relations) or "孤立"
            lines.append(f"{name}[{e.category}/{e.state}]@{e.pos} ({rels})")
        return "；".join(lines)


# ---------------------------------------------------------------------------
# 记忆库绑定（记忆 = 世界持久层，渲染 = 记忆实时投影）
# ---------------------------------------------------------------------------


def ingest_scene(agent, scene_desc: str, store=None, tag: str = "spatial"):
    """把语义场景描述摄入记忆库（带空间锚点 → 可被世界模型重建）。

    约定（记忆即缓存）：
    - 每个实体一条记忆节点，tags 含 `spatial`（场景成员）+ `cat:<类别>` + `ent:<名称>`
    - spatial_coordinates = {"x":.., "y":.., "z":..}（3D 锚点，米）
    - state_attributes = {"state": "happy"}（状态 → 呈现变化）
    返回写入的节点 id 列表。
    """
    if store is None:
        store = getattr(agent, "store", None)
    if store is None:
        engine = getattr(agent, "engine", None)
        if engine is not None:
            store = getattr(engine, "store", None)
    if store is None:
        raise ValueError("need store to ingest scene")
    engine = getattr(agent, "engine", None)

    lines = [ln.strip() for ln in scene_desc.splitlines() if ln.strip()]
    written = []
    for ln in lines:
        # 格式: 名称|类别|x,y,z|状态
        parts = [p.strip() for p in ln.split("|")]
        if len(parts) < 3:
            continue
        name, cat = parts[0], parts[1]
        try:
            x, y, z = (float(v) for v in parts[2].split(","))
        except ValueError:
            continue
        state = parts[3] if len(parts) > 3 else "neutral"
        content = f"场景实体 {name}（{cat}）位于 ({x},{y},{z})，状态 {state}"
        # 优先走 engine.add_perception（支持 spatial_coordinates）；退化用 agent.remember
        if engine is not None and hasattr(engine, "add_perception"):
            node = engine.add_perception(
                content,
                importance=0.6,
                spatial_coordinates={"x": x, "y": y, "z": z},
                tags=[tag, f"cat:{cat}", f"ent:{name}", "world_model"],
                entities=[name],
            )
        else:
            node = agent.remember(
                content=content, importance=0.6,
                tags=[tag, f"cat:{cat}", f"ent:{name}", "world_model"],
                entities=[name],
            )
        # 状态写入 state_attributes（store 直接更新）
        try:
            store.conn.execute(
                "UPDATE nodes SET state_attributes=? WHERE id=?",
                (json.dumps({"state": state}), node.id))
            store.conn.commit()
        except Exception:
            pass
        written.append(node.id)
    return written


def load_world_from_memory(agent_or_store, tag: str = "spatial") -> WorldModel:
    """从记忆库遍历 `spatial` 标签节点 → 重建世界模型场景。

    记忆就是缓存：场景 = 记忆节点的实时投影；新增/更新记忆节点即改场景。
    """
    store = getattr(agent_or_store, "store", agent_or_store)
    if not hasattr(store, "get_nodes_by_tag"):
        engine = getattr(agent_or_store, "engine", None)
        if engine is not None:
            store = getattr(engine, "store", store)
    nodes = store.get_nodes_by_tag(tag, limit=200)
    wm = WorldModel()
    for n in nodes:
        sp = n.spatial_coordinates or {}
        if not sp:
            continue
        pos = (sp.get("x", 0.0), sp.get("y", 0.5), sp.get("z", 5.0))
        # 类别 / 名称 / 状态从标签与 state_attributes 提取
        cat = "object"
        name = n.content.split(" ")[0] if n.content else "?" 
        for t in (n.tags or []):
            if t.startswith("cat:"):
                cat = t[4:]
            if t.startswith("ent:"):
                name = t[4:]
        state = "neutral"
        sa = n.state_attributes or {}
        if isinstance(sa, dict) and sa.get("state"):
            state = sa["state"]
        wm.add_entity(name, category=cat, pos=pos, state=state)
    wm.build()
    return wm


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import os

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)

    # 场景 1：肥鱼在房间（桌/椅/树），肥鱼害羞
    wm = WorldModel()
    wm.add_entity("肥鱼", "fatfish", pos=(0, 0.85, 5.0), state="shy")
    wm.add_entity("桌子", "table", pos=(1.5, 0.45, 6.0))
    wm.add_entity("椅子", "chair", pos=(2.0, 0.45, 5.5))
    wm.add_entity("树", "tree", pos=(-2.0, 1.0, 7.0))
    wm.relation("桌子", "right_of", "肥鱼")
    wm.relation("椅子", "near", "桌子")
    wm.relation("树", "behind", "肥鱼")
    wm.build()
    cam1 = Camera3D.look_at(eye=(0, 1.3, 1.2), target=(0, 0.9, 5.5), fov_deg=50)
    wm.render(500, 500, camera=cam1).save(os.path.join(out_dir, "worldmodel_scene.png"))
    print("场景 1（肥鱼害羞 + 桌/椅/树）→ data/worldmodel_scene.png")
    print("  ", wm.scene_text())

    # 场景 2：肥鱼开心（挥手）+ 侧视角
    wm2 = WorldModel()
    wm2.add_entity("肥鱼", "fatfish", pos=(0, 0.85, 5.0), state="happy")
    wm2.add_entity("访客", "person", pos=(0, 0.85, 6.5), state="happy")
    wm2.relation("肥鱼", "in_front", "访客")
    wm2.build()
    cam2 = Camera3D.look_at(eye=(1.8, 1.2, 1.5), target=(0, 0.9, 5.5), fov_deg=50)
    wm2.render(500, 500, camera=cam2).save(os.path.join(out_dir, "worldmodel_interact.png"))
    print("场景 2（肥鱼 + 访客，侧面）→ data/worldmodel_interact.png")
    print("  ", wm2.scene_text())
