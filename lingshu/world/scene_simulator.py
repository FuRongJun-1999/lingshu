# -*- coding: utf-8 -*-
"""scene_simulator · 场景级世界模拟器（世界模型阶段2 · 里程碑2.3）
============================================================================
核心（荣）：在服务器基础上，添加场景、实体、自主行为玩家。

SceneSimulator = WorldServer + 场景语义：
  - 场景（Scene）：环境/地形 + 实体集合 + 场景规则
  - 实体（SceneEntity）：玩家/动物/物品，带行为策略
  - 自主行为玩家（Autonomous）：确定性行为策略（wander/seek/avoid/flee/follow）
  - 决策循环：每 tick 所有自主实体各自决策 → 行动 → 世界响应 → 场景演化记录

行为策略（确定性 · 零 LLM · D-005）：
  - wander：随机游走（探索）
  - seek(target)：向目标移动（追逐/前往）
  - avoid(entity)：避开某实体（绕障）
  - flee(predator)：逃离追捕者（逃跑）
  - follow(path)：沿路径移动（巡逻）

受控动作（F08 · 沙盒离散移动 · 真实执行侧）：
  - set_command(entity_id, action, step_size)：登记受控实体**下一 tick** 的离散动作
    （east/west/north/south/stay，见 ACTION_NAMES），由 step() 消费一次；
  - 受控实体在该 tick **不**走自主行为分支（_decide 不被调用、不消耗 _rng），
    位移严格＝ 方向 × step_size（越界按边界钳制、坐标取整 2 位）；
  - 这是「执行 actuator」：它按登记的动作移动**实际**实体，不参与任何预测/评分。

设计参考：
  - Cosmos（3d-world/world-model）：世界基础模型（场景级演化）
  - DynamicCity（3d-world/4d-dynamic）：4D 占用序列
  - 游戏 NPC 行为树（确定性决策）
  - 智能论 3.4：多路并行 + 反馈闭环

环境影响层（本件 · 物理深化 C；规格见 `_evidence/lead/PHYSICS-BRIEF.md`）：
  - `Environment`：**整场级**环境参数面（重力/介质/地形摩擦/时间步），
    经 `set_environment(**kw)` 登记、`environment` 读取、`to_dict()` 可复现；
  - **每实体级覆盖**：`set_entity_environment(eid, **kw)`（字段级覆盖，优先级
    实体 > 整场 > 缺省）；
  - **动力学实体**：`SceneEntity.attrs["physics"] == "dynamics"` 的实体由
    `physics.simulate(...)`（队友 A · task-10）推进位置与速度，读数写入
    `entity.attrs["physics_readout"]`（模型/环境/是否闭式/t/dt/tick）。

★ **缺省行为逐位不变（硬判据）**：不登记环境、不把实体标成 `"dynamics"` 时，
`step()` 走的仍是既有运动学路径（沿注册路径平移），逐位与改动前一致。
环境**不会**自己动任何实体 —— 只有**显式**标成动力学实体的那些才受环境驱动。

诚实边界（本层**不**做的）：
  - **本层不自造物理**：一切演化都走 A 的 `physics.simulate`。三维二次阻力 / 介质+地形
    同时作用 ⇒ A 走**定步长积分**（读数 `integrated=True`、`steps>0`）；本层**恒显式传
    `dt`**（A 在无闭式时不静默选步长），故该路径不会因缺 `dt` 抛错；
  - **不做实体-实体碰撞**（属 A 的 `collision` 族，未接；要接需 Lead 裁决）；
  - **恒定坡度**（`slope_deg` 是整场/每实体一个常数，不支持地形函数）；
  - **体素占用不参与实体碰撞**（既有口径，#249 已如实收窄为「仅边界约束」）；
    本层沿用既有边界语义：x/z 走 `coord_contract.clamp_xz`、y 走
    `ground_level + 0.5` 地面钳（与 `VoxelWorld.simulate` 的「不穿地」同口径）。
  - **物理速度不量化**：动力学实体存 `physics.simulate` 给出的**物理速度**（不
    取整 2 位，与 `VoxelWorld.simulate` 同样不量化）——量化会把误差逐 tick 喂回
    下一 tick 的状态。运动学实体仍按 #193-B「本 tick 实际位移」取整 2 位。

纯标准库 · 零外部依赖（D-005）
"""
from __future__ import annotations

import math
import random
import time
import uuid
import warnings
from collections import deque
from dataclasses import dataclass, field, asdict
from .coord_contract import clamp_xz              # 第 4 轮 #4
from typing import Deque, Dict, List, Optional, Tuple

try:
    from .voxel_world import VoxelWorld, BLOCK_AIR
except ImportError:
    from .voxel_world import VoxelWorld, BLOCK_AIR


#: 单次 step 的 tick 上限（#424：此前 n 无上限直接透传，单次调用即可把
#: _history/_behavior_log 撑到 OOM）。上限值经验标定（追不到理论章节）：
#: 取 1e5，使单次调用内存峰值 ≈ 5 实体 × 1e5 × 400 B ≈ 200 MB 以内。
MAX_STEPS_PER_CALL = 100_000
#: 演化历史 / 行为日志环形缓冲长度（#424：长期运行只增不减）
HISTORY_MAXLEN = 10_000
BEHAVIOR_LOG_MAXLEN = 50_000

#: F08 受控实体的离散动作空间（平面 x/z，y 不变）——本表是本仓**唯一**方向真源。
#: 约定（本仓此前未定义方位 ⇒ 此处显式钉住）：east=+x / west=-x /
#: north=-z / south=+z / stay=原地不动。
ACTION_DELTAS: Dict[str, Tuple[float, float]] = {
    "east": (1.0, 0.0), "west": (-1.0, 0.0),
    "north": (0.0, -1.0), "south": (0.0, 1.0), "stay": (0.0, 0.0)}
#: 动作表的**稳定顺序**——决策 tie-break 与审计候选顺序的唯一来源（确定性）。
ACTION_NAMES: Tuple[str, ...] = ("east", "west", "north", "south", "stay")


# ---------------------------------------------------------------------------
# 环境影响层（物理深化 C）
# ---------------------------------------------------------------------------

#: `SceneEntity.attrs["physics"]` 取值：**运动学**（缺省，现状：沿注册路径平移）。
PHYSICS_KINEMATIC = "kinematic"
#: `SceneEntity.attrs["physics"]` 取值：**动力学**（受环境驱动，走 `physics.simulate`）。
PHYSICS_DYNAMICS = "dynamics"
#: 合法取值（fail-closed：标记打错不许静默降级成"没有物理"）。
PHYSICS_MODES: Tuple[str, ...] = (PHYSICS_KINEMATIC, PHYSICS_DYNAMICS)

#: `Environment.gravity` 的回退值——`physics.DEFAULT_G` 的**惰性**同源拷贝。
#: 为什么不在模块级 `import physics`：`physics.py` 在同批由队友 A（task-10）改写，
#: 模块级 import 会让「A 的文件处于半改状态」直接打断本模块的**导入**。
#: 取值与 `physics.DEFAULT_G` 由守卫钉住相等（`test_scene_sim_environment_guard`）。
DEFAULT_G_FALLBACK = 9.80665


def default_gravity() -> float:
    """惰性取 `physics.DEFAULT_G`（取不到则回退 `DEFAULT_G_FALLBACK`）。

    只在**构造 `Environment`** 时调用一次 ⇒ 不在模块导入期依赖 `physics`。
    """
    try:
        from . import physics
    except ImportError:                                  # pragma: no cover —— 独立加载
        return DEFAULT_G_FALLBACK
    return float(getattr(physics, "DEFAULT_G", DEFAULT_G_FALLBACK))


@dataclass
class Environment:
    """环境影响参数面（**整场级**；`SceneSimulator.environment`）。

    缺省 = **现状**：平地（`slope_deg=0`）、真空（`drag_c=0`）、标准重力
    （`gravity=physics.DEFAULT_G`）、1 秒/tick。⇒ 只要没有实体被标成
    `attrs["physics"] == "dynamics"`，登记环境**不会移动任何实体**（硬判据）。

    · `gravity` 是**沿 `-up` 轴的重力强度**（**须 > 0**；方向由 up 决定。内核
      `physics.make_environment` 要求 `gravity > 0`（`drag2` 的闭式真值以
      `v_t = sqrt(g/beta)` 为参数，`g = 0` 时退化）⇒ 本层同向拒 `0`，不提供
      「零重力」模式；做「隔离阻力」实验请用一个**极小正数**，实测 `1e-09` 两侧都通过）；
    · `up` 是上方向的轴索引（lingshu 世界 Y-up ⇒ 缺省 1）；
    · `slope_deg` / `mu_s` / `mu_k`：恒定坡度与静/动摩擦系数（库仑摩擦）；
    · `drag_c`：二次流体阻力系数（`c`，见 `physics.drag2_truth`）；`mass` 为质量；
    · `dt`：每 tick 的秒数（**唯一时间基准**；本层不支持每实体不同 dt）。

    校验 fail-closed：非有限 / 非正的 `gravity` / 负的摩擦与阻力 / 非正质量或步长 /
    `up` 越界 / `|slope_deg| >= 90` 一律 `ValueError`——不静默夹取、不静默改写。
    """
    gravity: Optional[float] = None
    up: int = 1
    slope_deg: float = 0.0
    mu_s: float = 0.0
    mu_k: float = 0.0
    drag_c: float = 0.0
    mass: float = 1.0
    dt: float = 1.0
    #: 下坡的**水平轴**（0/1/2；须 != `up`）。约定与 A 的 `terrain.incline_axis` 一致：
    #: 下坡 = 水平 `+incline_axis`、竖直 `−up`（缺省 up=1 ⇒ `û=(cosθ,−sinθ,0)`）。
    incline_axis: int = 0

    def __post_init__(self):
        if self.gravity is None:
            self.gravity = default_gravity()
        if isinstance(self.up, bool) or self.up not in (0, 1, 2):
            raise ValueError("Environment.up 需为 0/1/2（轴索引），收到 %r" % (self.up,))
        for name in ("gravity", "slope_deg", "mu_s", "mu_k", "drag_c", "mass", "dt"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float))                     or not math.isfinite(float(v)):
                raise ValueError("Environment.%s 需为有限数值，收到 %r" % (name, v))
            setattr(self, name, float(v))
        if self.gravity <= 0:
            raise ValueError("Environment.gravity 需 > 0（沿 -up 轴的强度，方向由 up 决定）："
                             "%r——内核 `physics.make_environment` 要求 `gravity > 0`，"
                             "本层不做第二套约定" % (self.gravity,))
        if self.mass <= 0:
            raise ValueError("Environment.mass 需 > 0：%r" % (self.mass,))
        if self.dt <= 0:
            raise ValueError("Environment.dt 需 > 0：%r" % (self.dt,))
        for name in ("mu_s", "mu_k", "drag_c"):
            if getattr(self, name) < 0:
                raise ValueError("Environment.%s 需 >= 0：%r" % (name, getattr(self, name)))
        # 坡度口径**对齐 A 的 `physics._normalize_terrain`**（[0, 90) 度）：本层不做
        # 第二套约定，否则会出现"本层收下、A 抛错"的分裂。
        if not (0.0 <= self.slope_deg < 90.0):
            raise ValueError("Environment.slope_deg 需在 [0, 90) 度（垂直不是坡度）：%r"
                             % (self.slope_deg,))
        if isinstance(self.incline_axis, bool) or self.incline_axis not in (0, 1, 2):
            raise ValueError("Environment.incline_axis 需为 0/1/2，收到 %r"
                             % (self.incline_axis,))
        if self.incline_axis == self.up:
            raise ValueError("Environment.incline_axis(%r) 不得等于 up(%r)——下坡的水平轴"
                             "与竖直轴不能同轴" % (self.incline_axis, self.up))

    def to_dict(self) -> Dict:
        """环境参数副本（可登记、可复现）。"""
        return asdict(self)

    def to_physics_env(self) -> Dict:
        """转成 A 冻结的 `physics.make_environment(...)` 环境上下文。

        · `drag_c == 0` ⇒ `medium=None`（A 的 `medium` 要求 `drag_c > 0`；0 = 真空）；
        · `slope_deg == 0 且 mu_s == 0 且 mu_k == 0` ⇒ `terrain=None`（平地）；
        · 其余原样透传（本层 `slope_deg` 已按 A 的口径限在 `[0, 90)`）。
        惰性 import：`physics.py` 与本法同批改写，模块级 import 会拖垮本模块导入。
        """
        from . import physics
        medium = ({"drag_c": self.drag_c, "mass": self.mass}
                  if self.drag_c > 0 else None)
        terrain = ({"slope_deg": self.slope_deg, "mu_s": self.mu_s, "mu_k": self.mu_k,
                    "incline_axis": self.incline_axis}
                   if (self.slope_deg > 0 or self.mu_s > 0 or self.mu_k > 0) else None)
        return physics.make_environment(gravity=self.gravity, up=self.up,
                                        medium=medium, terrain=terrain)

    def merged(self, **overrides) -> "Environment":
        """字段级覆盖 → 新 `Environment`（不就地改本对象）。"""
        data = self.to_dict()
        for k in overrides:
            if k not in data:
                raise ValueError("未知环境字段 %r（可用：%s）"
                                 % (k, "/".join(sorted(data))))
        data.update(overrides)
        return Environment(**data)


class _EnvView(dict):
    """环境视图：**既像 dict 又像对象**（`v["drag_c"]` 与 `v.drag_c` 都可用）。

    为什么要两副面孔：队友 A 的 `physics.simulate(state, env, t, …)` 里 `env` 到底
    是 dict 还是对象，在 A 落地前无从确认（PHYSICS-BRIEF 只冻结了名字）。给一个
    两种取法都成立的视图 ⇒ **一次调用**即可兼容，不必写成"先试对象、TypeError 再
    试 dict"（那种写法会把 A 内部真正的 TypeError 一起吞掉 ⇒ 静默降级）。
    """
    __slots__ = ()

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


class _StateView(_EnvView):
    """动力学状态视图（`pos`/`vel`/`mass`/`category`/`id`/`attrs`），同样双面孔。"""
    __slots__ = ()


def _unpack_sim(out) -> Tuple[Tuple, Tuple, Dict]:
    """把 `physics.simulate` 的返回统一成 `(pos, vel, meta)`；解析不了 ⇒ `TypeError`。

    · 主形态（A · task-10 的**冻结面**）：`{"pos","vel","kind","env_used","integrated",
      "steps","closed_form","reason",…}` ⇒ `meta` 原样带出（供读数）；
    · 兼容形态 `(pos, vel)`（旧签名）⇒ `meta = {}`。
    解析不了就**抛** —— 不静默当成"没动"（那会把接线错误伪装成"物理上确实没动"）。
    """
    if isinstance(out, dict):
        pos = out.get("pos")
        vel = out.get("vel", out.get("velocity"))
        if pos is not None and vel is not None:
            return tuple(pos), tuple(vel), out
    elif isinstance(out, (tuple, list)) and len(out) == 2:
        return tuple(out[0]), tuple(out[1]), {}
    raise TypeError("physics.simulate 返回体无法解析（期望 dict{pos, vel, …} 或 (pos, vel)）：%r"
                    % (out,))


@dataclass
class SceneEntity:
    """场景实体：位置/速度/行为策略/目标。

    behavior: wander / seek / avoid / flee / follow
    goal: 行为目标（seek 的目标实体 id / follow 的路径点列表）

    `attrs` 里两个**本件新增**的可选键（缺省 ⇒ 既有行为逐位不变）：
      - `attrs["physics"]`：`"kinematic"`（缺省，沿注册路径平移）或
        `"dynamics"`（受环境影响，走 `physics.simulate`）；取值非法 ⇒ `step()` 抛错；
      - `attrs["environment"]`：本实体的环境**字段级覆盖**（dict）或一个完整
        `Environment`（整场级的优先级低于它）。
    """
    category: str = "entity"
    pos: Tuple[float, float, float] = (0.0, 1.5, 0.0)
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    behavior: str = "wander"
    goal: str = ""                    # 目标实体 id 或路径 id
    speed: float = 0.3                # 移动速度
    id: str = field(default_factory=lambda: "scene_" + uuid.uuid4().hex[:8])
    attrs: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


class SceneSimulator:
    """场景级世界模拟器：多自主实体 + 决策循环 + 场景演化。

    能力：
      - create_scene(size)：创建场景（体素世界）
      - add_entity(category, behavior, pos, speed)：添加自主实体
      - add_path(path_id, points)：定义巡逻路径
      - set_environment(**kw) / set_entity_environment(eid, **kw)：登记环境影响
      - set_entity_physics(eid, p0=…, v0=…, beta=…) / reset_entity_physics(eid)：动力学初值
      - set_entity_collision_box(eid, half_or_box, static=…) / collision_report()：防穿模
      - step(n)：推进 n tick（受控动作 > 物理驱动 > 自主行为；三者各 tick 至多一条）
      - scene_state()：场景状态（实体+行为+演化历史）
      - entity_positions()：全部实体位置（4D 占用）
      - behavior_log()：实体决策记录（自主行为可审计）
    """

    def __init__(self, size: int = 24, ground_level: int = 1, seed: int = 42):
        # seed 必须**同时**播种两条独立随机流（此前只播了下面那条实体行为流）：
        #   · VoxelWorld._rng → 世界构建（build_flatland 的种树位置）
        #   · self._rng       → 实体行为（wander 抖动）
        # 漏传第一处的后果：`create_scene` 的地形与 seed 完全无关（VoxelWorld 恒用
        # 自身缺省 seed=0），「换 seed 重跑」在世界构建维度上是空转。
        self.world = VoxelWorld(size=size, ground_level=ground_level, seed=seed)
        # #1096：seed 的**可读回**。VoxelWorld 不留 seed（只吃进生成器），而
        # `CuriosityExplorer` 在 `world=` 注入路径下需要从注入世界回填 seed
        # （`_build_world` 重建「同世界配置」对比世界）——没有这个属性就只能
        # 读到形参缺省值。只增只读字段，既有读取方不受影响。
        self.seed = int(seed)
        self.entities: Dict[str, SceneEntity] = {}
        self.paths: Dict[str, List[Tuple[float, float, float]]] = {}
        # #424：环形缓冲——此前是普通 list，长期运行 / 单次大 n 只增不减，
        # 单次 n=1e6（5 实体）即 ≈2 GB，n=4e6 被 OOM killer 杀掉进程。
        # 读取方只有 len()/尾部切片/evolution()，故保留最近窗口即可。
        self._history: Deque[Dict] = deque(maxlen=HISTORY_MAXLEN)
        self._behavior_log: Deque[Dict] = deque(maxlen=BEHAVIOR_LOG_MAXLEN)
        # follow 的"已提交目标路径点"索引（entity_id -> path 下标）。
        # 巡逻必须**提交**目标：若每 tick 只重算"最近点"，实体推进一格后
        # 最近点会立刻变回原点 ⇒ 原地振荡，永不前进。
        self._follow_target: Dict[str, int] = {}
        # F08：受控实体**待消费**的离散动作（entity_id -> {action, step_size}）。
        # 单次消费：step() 取走即删——登记的动作只作用于紧接的那一个 tick，
        # 不登记 ⇒ 行为与历史完全一致（默认关闭时零影响）。
        self._commands: Dict[str, Dict] = {}
        self.tick_count = 0
        # 确定性随机：**同一实体插入顺序下**可复现（#428：step 按 entities 迭代
        # 序决策，随机数按该序切分，且决策读目标 pos 后同循环写回 ⇒ 换插入顺序
        # 结果变）。seed 缺省 42 = 历史硬编码值，既有实验行为不变。
        # 顺序无关化（排序/分相/每实体 rng）属设计裁定，本处不改（见 step
        # docstring 与 .tmp/待裁清单_20261010.md #428）。
        # #379 兼容性声明（如实）：本改动**同时**让 __init__ 起始处开始透传 seed
        # 给 VoxelWorld；此前 VoxelWorld 恒为自身缺省 0，故 `create_scene` 的
        # **地形默认输出会变**（树位置由 seed=0 的流改为 seed=42 的流）。这是修复
        # "seed 对世界构建无效"的必然后果，无法两全：要么地形与 seed 无关（旧），
        # 要么与 seed 有关（新）。
        self._rng = random.Random(seed)
        # 环境影响层（本件）：**整场级**环境（缺省 = 现状：平地/真空/标准重力）。
        # 登记环境本身**不移动任何实体**——只有被标成 `"dynamics"` 的实体才受它驱动。
        self._environment = Environment()
        # `physics.simulate` 未落地时按实体去重告警（每实体只喊一次，不刷屏）。
        self._physics_pending_warned: set = set()
        # 碰撞箱登记面（本件）：**缺省为空 ⇒ step() 完全不走碰撞分支**（硬判据）。
        # 值为 {"half","rot","offset","static","mass"}；世界箱每 tick 由实体 pos 重建。
        self._collision_boxes: Dict[str, Dict] = {}
        # 本 tick 的接触/分离读数（`collision_report()` 读它；可复现）。
        self._collision_last: List[Dict] = []
        # 本 tick 起始位置（**只在有碰撞箱时**建立 ⇒ 缺省路径零额外工作）。
        self._tick_start_pos: Dict[str, Tuple[float, float, float]] = {}

    # ---- 场景构建 ----

    def create_scene(self, trees: int = 4, water: bool = True) -> Dict:
        """创建场景（体素世界 + 地形）。"""
        blocks = self.world.build_flatland(trees=trees, water=water)
        return {"status": "ok", "blocks": blocks, "size": self.world.size,
                "scene": "flatland"}

    def add_entity(self, category: str, behavior: str = "wander",
                   pos: Tuple[float, float, float] = (2, 1.5, 2),
                   speed: float = 0.3, goal: str = "") -> str:
        """添加自主行为实体。返回 entity_id。"""
        e = SceneEntity(category=category, pos=tuple(float(v) for v in pos),
                        behavior=behavior, speed=speed, goal=goal)
        self.entities[e.id] = e
        return e.id

    def add_path(self, path_id: str, points: List[Tuple[float, float, float]]) -> None:
        """定义巡逻路径（follow 行为用）。

        #398：此前只做 `tuple(float(v))`，不校验维度/非空——2D 点照单全收，
        此后 `_decide` 的 follow 分支每次 `path[i][2]` 抛 IndexError（实体与
        路径常驻，之后每次 step 都崩，且失败 tick 仍计数）；空路径则使世界侧
        `if path:` 静默退化为随机游走，而验证器仍按 exact 判分。
        现拒绝：空路径 / 非 3 维点 / 非有限坐标。
        """
        pts = [tuple(float(v) for v in pt) for pt in points]
        if not pts or any(len(p) != 3 or not all(math.isfinite(v) for v in p)
                          for p in pts):
            raise ValueError(
                "path %r 需为非空的 (x, y, z) 有限坐标列表，收到 %r"
                % (path_id, points))
        self.paths[path_id] = pts

    # ---- 自主行为决策（确定性）----

    def _decide(self, e: SceneEntity) -> Tuple[float, float, float]:
        """行为决策 → 期望移动方向（dx, dy, dz）。"""
        bx, by, bz = e.pos

        if e.behavior == "seek":
            # 向目标实体移动
            target = self.entities.get(e.goal)
            if target is not None:
                dx = target.pos[0] - bx
                dz = target.pos[2] - bz
                return self._normalize(dx, 0, dz)
            return self._unresolved(e)

        elif e.behavior == "avoid":
            # 远离目标实体（绕障）
            target = self.entities.get(e.goal)
            if target is not None:
                dx = bx - target.pos[0]
                dz = bz - target.pos[2]
                return self._normalize(dx, 0, dz)
            return self._unresolved(e)

        elif e.behavior == "flee":
            # 逃离追捕者（反向 + 随机扰动）
            predator = self.entities.get(e.goal)
            if predator is not None:
                dx = bx - predator.pos[0]
                dz = bz - predator.pos[2]
                d = self._normalize(dx, 0, dz)
                return (d[0] + self._rng.uniform(-0.1, 0.1),
                        0, d[2] + self._rng.uniform(-0.1, 0.1))
            return self._unresolved(e)

        elif e.behavior == "follow":
            # 沿路径巡逻
            path = self.paths.get(e.goal)
            if path:
                # 取**已提交**的目标路径点；首次（或路径已变短）取最近点
                idx = self._follow_target.get(e.id)
                if idx is None or not (0 <= idx < len(path)):
                    idx = min(range(len(path)),
                              key=lambda i: math.hypot(path[i][0] - bx,
                                                       path[i][2] - bz))
                # 到达判定：已足够接近**当前目标** ⇒ 提交推进到下一个点。
                # 容差取 e.speed（"下一 tick 即可抵达"的距离）。缺此提交时，
                # 实体或在路径点上永久静止（方向为零向量），或只在最近点两侧
                # 振荡——「沿路径巡逻」两种情况都不成立。
                tgt = path[idx]
                if math.hypot(tgt[0] - bx, tgt[2] - bz) <= max(e.speed, 1e-9):
                    idx = (idx + 1) % len(path)
                self._follow_target[e.id] = idx
                tgt = path[idx]
                return self._normalize(tgt[0] - bx, 0, tgt[2] - bz)
            return self._unresolved(e)

        # wander（默认）：随机游走（确定性随机）
        return (self._rng.uniform(-1, 1), 0, self._rng.uniform(-1, 1))

    def _unresolved(self, e: SceneEntity) -> Tuple[float, float, float]:
        """goal 不可解析时的显式失败（issue #193-C）。

        此前 seek/avoid/flee/follow 的 `if 目标:` 守卫不成立时**静默穿透**到
        末尾的 wander 返回式：行为被改写成随机游走、且额外消耗共享 `_rng`
        （污染同场景其它依赖确定性随机的观测）。现改为显式告警 + 零向量
        （本 tick 不动），既不退化也不消耗 `_rng`。
        """
        warnings.warn(
            "scene_simulator: behavior=%s 的目标 %r 不可解析 ⇒ 本 tick 不移动"
            "（不再静默退化为 wander）" % (e.behavior, e.goal),
            RuntimeWarning, stacklevel=3)
        return (0.0, 0.0, 0.0)

    def _normalize(self, dx, dy, dz) -> Tuple[float, float, float]:
        n = math.hypot(dx, dz)
        if n < 1e-6:
            return (0.0, 0.0, 0.0)
        return (dx / n, dy, dz / n)

    # ---- F08 受控动作（真实执行侧 actuator）----

    @staticmethod
    def validate_action(action: str, step_size: float = 1.0) -> float:
        """fail-closed 校验离散动作 + 步长，返回 float(step_size)。

        未知动作 / 非数值 / 非有限 / 非正步长一律 ValueError——不静默夹取、
        不静默改写（判据＝《PR 审核细则 v0.1》§三.3 边界诚实：不静默降级）。
        """
        if action not in ACTION_DELTAS:
            raise ValueError("未知动作 %r（可用：%s）"
                             % (action, "/".join(ACTION_NAMES)))
        if isinstance(step_size, bool) or not isinstance(step_size, (int, float)):
            raise ValueError("step_size 需为有限正数，收到 %r" % (step_size,))
        size = float(step_size)
        if not math.isfinite(size) or size <= 0:
            raise ValueError("step_size 需为有限正数，收到 %r" % (step_size,))
        return size

    def set_command(self, entity_id: str, action: str,
                    step_size: float = 1.0) -> Dict:
        """登记受控实体**下一 tick** 的离散动作（单次消费）。返回登记副本。

        校验先于写入：失败时既不改 _commands，也不触碰实体状态（无污染）。
        """
        size = self.validate_action(action, step_size)
        if entity_id not in self.entities:
            raise ValueError("受控实体 %r 不在场景中（现有：%s）"
                             % (entity_id, "/".join(sorted(self.entities))))
        self._commands[entity_id] = {"entity": entity_id, "action": action,
                                     "step_size": size}
        return dict(self._commands[entity_id])

    def clear_command(self, entity_id: Optional[str] = None) -> Dict:
        """撤销待消费动作（entity_id=None ⇒ 撤销全部）。返回撤销清单。"""
        if entity_id is None:
            removed = sorted(self._commands)
            self._commands.clear()
        else:
            removed = [entity_id] if entity_id in self._commands else []
            self._commands.pop(entity_id, None)
        return {"removed": removed, "pending": sorted(self._commands)}

    def commands(self) -> Dict[str, Dict]:
        """当前待消费动作（副本，供审计）。"""
        return {eid: dict(cmd) for eid, cmd in self._commands.items()}

    # ---- 环境影响层（物理深化 C）----

    @property
    def environment(self) -> "Environment":
        """当前**整场级**环境（副本语义：返回对象本身，改动请走 `set_environment`）。"""
        return self._environment

    def set_environment(self, **kw) -> Dict:
        """登记整场级环境（字段级；未给的字段沿用当前值）。返回生效副本。

        校验先于写入（fail-closed）：非法值 ⇒ `ValueError` 且**不改**当前环境。
        登记环境本身**不移动任何实体**：只有 `attrs["physics"] == "dynamics"` 的
        实体才走动力学路径 ⇒ 既有场景逐位不变。
        """
        self._environment = self._environment.merged(**kw)
        return self._environment.to_dict()

    def clear_environment(self) -> Dict:
        """恢复缺省环境（平地/真空/标准重力）。返回缺省副本。"""
        self._environment = Environment()
        return self._environment.to_dict()

    def set_entity_environment(self, entity_id: str, **kw) -> Dict:
        """登记**每实体级**环境覆盖（字段级，优先级高于整场级）。返回该实体生效值。

        实体不在场景中 / 字段非法 ⇒ `ValueError`（不静默忽略）。
        """
        if entity_id not in self.entities:
            raise ValueError("实体 %r 不在场景中（现有：%s）"
                             % (entity_id, "/".join(sorted(self.entities))))
        base = self._environment.merged(**kw)              # 复用同一套字段校验
        merged = {k: base.to_dict()[k] for k in kw}        # 只留被覆盖的字段
        self.entities[entity_id].attrs["environment"] = merged
        return dict(merged)

    def clear_entity_environment(self, entity_id: Optional[str] = None) -> Dict:
        """撤销每实体级覆盖（None ⇒ 全部撤销）。返回撤销清单。"""
        if entity_id is None:
            removed = sorted(eid for eid, e in self.entities.items()
                             if "environment" in e.attrs)
            for e in self.entities.values():
                e.attrs.pop("environment", None)
        else:
            e = self.entities.get(entity_id)
            removed = [entity_id] if (e is not None and "environment" in e.attrs) else []
            if e is not None:
                e.attrs.pop("environment", None)
        return {"removed": removed}

    def environment_readout(self, entity_id: Optional[str] = None) -> Dict:
        """环境层可读读数：整场环境 + （给了实体）该实体的生效环境与物理读数。

        读数是**副本**，改它不动仿真；用于审计"这一 tick 用了哪个模型/环境、
        是否闭式、步长/步数"。
        """
        out: Dict = {"scene_environment": self._environment.to_dict(),
                     "dt": self._environment.dt, "tick": self.tick_count}
        if entity_id is None:
            out["entities"] = {eid: {"effective_environment": self._env_for(e).to_dict(),
                                     "physics": self._physics_mode(e),
                                     "readout": e.attrs.get("physics_readout")}
                               for eid, e in self.entities.items()}
            return out
        e = self.entities.get(entity_id)
        if e is None:
            raise ValueError("实体 %r 不在场景中（现有：%s）"
                             % (entity_id, "/".join(sorted(self.entities))))
        out["entity"] = entity_id
        out["effective_environment"] = self._env_for(e).to_dict()
        out["physics"] = self._physics_mode(e)
        out["readout"] = e.attrs.get("physics_readout")
        return out

    def _physics_mode(self, e: SceneEntity) -> str:
        """实体的物理模式（`"kinematic"` | `"dynamics"`）；取值非法 ⇒ ValueError。"""
        mode = e.attrs.get("physics", PHYSICS_KINEMATIC)
        if mode is None:
            mode = PHYSICS_KINEMATIC
        if mode not in PHYSICS_MODES:
            raise ValueError(
                "SceneEntity.attrs['physics'] 取值非法：%r（可用：%s）——"
                "不静默降级成 kinematic（那会让「声明了却没生效」无从察觉）"
                % (mode, "/".join(PHYSICS_MODES)))
        return mode

    def _is_dynamics(self, e: SceneEntity) -> bool:
        """实体是否被声明为**物理驱动**（缺省 False ⇒ 既有运动学路径逐位不变）。"""
        return self._physics_mode(e) == PHYSICS_DYNAMICS

    def _env_for(self, e: SceneEntity) -> "Environment":
        """该实体的**生效环境**：实体级覆盖（字段级）> 整场级 > 缺省。"""
        over = e.attrs.get("environment")
        if not over:
            return self._environment
        if isinstance(over, Environment):
            return over
        if isinstance(over, dict):
            return self._environment.merged(**over)
        raise ValueError("SceneEntity.attrs['environment'] 需为 dict 或 Environment，"
                         "收到 %r" % (type(over).__name__,))

    def set_entity_physics(self, entity_id: str, *, p0=None, v0=None, t_ref=None,
                           beta=None, radius=None, mass=None) -> Dict:
        """**显式**给动力学实体登记初值（`physics.simulate` 的 `state`）。

        A 的 `simulate(state, env, t, dt=…)` 取**绝对时刻** `t`（内部 `tau = t − t_ref`）
        —— 所以本层对每个动力学实体保存**一份初值** `{"p0","v0","t_ref"}`，逐 tick 只把
        `t` 往前推，而不是"每 tick 拿当前位置当新初值再外推"（那会把闭式解变成反复重启
        的伪积分）。不给初值时，首次动力学 tick 用实体**当时**的 `pos`/`velocity` 自动登记。

        `beta`/`radius`/`mass` 原样透传给 `simulate`（A 支持用它们覆盖环境缺省，如纯滚动
        `beta = I/(m r²)`）。
        """
        if entity_id not in self.entities:
            raise ValueError("实体 %r 不在场景中（现有：%s）"
                             % (entity_id, "/".join(sorted(self.entities))))
        e = self.entities[entity_id]
        st = dict(e.attrs.get("physics_state") or {})
        st["p0"] = tuple(float(v) for v in (p0 if p0 is not None else e.pos))
        st["v0"] = tuple(float(v) for v in (v0 if v0 is not None else e.velocity))
        st["t_ref"] = (float(t_ref) if t_ref is not None
                       else (st.get("t_ref") if st.get("t_ref") is not None
                             else self.tick_count * self._env_for(e).dt))
        for name, val in (("beta", beta), ("radius", radius), ("mass", mass)):
            if val is not None:
                st[name] = float(val)
        for k, v in list(st.items()):
            if isinstance(v, tuple) and not all(math.isfinite(x) for x in v):
                raise ValueError("set_entity_physics: %s 含非有限坐标：%r" % (k, v))
        e.attrs["physics_state"] = st
        return dict(st)

    def reset_entity_physics(self, entity_id: Optional[str] = None) -> Dict:
        """丢弃动力学实体的初值（下一 tick 用当时状态重新登记）。返回被清空的实体。"""
        if entity_id is None:
            cleared = sorted(eid for eid, e in self.entities.items()
                             if "physics_state" in e.attrs)
            for e in self.entities.values():
                e.attrs.pop("physics_state", None)
                e.attrs.pop("physics_readout", None)
        else:
            e = self.entities.get(entity_id)
            cleared = [entity_id] if (e is not None and "physics_state" in e.attrs) else []
            if e is not None:
                e.attrs.pop("physics_state", None)
                e.attrs.pop("physics_readout", None)
        return {"cleared": cleared}

    def _step_dynamics(self, eid: str, e: SceneEntity) -> Optional[Dict]:
        """物理驱动的单 tick 演化（环境层）：调 A 的 `physics.simulate` → 世界响应 → 写回。

        接线（A · task-10 的**冻结面**）：
          · `env` = `Environment.to_physics_env()`（`physics.make_environment` 的返回）；
          · `state` = 本实体登记的初值 `{"p0","v0","t_ref"[, "beta"/"radius"/"mass"]}`
            （闭式解按**绝对时刻** `t` 求值 ⇒ 逐 tick 只推 `t`，不重启积分）；
          · `t` = `tick_count * dt`；`dt` 恒显式传（A 在无闭式时才用它）。

        返回行动记录（有实际位移时）或 `None`。读数写进 `e.attrs["physics_readout"]`：
        `kind` / `env_used` / `integrated` / `steps` / `closed_form` / `reason` / `t` / `dt` / `tick`
        ——"用了哪个模型/环境、是否解析闭式、步长/步数"对外可读。

        `physics.simulate` 不存在时（极端：A 的文件半改状态）：**不假装成功、也不静默**
        —— 每实体告警一次 + 读数标 `pending`，本 tick 不演化。
        """
        env = self._env_for(e)
        t = self.tick_count * env.dt
        readout = {"entity": eid, "category": e.category, "physics": PHYSICS_DYNAMICS,
                   "tick": self.tick_count, "t": t, "dt": env.dt,
                   "environment": env.to_dict(), "kind": None, "closed_form": None,
                   "integrated": None, "steps": None, "reason": None}
        try:
            from . import physics as _phys
        except ImportError:                              # pragma: no cover —— 独立加载
            _phys = None
        api = getattr(_phys, "simulate", None) if _phys is not None else None
        if not callable(api):
            readout["pending"] = ("physics.simulate 不可用（A 的文件处于未就绪状态）；"
                                  "本层接线已就位，就绪后自动生效")
            e.attrs["physics_readout"] = readout
            if eid not in self._physics_pending_warned:
                self._physics_pending_warned.add(eid)
                warnings.warn(
                    "scene_simulator: 实体 %s 标记为 physics=%r，但 physics.simulate "
                    "不可用 ⇒ 本 tick 不演化" % (eid, PHYSICS_DYNAMICS),
                    RuntimeWarning, stacklevel=3)
            return None
        st = e.attrs.get("physics_state")
        if not isinstance(st, dict) or "p0" not in st or "v0" not in st:
            st = self.set_entity_physics(eid)            # 首 tick：用当时状态登记初值
            readout["seeded"] = dict(st)
        p_env = env.to_physics_env()
        out = _unpack_sim(api(st, p_env, t, dt=env.dt))
        pos = out[0] if len(out) > 0 else None
        vel = out[1] if len(out) > 1 else None
        meta = out[2] if len(out) > 2 else {}
        readout.update({k: meta.get(k) for k in
                        ("kind", "closed_form", "integrated", "steps", "reason")})
        readout["env_used"] = meta.get("env_used")
        nx, ny, nz = (float(pos[0]), float(pos[1]), float(pos[2]))
        if not all(math.isfinite(v) for v in (nx, ny, nz)):
            warnings.warn("scene_simulator: 动力学实体 %s 的下一位置非有限"
                          "（%r, %r, %r）⇒ 拒绝写入" % (eid, nx, ny, nz),
                          RuntimeWarning, stacklevel=3)
            e.velocity = (0.0, 0.0, 0.0)
            readout["rejected"] = "non_finite"
            e.attrs["physics_readout"] = readout
            return None
        # 世界响应：沿用**既有**边界语义（x/z 走 clamp_xz；y 走 ground_level+0.5
        # 地面钳，与 `VoxelWorld.simulate` 的「不穿地」同口径）——不新增碰撞语义。
        nx, nz = clamp_xz(nx, nz, self.world.size)
        ny = max(float(self.world.ground_level + 0.5), ny)
        old = e.pos
        e.pos = (nx, ny, nz)
        # 动力学实体存**物理速度**（不量化：量化会把误差逐 tick 喂回闭式解的初值）。
        e.velocity = (float(vel[0]), float(vel[1]), float(vel[2]))
        readout["pos"] = e.pos
        readout["vel"] = e.velocity
        e.attrs["physics_readout"] = readout
        self._behavior_log.append({"tick": self.tick_count, "entity": eid,
                                   "behavior": e.behavior, "physics": PHYSICS_DYNAMICS,
                                   "pos": e.pos})
        if e.pos == old:
            return None
        return {"entity": eid, "category": e.category, "behavior": e.behavior,
                "physics": PHYSICS_DYNAMICS, "new_pos": e.pos}

    # ---- 碰撞箱 / 防穿模（碰撞箱接线 · 用户第二条需求）----

    def _collision_api(self):
        """惰性取碰撞内核（`physics` 的 `sweep_*`/`resolve_penetration`）。

        **惰性 import**（函数体内）：`physics.py` 与本法同批改写，模块级 import 会
        拖垮本模块导入。缺内核 ⇒ 返回 `None`（由调用方决定 fail-closed 口径）。
        """
        try:
            from . import physics
        except ImportError:                              # pragma: no cover —— 独立加载
            return None
        for name in ("sweep_aabb", "sweep_obb", "aabb_mtv", "obb_mtv",
                     "resolve_penetration", "obb_of"):
            if not callable(getattr(physics, name, None)):
                return None
        return physics

    def set_entity_collision_box(self, entity_id: str, box_or_half_extents,
                                 *, static: bool = False) -> None:
        """给实体登记碰撞箱（opt-in；**不登记 ⇒ `step()` 逐位不变**）。

        参数
        ----
        · `box_or_half_extents`：
            - 三元组 ⇒ 半宽 `(hx,hy,hz)`，箱心 = 实体位置；
            - 箱体 dict（`physics.aabb_of/obb_of` 的返回，或
              `mesh_gen.collision_proxy(...)`）⇒ 取 `half`/`rot`，并把它的 `center`
              当作**相对实体位置的局部偏移**（网格局部系 → 世界系）；
        · `static=True`：该实体在推离时视为**不动**（无限质量）——挡板/墙用；它自己
          永不被推开（但仍参与拦截，且它的 `physics_state` 不被重播种）。

        校验 fail-closed：实体不存在 / 半宽非 3 个有限正数 / **碰撞内核不可用** ⇒ 抛错
        （**不静默当"没碰撞"**）。
        """
        if entity_id not in self.entities:
            raise ValueError("实体 %r 不在场景中（现有：%s）"
                             % (entity_id, "/".join(sorted(self.entities))))
        physics = self._collision_api()
        if physics is None:
            raise RuntimeError(
                "碰撞内核不可用（physics.sweep_aabb/resolve_penetration 缺失）⇒ "
                "拒绝登记碰撞箱（fail-closed：绝不静默当成'没碰撞'）")
        if isinstance(box_or_half_extents, dict):
            half = box_or_half_extents.get("half")
            rot = box_or_half_extents.get("rot")
            offset = box_or_half_extents.get("center") or (0.0, 0.0, 0.0)
            mass = box_or_half_extents.get("mass")
        else:
            half, rot, offset, mass = box_or_half_extents, None, (0.0, 0.0, 0.0), None
        half = tuple(float(v) for v in half)
        offset = tuple(float(v) for v in offset)
        if len(half) != 3 or any((not math.isfinite(v)) or v <= 0.0 for v in half):
            raise ValueError("碰撞箱半宽需为 3 个**有限正数**（零厚度箱不是代理），收到 %r"
                             % (half,))
        if len(offset) != 3 or not all(math.isfinite(v) for v in offset):
            raise ValueError("碰撞箱局部偏移需为 3 个有限数，收到 %r" % (offset,))
        if rot is not None:
            physics.obb_of(aabb={"center": (0.0, 0.0, 0.0), "half": half},
                           rot=rot)                     # 借内核校验 rot 形状/零行
        if mass is None:
            mass = self.entities[entity_id].attrs.get("mass")
        if mass is not None:
            mass = float(mass)
            if not math.isfinite(mass) or mass <= 0.0:
                raise ValueError("碰撞质量需为有限正数，收到 %r" % (mass,))
        self._collision_boxes[entity_id] = {"half": half, "rot": rot,
                                           "offset": offset, "static": bool(static),
                                           "mass": mass}

    def clear_entity_collision_box(self, entity_id: str) -> None:
        """撤销实体的碰撞箱（不存在则无操作——幂等）。"""
        self._collision_boxes.pop(entity_id, None)

    def _world_box(self, entity_id: str):
        """实体碰撞箱的**世界箱**（箱心 = 实体位置 + 局部偏移）。"""
        physics = self._collision_api()
        spec = self._collision_boxes[entity_id]
        pos = self.entities[entity_id].pos
        center = tuple(pos[i] + spec["offset"][i] for i in range(3))
        # 借内核构造箱体 ⇒ 键名/格式与 `COLLISION_SCHEMA` 同源，不自造第二套
        return physics.obb_of(aabb={"center": center, "half": spec["half"]},
                              rot=spec["rot"])

    def _mass_of(self, entity_id: str):
        """推离用的质量；`static` ⇒ `None`（无限质量，不动）。"""
        spec = self._collision_boxes[entity_id]
        if spec["static"]:
            return None
        return spec["mass"] if spec["mass"] is not None else 1.0

    def _reseed_physics(self, entity_id: str) -> None:
        """碰撞修正后**重播种**动力学初值（否则闭式解会从旧初值再穿一次）。

        运动学实体（`kinematic`）不重播种（它的 `velocity` 就是本 tick 实际位移）。
        """
        e = self.entities[entity_id]
        if "physics_state" not in e.attrs:
            return
        if self._physics_mode(e) != PHYSICS_DYNAMICS:
            return
        env = self._env_for(e)
        spec = self._collision_boxes.get(entity_id)
        if spec is not None and spec["static"]:
            return  # static=True ⇒ 它的 `physics_state` 不被重播种
        old_state = e.attrs["physics_state"]          # 先取旧值，再整体覆盖
        e.attrs["physics_state"] = {
            "p0": tuple(e.pos), "v0": tuple(e.velocity), "t_ref": self.tick_count * env.dt}
        for name in ("beta", "radius", "mass"):
            old = old_state.get(name)
            if old is not None:
                e.attrs["physics_state"][name] = old

    def _resolve_collisions(self) -> None:
        """本 tick 的碰撞后处理：**先 sweep 拦穿透、再 resolve 推离**（逐对）。

        口径（如实声明）：
        · 本 tick 的"速度"取**实际位移**（`pos − tick 起始位置`），`dt = 1.0`
          ⇒ `toi ∈ [0,1]` 是"本 tick 的时间比例"（与既有"velocity = 本 tick 实际
          位移"的运动学口径同源）；
        · 命中 ⇒ 两实体都**回退到 `toi`**（用内核 `_moved`），并把**朝对方的那一
          分量**从位移里扣掉（拦停，不做弹性反弹）；
        · 回退/推离后仍重叠（含 `toi == 0` 的初始重叠）⇒ `resolve_penetration`
          按质量反比推离；`static=True` 的一侧不动（无限质量）；
        · 动力学实体位置被修正后**重播种** `physics_state`（否则闭式解会从旧初值
          再穿一次）。
        """
        physics = self._collision_api()
        self._collision_last = []
        if physics is None:
            # fail-closed：有碰撞箱却无内核 ⇒ 响亮报错，绝不静默当"没碰撞"
            raise RuntimeError("碰撞内核不可用 ⇒ 无法做防穿模（拒绝静默降级）")
        ids = [eid for eid in self.entities if eid in self._collision_boxes]
        if len(ids) < 2:
            return
        start = self._tick_start_pos
        disp: Dict[str, Tuple[float, float, float]] = {}
        for eid in ids:
            s0 = start.get(eid, self.entities[eid].pos)
            disp[eid] = tuple(self.entities[eid].pos[i] - s0[i] for i in range(3))
        for ai in range(len(ids)):
            for bi in range(ai + 1, len(ids)):
                ia, ib = ids[ai], ids[bi]
                self._collide_pair(physics, ia, ib, disp)
        # 本 tick 的位移/速度口径：拦停后重算一次（运动学 = 实际位移）
        for eid in ids:
            e = self.entities[eid]
            s0 = start.get(eid, e.pos)
            moved = tuple(e.pos[i] - s0[i] for i in range(3))
            if self._physics_mode(e) == PHYSICS_DYNAMICS:
                env = self._env_for(e)
                e.velocity = tuple(moved[i] / env.dt for i in range(3))
            else:
                e.velocity = moved
            self._reseed_physics(eid)

    def _collide_pair(self, physics, ia: str, ib: str, disp: Dict) -> None:
        """单对实体的防穿透 + 推离（就地改 `pos`/`velocity`，并记读数）。"""
        da, db = disp[ia], disp[ib]
        start = self._tick_start_pos
        ba = physics.obb_of(aabb={"center": tuple(start.get(ia, self.entities[ia].pos)[i]
                                                 + self._collision_boxes[ia]["offset"][i]
                                                 for i in range(3)),
                                  "half": self._collision_boxes[ia]["half"]},
                            rot=self._collision_boxes[ia]["rot"])
        bb = physics.obb_of(aabb={"center": tuple(start.get(ib, self.entities[ib].pos)[i]
                                                 + self._collision_boxes[ib]["offset"][i]
                                                 for i in range(3)),
                                  "half": self._collision_boxes[ib]["half"]},
                            rot=self._collision_boxes[ib]["rot"])
        is_obb = self._collision_boxes[ia]["rot"] is not None             or self._collision_boxes[ib]["rot"] is not None
        sw = (physics.sweep_obb(ba, da, bb, db, 1.0) if is_obb
              else physics.sweep_aabb(ba, da, bb, db, 1.0))
        rec = {"i": ia, "j": ib, "hit": bool(sw["hit"]), "toi": sw["toi"],
               "depth": float(sw.get("depth") or 0.0),
               "depth_at_dt": float(sw.get("depth_at_dt") or 0.0),
               "axis": tuple(sw["normal"]) if sw.get("normal") else None,
               "conservative": bool(sw.get("conservative", False)),
               "resolved": False}
        if not sw["hit"]:
            self._collision_last.append(rec)
            return
        toi = float(sw["toi"] or 0.0)
        n = tuple(sw["normal"])
        if toi >= 0.0:      # ★含 toi==0（起始相切且相向）：此时公式退化为只留切向＝拦停
            # 回退到接触时刻（内核 `_moved`：走 toi 后的箱体 ⇒ 取其箱心反推实体位置）
            for eid, box, d in ((ia, ba, da), (ib, bb, db)):
                if self._collision_boxes[eid]["static"]:
                    continue
                moved = physics._moved(box, d, toi)
                c = moved["center"]
                off = self._collision_boxes[eid]["offset"]
                self.entities[eid].pos = tuple(c[i] - off[i] for i in range(3))
            # 拦停口径（★修复 v2）：**两侧同一套分解，不做符号门控**
            #   d_new = d·toi（走到接触点） + 切向分量·(1 − toi)（接触后沿面滑行）
            #   法向 = (d·n)n；切向 = d − (d·n)n（与朝向无关，故两侧同一式）。
            #   旧实现按"整 tick"施加切向分量并覆盖了 toi 推进 ⇒ 正面相撞时被推回起点、
            #   永久冻结（靠不近也停不下）；且对一侧用了 `comp > 0` 门控 ⇒ 另一侧完全不动。
            # ★分离态不拦停：`sweep_aabb` 对"起始相切"一律报 hit（t=0 确实接触），
            # 但若两者相对速度沿法向**离开**对方（dn_a 与 dn_b 同号且背向），拦停会把正在
            # 分离的实体冻住 ⇒ 此处跳过（只处理"相向/相切且相向"）。
            dn_a = sum(da[i] * n[i] for i in range(3))
            dn_b = sum(db[i] * n[i] for i in range(3))
            if dn_a - dn_b <= 0.0:
                self._collision_last.append(rec)
                return
            for eid, d in ((ia, da), (ib, db)):
                if self._collision_boxes[eid]["static"]:
                    continue
                dn = sum(d[i] * n[i] for i in range(3))
                tang = tuple(d[i] - dn * n[i] for i in range(3))
                norm = tuple(dn * n[i] for i in range(3))
                # 合规范式（本仓 time_core_lint 禁互补权重写法 `a·t + b·(1−t)` ⇒ 写 `b + (a−b)·t`）：
                new_d = tuple(tang[i] + (norm[i] - tang[i]) * toi for i in range(3))
                disp[eid] = new_d
                e = self.entities[eid]
                e.pos = tuple(self._tick_start_pos.get(eid, e.pos)[i] + new_d[i]
                              for i in range(3))
        # 仍重叠（含 toi==0 的初始重叠）⇒ 推离
        wa, wb = self._world_box(ia), self._world_box(ib)
        over = (physics.obb_overlap(wa, wb) if is_obb else physics.aabb_overlap(wa, wb))
        if over:
            mtv = (physics.obb_mtv(wa, wb) if is_obb else physics.aabb_mtv(wa, wb))
            axis, depth = (mtv if mtv is not None else ((0.0, 0.0, 0.0), 0.0))
            ma, mb = self._mass_of(ia), self._mass_of(ib)
            if ma is None and mb is None:
                sa = sb = (0.0, 0.0, 0.0)                # 两个都不动（都 static）
            elif ma is None:
                sa, sb = (0.0, 0.0, 0.0), tuple(axis[i] * depth for i in range(3))
            elif mb is None:
                sa, sb = tuple(-axis[i] * depth for i in range(3)), (0.0, 0.0, 0.0)
            else:
                sa, sb = physics.resolve_penetration(wa, wb, mass_a=ma, mass_b=mb)
            for eid, sh in ((ia, sa), (ib, sb)):
                if self._collision_boxes[eid]["static"]:
                    continue
                if sh == (0.0, 0.0, 0.0):
                    continue
                e = self.entities[eid]
                e.pos = tuple(e.pos[i] + sh[i] for i in range(3))
            rec["resolved"] = True
            rec["depth"] = float(depth)
            rec["axis"] = tuple(axis)
        self._collision_last.append(rec)

    def collision_report(self) -> Dict:
        """本 tick 的接触/分离读数（可读、可复现）。

        返回::

            {"available": bool,            # 碰撞内核是否可用（缺 ⇒ False，**不静默**）
             "tick": int, "n_boxes": int,
             "boxes": {eid: {"half","rot","offset","static","mass"}},
             "contacts": [{"i","j","hit","toi","depth","depth_at_dt","axis",
                           "conservative","resolved"}, …],
             "n_contacts": int, "max_depth": float}

        `max_depth` 取**本 tick 实际发生的最大穿透深度**（拦停后为 0；初始重叠被
        推离前为推离深度）；`depth_at_dt` 是"若不拦，dt 时刻两箱的嵌入深度"（诊断用），
        口径同 `physics.aabb_mtv`——三轴**同时**重叠时取各轴逃逸距离的**最小**值，
        dt 时刻不重叠（含相切/三轴非同时重叠）⇒ `0.0`；它**不参与决策**
        （`max_depth` 用的是 `depth`）。
        """
        physics = self._collision_api()
        contacts = [dict(c) for c in self._collision_last]
        depths = [c["depth"] for c in contacts if c.get("hit")]
        return {"available": physics is not None, "tick": self.tick_count,
                "n_boxes": len(self._collision_boxes),
                "boxes": {eid: dict(spec) for eid, spec in self._collision_boxes.items()},
                "contacts": contacts, "n_contacts": len(contacts),
                "max_depth": (max(depths) if depths else 0.0)}

    def _apply_command(self, eid: str, e: SceneEntity, cmd: Dict) -> Optional[Dict]:
        """执行受控动作（真实位移）——返回行动记录；原地 stay 返回 None。

        与自主行为分支共用同一套世界响应口径：边界钳制 + 坐标取整 2 位 +
        velocity＝本 tick 实际位移。step_size 已在登记时校验为有限正数，故
        位置必然有限（此处保留 #347 防御性拒绝）。
        """
        dx, dz = ACTION_DELTAS[cmd["action"]]
        step = cmd["step_size"]
        old = e.pos
        nx, nz = old[0] + dx * step, old[2] + dz * step
        if not (math.isfinite(nx) and math.isfinite(nz)):
            warnings.warn(
                "scene_simulator: 受控实体 %s 的下一位置非有限（%r, %r）⇒ 拒绝写入"
                % (eid, nx, nz), RuntimeWarning, stacklevel=3)
            e.velocity = (0.0, 0.0, 0.0)
            return None
        # ⚠ 边界语义收归 coord_contract.clamp_xz（与自主行为分支同一算式，逐字一致）
        nx, nz = clamp_xz(nx, nz, self.world.size)
        e.pos = (round(nx, 2), old[1], round(nz, 2))
        e.velocity = (round(e.pos[0] - old[0], 2), 0.0,
                      round(e.pos[2] - old[2], 2))
        # 决策记录（behavior_log）与自主行为同面：含动作名/步长，便于外部审计
        # 「这一 tick 谁被指挥动了」。stay 也留一条（动作确实被决策并执行，
        # 只是位移为 0）——而 actions 计数仍只数**实际位移**（#193-A 口径）。
        self._behavior_log.append({"tick": self.tick_count, "entity": eid,
                                   "behavior": e.behavior, "commanded": True,
                                   "action": cmd["action"], "step_size": step,
                                   "pos": e.pos})
        if e.velocity == (0.0, 0.0, 0.0):
            return None
        return {"entity": eid, "category": e.category, "behavior": e.behavior,
                "commanded": True, "action": cmd["action"], "new_pos": e.pos}

    # ---- 决策循环（场景演化）----

    def _approach_distance(self, e: SceneEntity) -> Optional[float]:
        """seek/follow 到目标的剩余平面距离（其余行为 / 目标不可解析 ⇒ None）。

        #249：`step` 的步长须按剩余距离封顶（`min(speed, dist)`），否则离目标
        不足 speed 时会越过目标、下一 tick 再反向越过 ⇒ 周期 2 永久振荡。
        必须在 `_decide` **之后**调用（follow 的已提交目标点由 `_decide` 推进）。
        """
        if e.behavior == "seek":
            t = self.entities.get(e.goal)
            if t is not None:
                return math.hypot(t.pos[0] - e.pos[0], t.pos[2] - e.pos[2])
        elif e.behavior == "follow":
            path = self.paths.get(e.goal)
            if path:
                idx = self._follow_target.get(e.id)
                if idx is None or not (0 <= idx < len(path)):
                    return None
                tgt = path[idx]
                return math.hypot(tgt[0] - e.pos[0], tgt[2] - e.pos[2])
        return None

    def step(self, n: int = 1) -> Dict:
        """推进 n tick：所有自主实体决策 → 行动 → 世界响应 → 场景演化。

        返回 `actions` = **最后一个 tick 实际发生位移的实体数**（#193-A：此前
        返回的是实体总数，与真算好的行动列表不同义）。

        #428（成立）：轨迹依赖实体**插入顺序**——本循环 `for eid, e in
        self.entities.items()` 的迭代序即插入序，且循环体内有两个顺序敏感点：
        ① `_decide` 按该序从共享 `self._rng` 取随机数（wander/flee 分支）⇒
        同一随机数流按迭代序切分给各实体；② `_decide` 读目标实体 `pos`，同循环
        随即写回 `e.pos` ⇒ 先动者改变后动者的决策依据。故「确定性随机」仅保证
        **同一插入顺序**下可复现（同 seed 逐位一致）；换插入顺序结果变（issue
        #428 实测：同 seed 换序末态差最大 0.77 m，`world_learner.speed_est`
        偏差约 14%）。本处**如实声明契约**，不改数值行为——顺序无关化（排序 /
        决策-行动分相 / 每实体 rng 流）三种取向都会改变既有实验数值，属设计
        裁定（`.tmp/待裁清单_20261010.md` #428「修复取向需设计者裁」）。
        判据来源：issue #428 正文（其判据即「:86 注释自称『可复现』但该确定性
        只保证同插入顺序可复现」）。

        每 tick 每个实体**至多走一条**分支，优先级：
        ① 受控动作（F08 `set_command`，本 tick 消费一次）→ ② **物理驱动**
        （`attrs["physics"] == "dynamics"`，走环境层 `physics.simulate`）→
        ③ 自主行为（`_decide` + 沿注册路径平移）。未标记 `"dynamics"` 的实体
        **永不**进入 ② ⇒ 既有场景逐位不变（硬判据）。
        """
        n = int(n)
        if n < 0 or n > MAX_STEPS_PER_CALL:
            raise ValueError("step n=%d 超出范围 [0, %d]" % (n, MAX_STEPS_PER_CALL))
        acted = 0
        for _ in range(n):
            self.tick_count += 1
            actions = []
            # 碰撞层（本件）：**只在有碰撞箱时**记录 tick 起始位置（缺省零开销、
            # 不触碰任何既有状态 ⇒ 逐位不变）。
            if self._collision_boxes:
                self._tick_start_pos = {eid: e.pos for eid, e in self.entities.items()}
            for eid, e in self.entities.items():
                # F08：受控动作优先——登记的动作在本 tick 消费一次，且**不**走
                # 自主行为分支（不调用 _decide ⇒ 不消耗共享 _rng、不读目标规则）。
                cmd = self._commands.pop(eid, None)
                if cmd is not None:
                    record = self._apply_command(eid, e, cmd)
                    if record is not None:
                        actions.append(record)
                    continue
                # 环境层（本件）：**受物理驱动**的实体走动力学演化。
                # 优先级：登记动作 > 物理驱动 > 自主行为（三者互斥，各 tick 至多一条）。
                # `_is_dynamics` 对未标记实体恒 False ⇒ 既有路径**逐位不变**。
                if self._is_dynamics(e):
                    record = self._step_dynamics(eid, e)
                    if record is not None:
                        actions.append(record)
                    continue
                # 决策
                direction = self._decide(e)
                if direction == (0.0, 0.0, 0.0):
                    e.velocity = (0.0, 0.0, 0.0)
                    continue
                # 行动（按速度移动；剩余距离不足 speed 时按 min(speed, dist) 封顶）
                dist = self._approach_distance(e)
                step_len = e.speed if dist is None else min(e.speed, dist)
                nx = e.pos[0] + direction[0] * step_len
                nz = e.pos[2] + direction[2] * step_len
                # 世界响应（仅 x/z 边界约束；体素方块不参与碰撞——#249：原注释
                # 「不穿地」在本模块无对应实现，此处如实收窄为「仅边界约束」）
                # #347：非有限坐标（NaN/±inf）一律拒绝写入——clamp 表达式
                # `max(0.5, min(size-0.5, nan))` 会把 NaN 静默抬成边界值
                # （min(hi, nan)==hi），使 non_finite 不变量对 x/z 成死代码。
                if not (math.isfinite(nx) and math.isfinite(nz)):
                    warnings.warn(
                        "scene_simulator: 实体 %s 的下一位置非有限（%r, %r）"
                        "⇒ 拒绝写入" % (eid, nx, nz), RuntimeWarning, stacklevel=2)
                    e.velocity = (0.0, 0.0, 0.0)
                    continue
                # ⚠ 第 4 轮 #4：边界语义收归 `coord_contract.clamp_xz`（算式逐字一致）
                nx, nz = clamp_xz(nx, nz, self.world.size)
                old = e.pos
                e.pos = (round(nx, 2), old[1], round(nz, 2))
                # #193-B：velocity 此前只有声明、从无写入点（对外恒为 (0,0,0)）。
                # 口径＝**本 tick 的实际位移**（与 pos 同源、经钳制与取整），
                # 不是「单位方向 × speed」。
                e.velocity = (round(e.pos[0] - old[0], 2), 0.0,
                              round(e.pos[2] - old[2], 2))
                actions.append({"entity": eid, "category": e.category,
                                "behavior": e.behavior, "new_pos": e.pos})
                self._behavior_log.append({"tick": self.tick_count,
                                           "entity": eid, "behavior": e.behavior,
                                           "pos": e.pos})
            # 碰撞层（本件）：**带碰撞箱**的实体之间 ⇒ 先 sweep 拦穿透、再 resolve 推离。
            # 缺省（无碰撞箱）⇒ 整段不执行 ⇒ 既有行为逐位不变（硬判据）。
            if self._collision_boxes:
                self._resolve_collisions()
            acted = len(actions)
            # 场景演化记录（4D 占用序列）
            self._history.append({"tick": self.tick_count,
                                  "entities": self.entity_positions()})
        return {"tick": self.tick_count, "actions": acted}

    # ---- 场景状态 ----

    def entity_positions(self) -> Dict[str, Tuple[float, float, float]]:
        """全部实体位置（4D 时空占用）。"""
        return {eid: e.pos for eid, e in self.entities.items()}

    def scene_state(self) -> Dict:
        """场景状态：实体 + 行为 + 演化历史。"""
        return {
            "tick": self.tick_count,
            "size": self.world.size,
            "entities": {eid: e.to_dict() for eid, e in self.entities.items()},
            "paths": self.paths,
            "history_len": len(self._history),
            "behavior_actions": len(self._behavior_log),
        }

    def behavior_log(self, limit: int = 30) -> List[Dict]:
        """实体自主行为决策记录（可审计）。deque 不支持切片，故先转 list。"""
        return list(self._behavior_log)[-limit:]

    def evolution(self) -> List[Dict]:
        """场景演化历史（4D 占用序列）。deque 不支持切片，故返回 list 副本。"""
        return list(self._history)
