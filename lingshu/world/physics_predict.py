# -*- coding: utf-8 -*-
"""physics_predict · 把 `lingshu.world.physics` 接进预测链的**纯追加包装层**

缺陷 / 修法
----------
缺陷：`world/` 里没有任何模块用 `physics.py` —— 物理模型只是"躺着的工具"，
      用户要的"世界模型**自己**做物理运动预测"不成立。
修法：本模块提供一个**纯追加**的 `generate()` 包装层（与该文件既有的
      `_generate_with_periodic` / static / smooth 各层同一手法）：
        · 只用 `physics.classify(观测史)` 判模型，**判据可复现**（残差比 + 加速度是否≈g）
        · 只在本层有**强证据**时改写 `rec["predicted"]`，并打 `mode="physics:<kind>"`
        · **尊重内层优先级**：`mode` 已是 `periodic` / `persist_guard` / `static` / `smooth`
          的实体**一律不碰**（与既有各层同一条不变量）
        · 返回体附 `physics` 诊断段；本层没生效时**逐字段与不装本层相同**

物理深化（任务 B · 本笔）
------------------------
`physics.py` 新增了三类**环境作用动力学**（`drag2` / `slope_friction` / `rolling`）
与环境上下文（`make_environment` / `simulate` / `environment_truth`）。本笔把两者接进预测链：

1. **新三类可判出**：`classify_entity` / `physics_candidates` 透传 `env`；
   `physics.classify` 只在**给了 `env`** 时让新三类参与竞争（可辨识性：纯位置观测下
   `slope_friction` 与 `rolling` 互相别名）。⇒ `env=None` 时新三类**不参与**，
   既有 6 类的判定结果**逐位不变**。
2. **环境感知预测**：调用方给出 env 时，候选点走 `physics.simulate(model, env, t)`
   （能闭式就闭式；闭式不可得且**未显式给 `dt`** ⇒ 本层**不接管**，不静默选步长）；
   没给 env 时仍走既有的 `physics.predict_at(model, t)` **同一条路径**
   ⇒ **不给环境 = 现状逐字段一致**（硬判据，守卫 G2 贴对照）。
3. **环境从哪来（opt-in，不改 `world_model`）**：`resolve_env(obj)` 依次看
   `obj.physics_env` → `obj.environment`；两者都缺省 ⇒ `None` ⇒ 与不装本层一致。
   显式调用 `physics_candidates(..., env=…)` 亦可直接驱动。

断言组（抽掉修复 ⇒ 哪组红）
-------------------------
    `tests/test_physics_predict.py::test_B1_*`（分类正确性）
    `tests/test_physics_predict.py::test_E1_*`（回退即红：抽掉 a≡g ⇒ accelerated 被误判成 projectile）
    `tests/test_physics_predict.py::test_G1_*`（新三类可判出）
    `tests/test_physics_predict.py::test_G2_*`（不给 env = 现状逐字段一致）
    `tests/test_physics_predict.py::test_G3_*`（给 env ⇒ 环境感知生效）
    `tests/test_physics_predict.py::test_G4_*`（定点变异：回退即红）

运行命令
--------
    pytest tests/test_physics_predict.py -q
    # 接进预测链（opt-in，一行）：
    python3 -c "from lingshu.world import physics_predict as PP; PP.install()"
    # 环境感知（opt-in，一行）：
    python3 -c "from lingshu.world import physics, physics_predict as PP; \
PP.install(); wm.physics_env = physics.make_environment(terrain={'slope_deg':20,'mu_k':0.1})"
"""
from __future__ import annotations

import math

from . import physics as PH

__all__ = ["PHYSICS_LAYER_SCHEMA", "ENV_ATTRS", "resolve_env", "classify_entity",
           "physics_candidates", "wrap_generate", "install"]


def _require_positive_finite(val, name: str) -> float:
    """正有限数值闸：拒绝 bool / 非数值 / NaN / ±inf / <= 0，统一抛 ValueError（#196）。"""
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError("%s 须为正有限数值，收到 %r" % (name, val))
    try:
        f = float(val)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("%s 须为正有限数值，收到 %r" % (name, val)) from exc
    if not math.isfinite(f) or f <= 0.0:
        raise ValueError("%s 须为正有限数值（> 0），收到 %r" % (name, val))
    return f


def _require_positive_int(val, name: str) -> int:
    """正整数闸：拒绝 bool / 浮点 / NaN / ±inf / <= 0，统一抛 ValueError（#196）。"""
    if isinstance(val, bool) or not isinstance(val, int) or val < 1:
        raise ValueError("%s 须为 >= 1 的整数，收到 %r" % (name, val))
    return int(val)


def _validate_predict_opts(up, window) -> None:
    """校验共用的轴索引与观测窗口参数，拒绝 bool / 非整数。"""
    if isinstance(up, bool) or not isinstance(up, int) or up not in (0, 1, 2):
        raise ValueError("up 须为轴索引 0/1/2，收到 %r" % (up,))
    if window is not None and (isinstance(window, bool) or not isinstance(window, int)):
        raise ValueError("window 须为整数或 None，收到 %r" % (window,))

#: 本层返回体里诊断段的 schema
PHYSICS_LAYER_SCHEMA = "lingshu.physics-layer/1"

#: **环境感知的 opt-in 挂载点**：调用方把 `physics.make_environment(...)` 的返回挂在模型实例的
#: 这两个属性之一上，预测链即自动走环境感知路径。**两者都缺省 ⇒ `env is None` ⇒
#: 与不装本层逐字段一致**（硬判据）。刻意不去改 `world_model`：那是别人的写面，
#: 且"环境"属于**场景输入**而非世界模型的固有状态。
ENV_ATTRS = ("physics_env", "environment")


def resolve_env(obj):
    """从模型实例上解析**环境上下文**（opt-in）：依次看 `ENV_ATTRS`。

    返回 `make_environment(...)` 的返回，或 `None`（= 调用方没给环境）。

    ⚠ `None` 与 `physics.make_environment()`（**真空平地**）**不是**同一件事：
    后者会让 `classify` 带上 `env_used`、让 `simulate` 走环境路径（虽然数值同源）；
    本函数**不替调用方造环境** —— 没挂就是没给。
    """
    for name in ENV_ATTRS:
        got = getattr(obj, name, None)
        if got is not None:
            return got
    return None

#: 内层已接管的 mode —— 本层**一律不碰**（与 `_generate_with_periodic` 同一不变量）
INNER_MODES = ("periodic", "persist_guard", "static", "smooth")

#: **只接管"弱兜底"模式**：`exact` / `bounded_*` 里内层已经有确定语义的模式一律不抢。
#: 依据（实测，不是猜）：不设这条时 `unified_occlusion` 的中位误差从 0.0000 退到 0.6000
#: （内层本来逐点为 0，被本层一个**样本内**拟合改写了外推）—— 教训：**样本内残差小
#: 不等于外推更准**，所以本层只在"内层自己都承认不确定"的模式上出手。
WEAK_MODES = ("bounded_stochastic", "chase_stochastic", "bounded_noisy", "unknown", None)

#: 本层要"强证据"才接管（比 `physics.classify` 的默认更严：预测改写的代价更高）
MIN_SAMPLES = 6
MAX_RMS = 0.25          # 拟合 rms 上限（米；与 bench 的 `fixed_threshold=0.5` 同量级的一半）


def classify_entity(history, eid, window=None, up=1, tick_hz=1.0, env=None):
    """把观测史 → `physics.classify` 的结果（含**选择依据**）。

    `history` 形如 `[{"tick":…, "entities":{eid:{"pos":{x,y,z}}}}…]`；
    取最近 `window` 个**真实观测**（缺帧跳过，不插值 —— 与 `motion.observed_positions` 同纪律）。

    ⚠ `tick_hz`：观测史的 `tick` 是**整数拍**，物理模块的时间是**秒**。不换算会把时间轴
    压缩/拉伸 `tick_hz` 倍 ⇒ 拟合出的加速度差 `tick_hz²` 倍（实测：漏这一步时
    `projectile` 的残差从 3.2e-16 涨到 20.4，抛体被误判成 accelerated）。这是必修的一步。

    `env`（本笔新增，**默认 None**）：`physics.make_environment(...)` 的返回。给了它
    `physics.classify` 才让 `drag2` / `slope_friction` / `rolling` 参与竞争
    （纯位置观测下后两者互相别名，故**必须**给 terrain 才可辨识）；
    **不给 ⇒ 新三类不参与 ⇒ 既有 6 类的判定逐位不变**（硬判据）。
    """
    hz = _require_positive_finite(tick_hz, "tick_hz")
    _validate_predict_opts(up, window)
    from . import motion as MO
    pts = MO.observed_positions(history, eid, window)
    if not pts:
        return None
    samples = []
    for tick, pos in pts:
        # `motion.observed_positions(nested=False)` 给的是"实体值"本身；本仓 history 里它
        # 有时是裸三元组、有时是 `{"pos": {...}}` 包一层 —— 两种都收（不猜、直接判形）。
        if isinstance(pos, dict) and "pos" in pos:
            pos = pos["pos"]
        try:
            coords = (pos["x"], pos["y"], pos["z"]) if isinstance(pos, dict) else (pos[0], pos[1], pos[2])
        except (KeyError, IndexError, TypeError):
            return None
        if any(isinstance(c, bool) for c in coords):
            return None
        try:
            vec = (float(coords[0]), float(coords[1]), float(coords[2]))
            t_val = float(tick)
        except (TypeError, ValueError, OverflowError):
            return None
        if isinstance(tick, bool) or not math.isfinite(t_val) or not all(math.isfinite(c) for c in vec):
            return None
        samples.append((t_val, vec))
    samples = [(t / hz, pos) for t, pos in samples]
    if len(samples) < MIN_SAMPLES:
        return None
    try:
        return PH.classify(samples, up=up, env=env)
    except PH.PhysicsError:
        return None


def physics_candidates(history, eid, horizon, up=1, window=None, tick_hz=1.0,
                       env=None, dt=None):
    """观测史 → 本层要提交的候选未来 `[(tick, (x,y,z)), …]`，或 `None`（不接管）。

    **强证据门槛**（两道，缺一不接管）：
        1. `kind != "static"`（静止交给既有 static 层）
        2. `residual_rms <= MAX_RMS`（拟合误差要小到"这个模型真的描述了它"）

    **环境感知**（本笔新增）：
      · `env is None`（缺省）⇒ 候选点走既有 `physics.predict_at(model, t)` ——
        **与改动前同一条路径 ⇒ 逐字段一致**（硬判据，守卫 G2 贴对照）；
      · `env` 给了 ⇒ 候选点走 `physics.simulate(model, env, t, dt=dt)`
        （能闭式就闭式；`dt` 只在闭式不可得时才用得上）。
      · ⚠ **闭式不可得且 `dt is None` ⇒ 本层不接管**（返回 `None`）——
        不替调用方静默选步长（`physics._integrate_env` 的 fail-closed 纪律）。
        要给就显式给 `dt`（或让模型实例带 `physics_dt`）。
    """
    h = _require_positive_int(horizon, "horizon")
    hz = _require_positive_finite(tick_hz, "tick_hz")
    _validate_predict_opts(up, window)
    if dt is not None:
        dt = _require_positive_finite(dt, "dt")
    got = classify_entity(history, eid, window=window, up=up, tick_hz=hz, env=env)
    if got is None or got["kind"] == "static":
        return None
    raw_rms = got["model"].get("residual_rms", 1e9)
    if isinstance(raw_rms, bool):
        return None
    try:
        rms = float(raw_rms)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(rms) or rms > MAX_RMS:
        return None
    ticks = [rec["tick"] for rec in history if (rec.get("entities") or {}).get(eid) is not None]
    if not ticks:
        return None
    last = float(max(ticks))
    out = []
    sim_last = None
    for step in range(1, h + 1):
        # 拍 → 秒：预测点取第 (last+step) 拍对应的**秒**时刻
        t = (last + step) / hz
        if env is None:
            p, _v = PH.predict_at(got["model"], t)
        else:
            try:
                sim_last = PH.simulate(got["model"], env, t, dt=dt)
            except PH.PhysicsError:
                # 闭式不可得且未显式给 dt ⇒ 不接管（不静默选步长；见 docstring）
                return None
            p = sim_last["pos"]
        if not all(math.isfinite(float(c)) for c in p):
            return None
        out.append((int(last) + step, p))
    res = {"kind": got["kind"], "reason": got["reason"], "candidates": out,
           "residuals": got["residuals"]}
    if env is not None:
        # 只在**真的用了环境**时才加这几个键 ⇒ 不给 env 的返回体键集与改动前一致
        res["env_aware"] = True
        res["env_used"] = (sim_last or {}).get("env_used")
        res["simulated"] = {"closed_form": bool((sim_last or {}).get("closed_form")),
                            "integrated": bool((sim_last or {}).get("integrated")),
                            "steps": int((sim_last or {}).get("steps") or 0),
                            "reason": (sim_last or {}).get("reason")}
        res["env_evidence"] = _env_evidence(got, env)
    return res


def _env_evidence(got, env):
    """环境作用类的**可读依据**（"该用哪类"）—— 逐类残差与该类是否被 env 允许。

    为什么单独给这一段：`classify` 的 `reason` 是一句话，而"新三类为什么没被选中"
    需要**逐类数字**才能复核（本节是任务要求的"候选集要有可读依据"）。

    只看 env 的 `medium` / `terrain` **是否在场**（`make_environment` 的返回一定带这两个键），
    不反向依赖 `physics` 的私有归一函数 —— 归一由 `physics.simulate` 在内部自己完成，
    那边的结果已由 `res["env_used"]` 回报。
    """
    med = env.get("medium") if isinstance(env, dict) else None
    ter = env.get("terrain") if isinstance(env, dict) else None
    allowed = []
    if med is not None:
        allowed.append("drag2")
    if ter is not None:
        allowed += ["slope_friction", "rolling"]
    res = got.get("residuals") or {}
    return {"picked": got["kind"],
            "env_allowed_kinds": allowed,
            "env_context": ("medium+terrain" if (med is not None and ter is not None)
                            else ("medium" if med is not None
                                  else ("terrain" if ter is not None else "vacuum_flat"))),
            "residuals": {k: (None if v is None else round(float(v), 12))
                          for k, v in sorted(res.items())}}


def wrap_generate(inner):
    """把 `inner`（原有的 generate）包一层 → 新的 generate（**纯追加**）。"""

    def generate(self, horizon: int = 1, use_patterns: bool = True):
        out = inner(self, horizon, use_patterns)
        preds = out.get("predictions") or {}
        hist = getattr(self, "history", None) or getattr(self, "_history", None) or []
        raw_hz = getattr(self, "tick_hz", 1.0)
        hz = 1.0 if raw_hz is None else _require_positive_finite(raw_hz, "tick_hz")
        raw_h = out.get("horizon")
        if raw_h is None:
            raw_h = 1 if horizon is None else horizon
        if isinstance(raw_h, bool) or not isinstance(raw_h, int):
            raise ValueError("horizon 须为整数，收到 %r" % (raw_h,))
        h = max(1, raw_h)
        # 环境感知是 **opt-in**：实例上挂了 `physics_env` / `environment` 才启用；
        # 没挂 ⇒ env=None ⇒ 走既有 `predict_at` 同一条路径（逐字段一致，硬判据）。
        env = resolve_env(self)
        dt = getattr(self, "physics_dt", None)
        if dt is not None:
            dt = _require_positive_finite(dt, "physics_dt")
        n_take = n_skip = 0
        env_used = None
        for eid, rec in preds.items():
            if rec.get("mode") in INNER_MODES:
                n_skip += 1                      # 内层已接管 —— 不碰（主线不变量）
                continue
            if rec.get("mode") not in WEAK_MODES:
                n_skip += 1                      # 内层有确定语义（如 exact）—— 不抢
                continue
            got = physics_candidates(hist, eid, h, tick_hz=hz, env=env, dt=dt)
            if not got:
                continue
            # 与既有各层同一口径：`predicted` 是**一个点**（该 horizon 上的位置），
            # 不是 [(tick, pos)] 序列 —— bench 用 `math.dist(predicted, actual)` 核。
            rec["predicted"] = list(got["candidates"][-1][1])
            rec["mode"] = "physics:%s" % got["kind"]
            rec["physics_reason"] = got["reason"]
            if env is not None:
                # 环境感知的可读读数：用了哪个环境、闭式还是积分、依据是什么
                rec["physics_env"] = got.get("env_evidence")
                rec["physics_simulated"] = got.get("simulated")
                env_used = got.get("env_used") or env_used
            n_take += 1
        diag = {"schema": PHYSICS_LAYER_SCHEMA, "taken": n_take,
                "skipped_inner_modes": n_skip,
                "model_kinds": list(PH.MODEL_KINDS)}
        if env is not None:
            # 只在**真的给了环境**时加这几个键 ⇒ 不给 env 的返回体键集与改动前逐字段一致
            diag["env_aware"] = True
            diag["env_source"] = [n for n in ENV_ATTRS if getattr(self, n, None) is not None]
            diag["env_used"] = env_used
        out["physics"] = diag
        return out

    generate.__name__ = "generate_with_physmodel"
    return generate


def compose(factory):
    """把本层组合进给定 factory 的 `generate`（**幂等**，按 `is`/`__name__` 自查）。

    返回 `True` 表示这次真的加了一层；`False` 表示已经装过（重复调用只加一层）。
    """
    cur = factory["generate"]
    if getattr(cur, "__name__", "") == "generate_with_physmodel":
        return False
    factory["generate"] = wrap_generate(cur)
    return True


def active_generate_is_physics():
    """**直接身份核对**：当前生效的 `generate` 是不是本层（D-16 要求的强断言，不靠赋值次序）。"""
    from . import world_model as WM
    return (getattr(WM.UnifiedWorldModel.generate, "__name__", "") == "generate_with_physmodel"
            and WM.UnifiedWorldModel.generate is WM._WM_FACTORY["generate"])


def install(*, verify=True):
    """把本层接进 `UnifiedWorldModel` 的唯一绑定入口（**opt-in，一行**）。

    做法：把 `_WM_FACTORY["generate"]` 换成"原层 + 本层"，再走
    `install_prediction_wrappers(verify=verify)` 重新绑定 —— **不新增第二处绑定语句**
    （与 P3 坐标契约同一条纪律）。幂等：重复调用只加一层。
    """
    from . import world_model as WM
    # 刚改过 factory ⇒ 已绑定的属性必然是"旧的" ⇒ 这里必须 verify=False 再绑一次；
    # 绑完再跑身份核对（D-16：不靠"文件里的赋值次序"）。
    if compose(WM._WM_FACTORY):
        WM.install_prediction_wrappers(verify=False)
    if verify:
        WM.assert_wrapper_pair()
        if not active_generate_is_physics():
            raise RuntimeError("物理预测层未生效（身份核对失败）—— 见 LEAD-FINDINGS D-16")
    return WM.active_wrapper_pair()
