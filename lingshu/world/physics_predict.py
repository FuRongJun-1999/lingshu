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

判据来源
--------
    `lingshu.world.physics.classify`（六类模型 fit/predict/解析真值三件套）
    判据：`rel_gain=1.6` 简约性倍数、`abs_tol=1e-9` 数值地板、`gravity_tol=0.10` 重力同一性。
    阈值理由与敏感性读数见 `tests/test_physics_predict.py` 的 C 组。

断言组（抽掉修复 ⇒ 哪组红）
--------------------------
    `tests/test_physics_predict.py::test_B1_*`（分类正确性）
    `tests/test_physics_predict.py::test_E1_*`（回退即红：抽掉 a≡g ⇒ accelerated 被误判成 projectile）

运行命令
--------
    pytest tests/test_physics_predict.py -q
    # 接进预测链（opt-in，一行）：
    python3 -c "from lingshu.world import physics_predict as PP; PP.install()"
"""
from __future__ import annotations

from . import physics as PH

__all__ = ["PHYSICS_LAYER_SCHEMA", "classify_entity", "physics_candidates",
           "wrap_generate", "install"]

#: 本层返回体里诊断段的 schema
PHYSICS_LAYER_SCHEMA = "lingshu.physics-layer/1"

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


def classify_entity(history, eid, window=None, up=1, tick_hz=1.0):
    """把观测史 → `physics.classify` 的结果（含**选择依据**）。

    `history` 形如 `[{"tick":…, "entities":{eid:{"pos":{x,y,z}}}}…]`；
    取最近 `window` 个**真实观测**（缺帧跳过，不插值 —— 与 `motion.observed_positions` 同纪律）。

    ⚠ `tick_hz`：观测史的 `tick` 是**整数拍**，物理模块的时间是**秒**。不换算会把时间轴
    压缩/拉伸 `tick_hz` 倍 ⇒ 拟合出的加速度差 `tick_hz²` 倍（实测：漏这一步时
    `projectile` 的残差从 3.2e-16 涨到 20.4，抛体被误判成 accelerated）。这是必修的一步。
    """
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
        if isinstance(pos, dict):
            samples.append((float(tick), (float(pos["x"]), float(pos["y"]), float(pos["z"]))))
        else:
            samples.append((float(tick), (float(pos[0]), float(pos[1]), float(pos[2]))))
    hz = float(tick_hz) if tick_hz else 1.0
    if hz <= 0:
        raise ValueError("tick_hz 须 > 0")
    samples = [(t / hz, pos) for t, pos in samples]
    if len(samples) < MIN_SAMPLES:
        return None
    try:
        return PH.classify(samples, up=up)
    except PH.PhysicsError:
        return None


def physics_candidates(history, eid, horizon, up=1, window=None, tick_hz=1.0):
    """观测史 → 本层要提交的候选未来 `[(tick, (x,y,z)), …]`，或 `None`（不接管）。

    **强证据门槛**（两道，缺一不接管）：
        1. `kind != "static"`（静止交给既有 static 层）
        2. `residual_rms <= MAX_RMS`（拟合误差要小到"这个模型真的描述了它"）
    """
    got = classify_entity(history, eid, window=window, up=up, tick_hz=tick_hz)
    if got is None or got["kind"] == "static":
        return None
    if float(got["model"].get("residual_rms", 1e9)) > MAX_RMS:
        return None
    ticks = [rec["tick"] for rec in history if (rec.get("entities") or {}).get(eid) is not None]
    if not ticks:
        return None
    last = float(max(ticks))
    hz = float(tick_hz) if tick_hz else 1.0
    out = []
    for step in range(1, int(horizon) + 1):
        # 拍 → 秒：预测点取第 (last+step) 拍对应的**秒**时刻
        p, _v = PH.predict_at(got["model"], (last + step) / hz)
        out.append((int(last) + step, p))
    return {"kind": got["kind"], "reason": got["reason"], "candidates": out,
            "residuals": got["residuals"]}


def wrap_generate(inner):
    """把 `inner`（原有的 generate）包一层 → 新的 generate（**纯追加**）。"""

    def generate(self, horizon: int = 1, use_patterns: bool = True):
        out = inner(self, horizon, use_patterns)
        preds = out.get("predictions") or {}
        hist = getattr(self, "history", None) or getattr(self, "_history", None) or []
        hz = float(getattr(self, "tick_hz", 1.0) or 1.0)
        h = max(1, int(out.get("horizon") or horizon or 1))
        n_take = n_skip = 0
        for eid, rec in preds.items():
            if rec.get("mode") in INNER_MODES:
                n_skip += 1                      # 内层已接管 —— 不碰（主线不变量）
                continue
            if rec.get("mode") not in WEAK_MODES:
                n_skip += 1                      # 内层有确定语义（如 exact）—— 不抢
                continue
            got = physics_candidates(hist, eid, h, tick_hz=hz)
            if not got:
                continue
            # 与既有各层同一口径：`predicted` 是**一个点**（该 horizon 上的位置），
            # 不是 [(tick, pos)] 序列 —— bench 用 `math.dist(predicted, actual)` 核。
            rec["predicted"] = list(got["candidates"][-1][1])
            rec["mode"] = "physics:%s" % got["kind"]
            rec["physics_reason"] = got["reason"]
            n_take += 1
        out["physics"] = {"schema": PHYSICS_LAYER_SCHEMA, "taken": n_take,
                          "skipped_inner_modes": n_skip,
                          "model_kinds": list(PH.MODEL_KINDS)}
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
