# -*- coding: utf-8 -*-
"""守卫 · `parse_anchor` 必须读回 `anchor_text()`/`describe()` 的**全部**产物，
含**负号坐标**（lingshu issue #730）

缺陷（修复前 `lingshu/world/vprim.py::parse_anchor`）
    写出侧 `anchor_text()` / `describe()` 用 `int(v)`（负值天然带 `-`），读回侧正则
    的坐标组却是 `\\d+`（纯无符号）——两侧各自定义格式、无共享 schema，docstring 承诺
    的「逆」只在非负域成立。实测（本机 Python 3.12.10，基线 29d2a87）：

        v = VPrim("person", (-10, -20, 30, 40), 0.80)
        v.anchor_text() → 'person@(-10,-20,30,40)'      → parse_anchor(...) is None
        v.describe()    → 'person@(-10,-20,30,40) 40x60 conf=0.80' → None
        bbox=(10, 20, 30, 40) → 读回正常（非负域「逆」成立，故旧守卫看不出来）
        端到端：core.world3d("add") 落库正文首行 `person@(-10,-20,30,40) 40x60 …`，
        随后 core.world3d("build") 报 observations=0（锚点全丢）——非负框同路径报 1。

判据来源（为什么判「改正则」而不是「在生成侧加非负断言」）
    · 谁生成负坐标：`Camera3D.project`（`lingshu/world/world3d.py`）是**线性**针孔投影
      `sx = screen_w/2 + f·x/z`，目标靠左/靠上时 `sx`/`sy` 直接为负——这正是
      `docs/second_third_batch_usage.md` 自带配方 `bbox=[px-20, py-30, px+20, py+30]`
      的形态（实测 truth=(-3.6, 2.6, 6.0) ⇒ px=-15.7 ⇒ bbox x1=-35.7）。
      `Camera3D.unproject` 对负像素同样精确可逆（其 docstring 声明
      `project(unproject(...)) == 原像素`，无定义域限制）。
    · 谁在读回侧收窄：`world3d.build_camera_observation` / `add_vprim` 只校验
      「四个**有限数**」（`_meta_number`），全仓**无**任何非负钳位（`git grep` 无
      `max(0, …)` 作用于 bbox）。⇒ 坐标本就是**有符号**量，非负是读回侧单方面的
      额外假设，不是生成侧的契约。故改正则（读回侧对齐写出侧），不在生成侧加断言。
    · 最小改动：只放宽四个坐标组为 `-?\\d+`；`-?` 是定长可选前缀，不引入新的回溯面
      （同文件既有 `test_parse_anchor_is_linear_on_adversarial_input` 继续钉住）。

定点变异自证（抽掉修复 ⇒ 必红）
    把坐标组回退成 `\\d+` ⇒ `test_roundtrip_property_over_sampled_boxes` 与
    `test_negative_coordinate_anchor_text_and_describe_roundtrip` 两条红；
    只回退**第一个**坐标组（部分回退）⇒ 同样两条红（x1 负号读不回）；
    把坐标组过修成 `-?[\\d.]+` ⇒ `test_rejects_non_anchor_junk` 红。

运行（仓根）：python -X utf8 -m pytest tests/test_issue730_vprim_negative_bbox_roundtrip.py -q --no-header
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lingshu.world.vprim import VPrim, parse_anchor  # noqa: E402

#: 采样 bbox（**整数**坐标——文本格式是 `int(v)`，非整数会被截断，属格式固有）。
#: 覆盖：全负 / 单轴负 / 跨零 / 全零 / 大负数 / 非负（正对照段）。
SAMPLED_BBOXES = (
    (10, 20, 30, 40),          # 非负（正对照）
    (0, 0, 0, 0),              # 全零
    (0, 0, 1, 1),              # 含零边界
    (-1, 5, 50, 60),           # 仅 x1 负（issue 实测形态）
    (5, -1, 50, 60),           # 仅 y1 负
    (5, 6, -1, 60),            # 仅 x2 负（退化框，符号仍须无损）
    (-10, -20, 30, 40),        # 全负（issue 实测形态）
    (-10, -20, -5, -3),        # 四角皆负
    (-800, -600, 800, 600),    # 大负（整幅溢出）
    (-1000000, 2000000, 3, 4), # 极端量级 + 跨零
    (-1, -1, 1, 1),            # 跨零小框
)

#: 采样类别（含非 `\\w` 字符——与 #174 的放宽面共存，防本次改动回退它）。
SAMPLED_CATEGORIES = ("person", "obj.2", "person-1", "a/b", "月亮")


def test_guard_inputs_are_not_empty():
    """硬性断言输入非空：防 `for` 空转成假绿（两条采样集都要有内容）。"""
    assert len(SAMPLED_BBOXES) >= 8, SAMPLED_BBOXES
    assert len(SAMPLED_CATEGORIES) >= 3, SAMPLED_CATEGORIES
    assert any(v < 0 for bb in SAMPLED_BBOXES for v in bb), "采样须含负坐标"
    assert any(all(v >= 0 for v in bb) for bb in SAMPLED_BBOXES), "采样须含非负正对照"


# ==================== ① 核心：往返性质 ====================

def test_negative_coordinate_anchor_text_and_describe_roundtrip():
    """★ 核心：负号坐标的 `anchor_text()` / `describe()` 必须被读回（原为 None）。"""
    for bb in ((-10, -20, 30, 40), (-1, 5, 50, 60), (-10, -20, -5, -3)):
        v = VPrim(category="person", bbox=bb, confidence=0.80)
        a = parse_anchor(v.anchor_text())
        assert a is not None, ("anchor_text 读不回", v.anchor_text())
        assert (a.category, tuple(a.bbox)) == ("person", bb), (v.anchor_text(), a)
        d = parse_anchor(v.describe())
        assert d is not None, ("describe 读不回", v.describe())
        assert (d.category, tuple(d.bbox)) == ("person", bb), (v.describe(), d)


def test_roundtrip_property_over_sampled_boxes():
    """★ 性质：采样集上 `parse_anchor(v.describe()).bbox == v.bbox` 全部成立。

    这是本件的**契约断言**（写出侧是契约：`int(v)` 产出即格式）。含负数、零、
    边界值与非负正对照段；任一样本失败即点名该样本。
    """
    checked = 0
    for bb in SAMPLED_BBOXES:
        for cat in SAMPLED_CATEGORIES:
            v = VPrim(category=cat, bbox=bb, confidence=0.52)
            d = parse_anchor(v.describe())
            assert d is not None, ("describe 读不回", cat, bb, v.describe())
            assert tuple(d.bbox) == bb, (cat, bb, v.describe(), d.bbox)
            assert d.category == cat, (cat, bb, v.describe(), d.category)
            a = parse_anchor(v.anchor_text())
            assert a is not None, ("anchor_text 读不回", cat, bb, v.anchor_text())
            assert (a.category, tuple(a.bbox)) == (cat, bb), (cat, bb, v.anchor_text())
            checked += 1
    assert checked == len(SAMPLED_BBOXES) * len(SAMPLED_CATEGORIES), checked


def test_roundtrip_keeps_confidence_from_describe():
    """`conf=` 一并读回（#174 面不得被本次改动回退）；两位小数内无损。"""
    for bb in ((-10, -20, 30, 40), (10, 20, 30, 40)):
        for conf in (0.5, 0.8, 0.99, 1.0):
            v = VPrim(category="person", bbox=bb, confidence=conf)
            d = parse_anchor(v.describe())
            assert d is not None, v.describe()
            assert abs(d.confidence - conf) <= 0.005, (bb, conf, d.confidence)


def test_negative_anchor_lands_in_scene_text_and_reads_back():
    """端到端：负坐标锚点经 `vprims_to_scene_text` 入文本后仍可读回（真实入口形态）。"""
    from lingshu.world.vprim import vprims_to_scene_text

    v = VPrim(category="person", bbox=(-10, -20, 30, 40), confidence=0.80)
    text = vprims_to_scene_text([v], scene=" 山景")
    token = text.split("；")[0]
    got = parse_anchor(token)
    assert got is not None, token
    assert (got.category, tuple(got.bbox)) == ("person", (-10, -20, 30, 40)), token


def test_negative_bbox_survives_core_world3d_roundtrip():
    """端到端（真实调用路径）：负坐标锚点落库后 `core.world3d("build")` 必须读回。

    缺陷形态下 `build` 报 `observations=0`（解析全丢）；非负框同路径报 1。
    这条把「能写不能读」直接钉在消费者读数上，不依赖内部函数名。
    """
    from lingshu.core.core import SpacetimeMemoryEngine
    from lingshu.world.world3d import Camera3D

    cam = Camera3D()
    results = {}
    for name, bb in (("neg", (-10, -20, 30, 40)), ("pos", (10, 20, 30, 40))):
        eng = SpacetimeMemoryEngine(":memory:")
        add = eng.world3d("add", {"category": "person", "bbox": list(bb),
                                  "camera": cam.to_meta(), "eid": "p1", "tick": 1,
                                  "screen_w": 800, "screen_h": 600, "timestamp": 1.0})
        assert add.get("status") == "ok", (name, add)
        built = eng.world3d("build", {"limit": 20})
        results[name] = built.get("observations")
    assert results["pos"] == 1, results           # 正对照：非负框读回 1
    assert results["neg"] == 1, results           # 负框必须同样读回 1


# ==================== ② 正对照（量具有效性） ====================

def test_positive_control_non_negative_roundtrip_unchanged():
    """正对照 1：非负 bbox 往返在修复前后都成立（证明守卫不是恒红）。"""
    v = VPrim(category="moon", bbox=(452, 84, 516, 148), confidence=0.52)
    for text in (v.anchor_text(), v.describe()):
        got = parse_anchor(text)
        assert got is not None, text
        assert (got.category, tuple(got.bbox)) == ("moon", (452, 84, 516, 148)), text


def test_positive_control_negative_probe_actually_detects_negative():
    """正对照 2：量具（`int()` 写出）确实产出了负号——证明「负读数」不是量具假象。"""
    v = VPrim(category="person", bbox=(-10, -20, 30, 40), confidence=0.80)
    assert v.anchor_text() == "person@(-10,-20,30,40)", v.anchor_text()
    assert v.describe().startswith("person@(-10,-20,30,40) "), v.describe()


# ==================== ③ 过修护栏（判别力反向） ====================

def test_rejects_non_anchor_junk():
    """过修护栏：放宽负号**不得**把垃圾串也吃成锚点。

    过修形态 `-?[\\d.]+` 会把 `obj@(1.5,2.5,3.5,4.5)`（浮点，非本格式）当锚点读回。
    """
    for junk in ("obj@(1.5,2.5,3.5,4.5)",      # 浮点坐标不在文本格式内
                 "obj@(1,2,3,4",                # 缺右括号
                 "obj@(1,2,3)",                 # 坐标数不足
                 "obj@(,2,3,4)",                # 空组
                 "obj@(-,2,3,4)",               # 裸负号
                 "obj@(1,2,3,4.5)"):            # 末组浮点
        assert parse_anchor(junk) is None, (junk, parse_anchor(junk))


def test_parse_anchor_still_linear_on_adversarial_input():
    """回归护栏：`-?` 是定长可选前缀，不得引入新的回溯（同文件 #374 面）。"""
    import time as _time

    for payload in ("a" * 40000, "@(" + "1" * 40000, "@(" + "-1" * 20000):
        t0 = _time.perf_counter()
        parse_anchor(payload)
        elapsed = _time.perf_counter() - t0
        assert elapsed < 1.0, (len(payload), elapsed)
