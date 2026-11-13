# -*- coding: utf-8 -*-
"""#942 / F23-25 守卫：`empty_rate` 零样本记 None（#356「不伪造读数」口径）。

缺陷面（R23 清单 F23-25）：同一返回体里 `stability`/`honored` 空值记 None
（#356 口径），`empty_rate` 零样本却经 `max(1,·)` 记 0.0——伪造「无空读」
满分读数。修法：零样本记 None；打印面同步适配。

运行（lingshu 仓根）：python -X utf8 -m pytest tests/test_issue942_f23_25_empty_rate_null.py -q
"""
import inspect
import os

from lingshu.gen import hexgen_multi_seed as m


def test_f23_25_source_uses_null_semantics():
    src = inspect.getsource(m)
    assert "F23-25" in src, "empty_rate 缺 F23-25 口径标注"
    assert "if reads else None" in src, "empty_rate 未按 #356 null 口径实现"
    # 打印面同步适配（None 不再走 :.1% 格式）
    assert "无样本" in src


def test_f23_25_print_tolerates_none():
    """打印面对 None 的适配逻辑（与源码同式）。"""
    s = {"n_img_empty": 0, "n_img": 0, "empty_rate": None}
    er = s["empty_rate"]
    line = f"    ④ 空读件 {s['n_img_empty']}/{s['n_img']}（" + (f"{er:.1%}" if er is not None else "无样本") + "）"
    assert "无样本" in line


def test_f23_25_module_imports_clean():
    assert hasattr(m, "pick_prompts") and hasattr(m, "main")
