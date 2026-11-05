"""#942 / F23-23 · rust_bridge.py "last" 命名守卫。

缺陷：`py_last` / `rust_last` / `d_last` / `ref_last` 名不副实。
本仓实测（base 50ee03b 与 main 8f4f23ff 两版均成立）：两处执行器都在**施加本步更新之前**
记读数（`d = loss(vec); curve.append(d)` 之后才改 `vec`），所以 `curve[-1]` 是
「最后一次更新前」的读数，不是训练终值。

判据：
① 运行时口径：五个返回点的字典字面量键（编译进 `co_consts`）里不再有旧名、必须是新名；
② 源码口径：目标文件里旧名作为独立字符串键零残留；
③ 命名依据自证：两处取样点仍是「先 append、后更新」。

变异：把任一 `*_before_last_update` 改回旧名 ⇒ ①② 红。
"""

import pathlib

import lingshu.nn.rust_bridge as rb

SRC = pathlib.Path(rb.__file__)

OLD_KEYS = ("py_last", "rust_last", "d_last", "ref_last")
NEW_KEYS = (
    "py_before_last_update",
    "rust_before_last_update",
    "d_before_last_update",
    "ref_before_last_update",
)

# 五个返回点与各自应有的新键（出现次数按站点计）
SITES = {
    "equivalence_par": ("py_before_last_update", "rust_before_last_update"),
    "determinism_check": ("ref_before_last_update",),
    "bench_threads": ("d_before_last_update",),
    "equivalence_check": ("py_before_last_update", "rust_before_last_update"),
    "bench_rust": ("d_before_last_update",),
}


def _consts(fn):
    """该函数及其嵌套 code 对象的字符串常量（含元组内的键名；字典字面量键在此）。"""
    out = set()
    todo = [fn.__code__]
    while todo:
        code = todo.pop()
        for c in code.co_consts:
            if isinstance(c, str):
                out.add(c)
            elif isinstance(c, (tuple, frozenset)):
                out.update(x for x in c if isinstance(x, str))
            elif hasattr(c, "co_consts"):
                todo.append(c)
    return out


def test_runtime_dict_keys_are_renamed():
    for name, want in SITES.items():
        consts = _consts(getattr(rb, name))
        for key in want:
            assert key in consts, "%s 的返回键里缺 %r；实际键: %r" % (
                name, key, sorted(k for k in consts if k.endswith("last_update") or k.endswith("_last")))
        for key in OLD_KEYS:
            assert key not in consts, "%s 仍用旧键 %r" % (name, key)


def test_source_has_no_old_keys():
    src = SRC.read_text(encoding="utf-8")
    for key in OLD_KEYS:
        assert '"%s"' % key not in src, "源码仍残留旧键 %r" % (key,)


def test_sample_is_taken_before_update():
    """命名依据本身也要成立：curve 必须先 append、后改 vec（两处执行器）。"""
    src = SRC.read_text(encoding="utf-8")
    anchor = "d = loss(vec)"
    n = 0
    for block in src.split(anchor)[1:]:
        head = block.lstrip()
        head = head.split("\n", 1)[1] if "\n" in head else ""
        if "curve.append(d)" in head.split("for ")[0]:
            n += 1
    assert n == 2, "取样点形状变了，需重新核对命名依据：%d" % n
