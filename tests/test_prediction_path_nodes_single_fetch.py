# -*- coding: utf-8 -*-
"""test_prediction_path_nodes_single_fetch.py · `_score_route` 路径节点「取一次」守卫

被优化处（`lingshu/world/prediction.py`）：
    `_score_route` 先后调 `_boundary_consistency(route["path"])` 与
    `_branch_diversity(route["path"])`，二者**各自** `for nid in path: get_node(nid)`
    ⇒ 同一路径的节点被 `get_node` 抓**两遍**（每次读整行 + 解析 JSON）= 2N 次。
    改为 `_path_nodes(path)` **取一次**（N 次）后下传。

本守卫四组：
  A 等价性：`_boundary_consistency` / `_branch_diversity` 的取值在
    「预取节点」与「逐个 get_node」两条路径上**逐位相同**（含空路径、缺节点、重复 id）；
  B 取数次数：`_score_route` 走预取 ⇒ `get_node` 调用数为 **N**（旧为 2N）；
  C 位置对齐：`_path_nodes` 长度恒等于 `len(path)`，缺节点位为 `None`（不去重）；
  D 回落（不误杀）：不传 `nodes` 时两方法仍可独立工作（等价旧行为）。

体例与 `tests/` 下其它 test_*.py 一致（`check()` + 退出码）。
"""
import sys
import os
import ast
import textwrap

sys.stdout.reconfigure(encoding='utf-8')
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

pass_n = fail_n = 0


def check(name, ok, detail=''):
    global pass_n, fail_n
    if ok:
        pass_n += 1
    else:
        fail_n += 1
    print(f'[{"✓" if ok else "✘"}] {name}{" — " + detail if detail else ""}')


PRED = os.path.join(_REPO, "lingshu", "world", "prediction.py")
SRC = open(PRED, encoding="utf-8").read()
TREE = ast.parse(SRC)


class _Node:
    def __init__(self, i, tags, content, sc):
        self.id, self.tags, self.content, self.semantic_coordinates = i, tags, content, sc


NODES = {
    "a": _Node("a", ["boundary"], "x", {"protocol": {"p1": 1, "p2": 1}}),
    "b": _Node("b", [], "边界说明", {"protocol": {"p3": 1}}),
    "c": _Node("c", [], "普通", {}),
    "d": _Node("d", [], "不确定", {"protocol": {"p4": 1, "p1": 1}}),
}


class _Store:
    def __init__(self):
        self.calls = 0

    def get_node(self, nid):
        self.calls += 1
        return NODES.get(nid)


class _Eng:
    def __init__(self):
        self.store = _Store()


# ---- 被测新实现（原样 exec 三个方法，保留真签名）----
_blocks = []
for _name in ("_path_nodes", "_boundary_consistency", "_branch_diversity"):
    _fn = next(n for n in ast.walk(TREE) if isinstance(n, ast.FunctionDef) and n.name == _name)
    _blocks.append(textwrap.indent(ast.get_source_segment(SRC, _fn), "    "))
_ns = {"List": list, "Dict": dict}
exec("class New:\n    def __init__(self, e):\n        self.engine = e\n" + "\n\n".join(_blocks), _ns)
New = _ns["New"]


class Old:
    """优化前的两个方法（逐节点 get_node）——等价性参照。"""

    def __init__(self, e):
        self.engine = e

    def _boundary_consistency(self, path):
        if not path:
            return 0.0
        ok = 0
        for nid in path:
            n = self.engine.store.get_node(nid)
            if n and ("boundary" in n.tags or "不确定" in n.content or "边界" in n.content):
                ok += 1
        return round(ok / len(path), 4)

    def _branch_diversity(self, path):
        dims = set()
        for nid in path:
            n = self.engine.store.get_node(nid)
            if n:
                sc = n.semantic_coordinates or {}
                for k in sc.get("protocol", {}):
                    dims.add(k)
        return round(min(1.0, len(dims) / 4.0), 4)


CASES = [["a", "b", "c", "d"], ["a", "c"], [], ["a", "missing"], ["a", "a", "b"], ["c"]]

print('— A 等价性（预取 vs 逐节点）—')
for path in CASES:
    _e1 = _Eng(); _o = Old(_e1)
    _obc, _obd = _o._boundary_consistency(path), _o._branch_diversity(path)
    _e2 = _Eng(); _n = New(_e2)
    _nodes = _n._path_nodes(path)
    _nbc = _n._boundary_consistency(path, _nodes)
    _nbd = _n._branch_diversity(path, _nodes)
    check(f'A {path} 取值逐位相同', _obc == _nbc and _obd == _nbd,
          f'bc {_obc}→{_nbc} bd {_obd}→{_nbd}')

print('— B 取数次数 —')
for path in CASES:
    _e1 = _Eng(); _o = Old(_e1)
    _o._boundary_consistency(path); _o._branch_diversity(path)
    _old_n = _e1.store.calls
    _e2 = _Eng(); _n = New(_e2)
    _pn = _n._path_nodes(path)
    _n._boundary_consistency(path, _pn); _n._branch_diversity(path, _pn)
    _new_n = _e2.store.calls
    check(f'B {path} get_node 次数 {_old_n} → {_new_n}（应 = len(path)={len(path)}）',
          _new_n == len(path) and (_old_n == 2 * len(path)), '')

print('— C 位置对齐 —')
for path in CASES:
    _n = New(_Eng())
    _pn = _n._path_nodes(path)
    check(f'C {path} 长度对齐且缺位为 None',
          len(_pn) == len(path) and all((_pn[i] is None) == (path[i] not in NODES)
                                        for i in range(len(path))), '')

print('— D 回落（不传 nodes）—')
_e = _Eng(); _n = New(_e)
check('D 两方法在 nodes=None 时仍等价旧行为',
      _n._boundary_consistency(["a", "b"]) == 1.0
      and _n._branch_diversity(["a", "b"]) == 0.75, '')
check('D 回落路径的 get_node 次数 = 2N', _e.store.calls == 4, f'实测 {_e.store.calls}')

print(f'\n=== 路径节点「取一次」守卫: {pass_n}/{pass_n + fail_n} 通过 ===')
sys.exit(0 if fail_n == 0 else 1)
