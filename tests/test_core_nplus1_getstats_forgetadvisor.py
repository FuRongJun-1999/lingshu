# -*- coding: utf-8 -*-
"""test_core_nplus1_getstats_forgetadvisor.py · 两处 N+1 优化的**等价性 + 查询数**守卫

被优化的两处（`lingshu/core/core.py`）：

  ① `LayeredStore.get_stats`：旧实现按 `MemoryLayer` **逐层** `SELECT COUNT(*)`
     （N 次），再各发一条查边（2 次）⇒ **N+2** 次往返。改为节点侧一趟
     `GROUP BY layer`、边侧一趟条件聚合 ⇒ **2** 次。

  ② `SpacetimeMemoryEngine.forget_advisor`：旧实现每轮除首查外还有
     · `self.store.get_node(nid)`（赋给局部量 `stale`，**该量从未被读取**）
     · `SELECT access_count FROM nodes WHERE id=?`（N 次往返）
     · 首查多取未用的 `content` 大文本列
     ⇒ 收成「首查 + ≤N×UPDATE」。

本守卫四组：
  A 等价性（`get_stats`）：空库 / 零节点层 / 每层都有 / 边全 verified /
    边混合含 NULL / 边全 NULL / 混入历史未知层值 —— **新旧取值必须逐位相同**。
  B 查询数（`get_stats`）：新实现发出的 SQL 条数**不随层数增长**（= 2）。
  C 静态面（`forget_advisor`）：AST 断言循环体内**不再**出现
    `get_node` 调用与 `SELECT access_count`；且首查不取 `content`。
  D 对照（不误杀）：`forget_advisor` 的判定口径未动 —— 归档只降不升、
    `no_forget`/`archived` 跳过、`acc == 0` 的访问信号仍生效。

体例与 `tests/` 下其它 test_*.py 一致（`check()` + 退出码）。
"""
import sys
import os
import ast
import json
import time
import sqlite3

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


CORE = os.path.join(_REPO, "lingshu", "core", "core.py")
SRC = open(CORE, encoding="utf-8").read()

# ---------------------------------------------------------------- A/B get_stats
#: 5 层（与 LayeredStore 的 MemoryLayer 同名同序）
LAYERS = ("anchor", "structure", "knowledge", "contextual", "self")

#: 从被测源码里抽 `get_stats` 的实现体，独立 exec（不导入整个 lingshu 包）
_tree = ast.parse(SRC)
_gs = None
for _n in ast.walk(_tree):
    if isinstance(_n, ast.FunctionDef) and _n.name == "get_stats":
        _gs = _n
        break
check('A0 已定位 get_stats', _gs is not None, f'{CORE}')
if _gs is None:
    print(f'\n=== N+1 优化守卫: {pass_n}/{pass_n + fail_n} 通过 ===')
    sys.exit(1)

_gs_src = ast.get_source_segment(SRC, _gs)


import enum


class MemoryLayer(enum.Enum):             # noqa: N801 —— 故意同名，供被测代码引用
    """`get_stats` 依赖的 `MemoryLayer` 的最小替身（成员名/值与真仓同序同值）。"""

    ANCHOR = "anchor"
    STRUCTURE = "structure"
    KNOWLEDGE = "knowledge"
    CONTEXTUAL = "contextual"
    SELF = "self"


# 抽出实现体，独立 exec（不导入整个 lingshu 包）
_ns = {"Dict": dict, "MemoryLayer": MemoryLayer}
exec("class _S:\n" + "\n".join("    " + l for l in _gs_src.splitlines()), _ns)
get_stats_new = _ns["_S"].get_stats

#: 与替身枚举同序同值
LAYERS = ("anchor", "structure", "knowledge", "contextual", "self")

SCHEMA = ("CREATE TABLE nodes (id TEXT PRIMARY KEY, layer TEXT);"
          "CREATE TABLE edges (id TEXT PRIMARY KEY, verified INTEGER);")


def mk(nodes, edges):
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA)
    conn.executemany("INSERT INTO nodes VALUES (?,?)", nodes)
    conn.executemany("INSERT INTO edges VALUES (?,?)", edges)
    return _Shim(conn)


class _Shim:
    """`get_stats` 走 `self.conn.cursor()`，故给它一个带 `conn` 的薄壳。"""

    def __init__(self, conn):
        self.conn = conn


def get_stats_old(shim):
    """旧实现（逐层 COUNT + 两条查边）——等价性判据的参照。"""
    c = shim.conn.cursor()
    stats = {}
    for lv in LAYERS:
        c.execute("SELECT COUNT(*) FROM nodes WHERE layer=?", (lv,))
        stats[f"{lv}_nodes"] = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM edges")
    stats["total_edges"] = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM edges WHERE verified=1")
    stats["verified_edges"] = c.fetchone()[0]
    return stats


CASES = [
    ("空库", [], []),
    ("只有零节点层", [("n1", "self")], []),
    ("每层都有", [("a", "anchor"), ("b", "anchor"), ("c", "structure"),
                  ("d", "knowledge"), ("e", "contextual"), ("f", "self"),
                  ("g", "self")], []),
    ("边：全 verified", [("a", "anchor")], [("e1", 1), ("e2", 1)]),
    ("边：混合 + NULL", [("a", "anchor")], [("e1", 1), ("e2", 0), ("e3", None)]),
    ("边：全 NULL", [("a", "anchor")], [("e1", None)]),
    ("混入历史/未知层值", [("a", "anchor"), ("x", "legacy_layer"), ("y", "")], [("e1", 1)]),
]

print('— A 等价性（get_stats）—')
for name, nodes, edges in CASES:
    _o = get_stats_old(mk(nodes, edges))
    _n = get_stats_new(mk(nodes, edges))
    check(f'A {name}', _o == _n, '' if _o == _n else f'旧{_o} ≠ 新{_n}')

print('— B 查询数（get_stats）—')


class _Counting:
    """包一层 cursor，统计 `execute` 次数。"""

    def __init__(self, cur, owner):
        self._c, self._o = cur, owner

    def execute(self, *a, **k):
        self._o.n += 1
        return self._c.execute(*a, **k)

    def fetchone(self):
        return self._c.fetchone()

    def fetchall(self):
        return self._c.fetchall()


class _CountConn:
    """包一层 connection，`cursor()` 返回计数游标。"""

    def __init__(self, conn):
        self._conn, self.n = conn, 0

    def cursor(self):
        return _Counting(self._conn.cursor(), self)

    def executemany(self, *a, **k):
        return self._conn.executemany(*a, **k)

    def executescript(self, *a, **k):
        return self._conn.executescript(*a, **k)


_cc = _CountConn(mk([("a", "anchor")], [("e1", 1)]).conn)
get_stats_new(_Shim(_cc))
check('B 新实现 SQL 条数 == 2（与层数无关）', _cc.n == 2, f'实测 {_cc.n} 条')
check('B 旧实现 SQL 条数 == N+2', len(LAYERS) + 2 == 7, '')

# --------------------------------------------------------- C/D forget_advisor
print('— C 静态面（forget_advisor）—')
_fa = None
for _n in ast.walk(_tree):
    if isinstance(_n, ast.FunctionDef) and _n.name == "forget_advisor":
        _fa = _n
        break
check('C0 已定位 forget_advisor', _fa is not None, '')

if _fa is not None:
    _inner_calls, _inner_selects = [], []
    for _loop in ast.walk(_fa):
        if not isinstance(_loop, (ast.For, ast.While)):
            continue
        for _x in ast.walk(_loop):
            if isinstance(_x, ast.Call) and isinstance(_x.func, ast.Attribute):
                if _x.func.attr == "get_node":
                    _inner_calls.append(_x.lineno)
            if isinstance(_x, ast.Constant) and isinstance(_x.value, str) \
                    and "access_count" in _x.value and "SELECT" in _x.value:
                _inner_selects.append(_x.lineno)
    check('C 循环体内无 get_node 调用（死变量 stale 已删）', not _inner_calls,
          f'残留于 L{_inner_calls}' if _inner_calls else '')
    check('C 循环体内无逐节点 SELECT access_count', not _inner_selects,
          f'残留于 L{_inner_selects}' if _inner_selects else '')

    _first_select = [c.value for c in ast.walk(_fa)
                     if isinstance(c, ast.Constant) and isinstance(c.value, str)
                     and "FROM nodes" in c.value]
    _has_content = any("content" in s for s in _first_select)
    check('C 首查不再多取未用的 content 列', not _has_content,
          f'仍取 content：{_first_select[:1]}' if _has_content else '')
    check('C 首查已带出 access_count', any("access_count" in s for s in _first_select), '')

    # D 对照：判定口径的关键字面量仍在
    _fa_src = ast.get_source_segment(SRC, _fa) or ""
    check('D 归档「只降不升」仍在（min(imp_v, archived_imp)）',
          "min(imp_v, archived_imp)" in _fa_src, '')
    check('D `no_forget`/`archived` 跳过仍在', '"no_forget" in tags' in _fa_src, '')
    check('D `acc == 0` 访问信号仍在',
          "acc == 0" in _fa_src and "signal += 1" in _fa_src, '')
    check('D `imp is None` 回默认 0.5（不用 or）仍在',
          "0.5 if imp is None else imp" in _fa_src, '')

# ------------------------------------------------- E recalc_structural_importance
print('— E 静态面（recalc_structural_importance 的 structural_boost 不再逐节点回查）—')
_rs = None
for _n in ast.walk(_tree):
    if isinstance(_n, ast.FunctionDef) and _n.name == "recalc_structural_importance":
        _rs = _n
        break
check('E0 已定位 recalc_structural_importance', _rs is not None, '')

if _rs is not None:
    _loop_reads = []
    for _loop in ast.walk(_rs):
        if not isinstance(_loop, (ast.For, ast.While)):
            continue
        for _x in ast.walk(_loop):
            if isinstance(_x, ast.Call) and isinstance(_x.func, ast.Attribute) \
                    and _x.func.attr == "_read_state_attr":
                _loop_reads.append(_x.lineno)
    check('E 循环体内无 `_read_state_attr` 逐节点回查', not _loop_reads,
          f'残留于 L{_loop_reads}' if _loop_reads else '')

    _sel = [c.value for c in ast.walk(_rs)
            if isinstance(c, ast.Constant) and isinstance(c.value, str)
            and "FROM nodes" in c.value]
    check('E 主查询已带出 state_attributes',
          any("state_attributes" in s for s in _sel),
          f'主查询：{_sel[:2]}' if _sel else '未找到主查询')
    check('E 主查询仍取 id/importance（读法未变）',
          any("id, importance" in s for s in _sel), '')

    # 三层解包必须同步（rows 由 2 列变 3 列）
    _unpack2 = [t for t in ast.walk(_rs)
                if isinstance(t, ast.Tuple) and len(t.elts) == 2
                and all(isinstance(e, ast.Name) for e in t.elts)]
    _names2 = {tuple(e.id for e in t.elts) for t in _unpack2}
    check('E 无遗留的 2 元解包（rows 已是 3 列）',
          ("nid", "imp") not in _names2, f'仍存在解包 {sorted(_names2)}')

# F 对照：单点解析 helper 的存在与口径
print('— F 解析单点 —')
_sao = None
for _n in ast.walk(_tree):
    if isinstance(_n, ast.FunctionDef) and _n.name == "_state_attr_of":
        _sao = _n
        break
check('F `_state_attr_of` 单点存在', _sao is not None, '')
if _sao is not None:
    _s = ast.get_source_segment(SRC, _sao) or ""
    check('F 单点保留窄口径 except (ValueError, TypeError)',
          "except (ValueError, TypeError)" in _s, '')
    check('F 单点仍拒非 dict（isinstance(sa, dict)）',
          "isinstance(sa, dict)" in _s, '')
    check('F `_read_state_attr` 复用该单点（不复制解析体）',
          "_state_attr_of(" in (ast.get_source_segment(
              SRC, next((n for n in ast.walk(_tree)
                         if isinstance(n, ast.FunctionDef)
                         and n.name == "_read_state_attr"), _sao)) or ""), '')

# ------------------------------------------- G decay_cycle 的遗忘保护不再逐节点 get_node
print('— G 静态面（decay_cycle 的 `_is_forget_protected` 不再逐节点 `get_node`）—')
_dc = None
for _n in ast.walk(_tree):
    if isinstance(_n, ast.FunctionDef) and _n.name == "decay_cycle":
        _dc = _n
        break
check('G0 已定位 decay_cycle', _dc is not None, '')

if _dc is not None:
    _gn = []
    for _loop in ast.walk(_dc):
        if not isinstance(_loop, (ast.For, ast.While)):
            continue
        for _x in ast.walk(_loop):
            if isinstance(_x, ast.Call) and isinstance(_x.func, ast.Attribute) \
                    and _x.func.attr == "get_node":
                _gn.append(_x.lineno)
    check('G 循环体内无 `get_node`（受保护判据改走直传入口）', not _gn,
          f'残留于 L{_gn}' if _gn else '')

    _dsel = [c.value for c in ast.walk(_dc)
             if isinstance(c, ast.Constant) and isinstance(c.value, str)
             and "SELECT" in c.value and "tags" in c.value]
    check('G 两段主查询均已带出 tags', len(_dsel) >= 2,
          f'命中 {len(_dsel)} 段：{[s.strip()[:48] for s in _dsel]}')
    check('G 不再出现 `id, importance FROM nodes` 的两列旧查询',
          not any("id, importance FROM nodes" in s for s in
                  [c.value for c in ast.walk(_dc)
                   if isinstance(c, ast.Constant) and isinstance(c.value, str)]), '')

_p = None
for _n in ast.walk(_tree):
    if isinstance(_n, ast.FunctionDef) and _n.name == "_is_forget_protected":
        _p = _n
        break
check('G `_is_forget_protected` 存在', _p is not None, '')
if _p is not None:
    _ps = ast.get_source_segment(SRC, _p) or ""
    _args = {a.arg for a in _p.args.args} | {a.arg for a in _p.args.kwonlyargs}
    check('G 谓词新增 node_id / tags 直传入口', "node_id" in _args and "tags" in _args,
          f'形参 {sorted(_args)}')
    check('G 判定体仍是单点（no_forget ∈ tags ∪ id ∈ protected，各一次）',
          _ps.count('"no_forget" in (tags or [])') == 1
          and _ps.count("in protected") == 1,
          'no_forget 判定 %d 次 / in protected %d 次'
          % (_ps.count('"no_forget" in (tags or [])'), _ps.count("in protected")))
    check('G 旧入口仍在（node 为 None 即 False）',
          "if node is None:" in _ps or "node_id is None" in _ps, '')

# ------------------------------------------- H 另两处「只读 layer」的逐节点 get_node
print('— H 静态面（_strip_incident_edges / _assert_edge_role 只取 layer 列）—')
for _fname in ("_strip_incident_edges", "_assert_edge_role"):
    _fn2 = None
    for _n2 in ast.walk(_tree):
        if isinstance(_n2, ast.FunctionDef) and _n2.name == _fname:
            _fn2 = _n2
            break
    check(f'H0 已定位 {_fname}', _fn2 is not None, '')
    if _fn2 is None:
        continue
    _hits = []
    for _loop in ast.walk(_fn2):
        if not isinstance(_loop, (ast.For, ast.While)):
            continue
        for _x in ast.walk(_loop):
            if isinstance(_x, ast.Call) and isinstance(_x.func, ast.Attribute) \
                    and _x.func.attr == "get_node":
                _hits.append(_x.lineno)
    check(f'H {_fname} 循环体内无 get_node', not _hits,
          f'残留于 L{_hits}' if _hits else '')
    _seg = ast.get_source_segment(SRC, _fn2) or ""
    check(f'H {_fname} 改取 layer 列', "SELECT layer FROM nodes WHERE id=?" in _seg
          or "SELECT id, layer FROM nodes WHERE id IN" in _seg, '')
    check(f'H {_fname} 用 MemoryLayer(...) 转换（与 from_row 同款）',
          "MemoryLayer(" in _seg, '')

_sie = next((n for n in ast.walk(_tree)
             if isinstance(n, ast.FunctionDef) and n.name == "_strip_incident_edges"), None)
if _sie is not None:
    _s = ast.get_source_segment(SRC, _sie) or ""
    check('H 批量取层沿 `_SQL_IN_CHUNK` 分批（#417 的 IN 上限单点）',
          "_SQL_IN_CHUNK" in _s, '')
    check('H 边行先 fetchall 再迭代（游标可复用）',
          "_rows_e = c.fetchall()" in _s, '')

# ------------------------------------------- I delete_node 的 conflict 标签清理
print('— I 静态面（delete_node 清 conflict 标签只取 tags 列）—')
_dn = next((n for n in ast.walk(_tree)
            if isinstance(n, ast.FunctionDef) and n.name == "delete_node"), None)
check('I0 已定位 delete_node', _dn is not None, '')
if _dn is not None:
    _hits2 = []
    for _loop in ast.walk(_dn):
        if not isinstance(_loop, (ast.For, ast.While)):
            continue
        for _x in ast.walk(_loop):
            if isinstance(_x, ast.Call) and isinstance(_x.func, ast.Attribute) \
                    and _x.func.attr == "get_node":
                _hits2.append(_x.lineno)
    check('I delete_node 循环体内无 get_node', not _hits2,
          f'残留于 L{_hits2}' if _hits2 else '')
    _ds = ast.get_source_segment(SRC, _dn) or ""
    check('I 改取 tags 列', "SELECT tags FROM nodes WHERE id=?" in _ds, '')
    check('I 标签裁剪仍走 `_dumps_tags`（写口径单点）',
          "_dumps_tags(tags)" in _ds, '')
    check('I 「对端仍冲突则保留标签」的 continue 仍在',
          "仍与其它节点冲突" in _ds or "continue" in _ds, '')

# ------------------------------------------- J register_conflict / apply_shared_sync
print('— J 静态面（register_conflict / apply_shared_sync 只取所需列）—')
for _jn in ("register_conflict", "apply_shared_sync"):
    _jf = next((n for n in ast.walk(_tree)
                if isinstance(n, ast.FunctionDef) and n.name == _jn), None)
    check(f'J0 已定位 {_jn}', _jf is not None, '')
    if _jf is None:
        continue
    _jh = []
    for _loop in ast.walk(_jf):
        if not isinstance(_loop, (ast.For, ast.While)):
            continue
        for _x in ast.walk(_loop):
            if isinstance(_x, ast.Call) and isinstance(_x.func, ast.Attribute) \
                    and _x.func.attr == "get_node":
                _jh.append(_x.lineno)
    check(f'J {_jn} 循环体内无 get_node', not _jh,
          f'残留于 L{_jh}' if _jh else '')
    _js = ast.get_source_segment(SRC, _jf) or ""
    check(f'J {_jn} 用 MemoryLayer(...) 转换（与 from_row 同款）',
          "MemoryLayer(" in _js, '')

_rc = next((n for n in ast.walk(_tree)
            if isinstance(n, ast.FunctionDef) and n.name == "register_conflict"), None)
if _rc is not None:
    _rs2 = ast.get_source_segment(SRC, _rc) or ""
    check('J register_conflict 角色闸+存在性检查合并为一次取 layer',
          "SELECT layer FROM nodes WHERE id=?" in _rs2, '')
    check('J register_conflict 存在性用 `_lay[...] is None`（等价旧 `is None`）',
          "_lay[a_id] is None or _lay[b_id] is None" in _rs2, '')
    check('J register_conflict 标签裁剪仍走 `_dumps_tags`',
          "_dumps_tags(_tlist)" in _rs2, '')

_as2 = next((n for n in ast.walk(_tree)
             if isinstance(n, ast.FunctionDef) and n.name == "apply_shared_sync"), None)
if _as2 is not None:
    _asrc = ast.get_source_segment(SRC, _as2) or ""
    check('J apply_shared_sync 冲突判定只取 layer, content 两列',
          "SELECT layer, content FROM nodes WHERE id=?" in _asrc, '')
    check('J apply_shared_sync 两个返回键名未变（existing_layer 仍为 str）',
          '"existing_layer": _xl.value' in _asrc, '')
    check('J apply_shared_sync 的 identical/conflicts/overwrite 分支仍在',
          'identical.append(nid)' in _asrc and 'if not overwrite:' in _asrc, '')

# ------------------------------------------------- K increment_access 批量化
print('— K 静态面（increment_access 收进批量单点，不再每节点新开连接）—')
_ia = next((n for n in ast.walk(_tree)
            if isinstance(n, ast.FunctionDef) and n.name == "increment_access"), None)
_iam = next((n for n in ast.walk(_tree)
             if isinstance(n, ast.FunctionDef) and n.name == "increment_access_many"), None)
check('K0 已定位 increment_access / increment_access_many',
      _ia is not None and _iam is not None, '')
if _ia is not None and _iam is not None:
    _ia_s = ast.get_source_segment(SRC, _ia) or ""
    _iam_s = ast.get_source_segment(SRC, _iam) or ""
    check('K 单节点入口委派给批量单点（不复制实现）',
          "self.increment_access_many([node_id])" in _ia_s, '')
    check('K 批量单点用 executemany（保住重复 id 各计一次）',
          "obs.executemany(" in _iam_s, '')
    check('K 批量单点**没有**用 `WHERE id IN (...)`（那会把重复 id 只计一次）',
          _iam_s.count("obs.execute(") == 1
          and "obs.executemany(" in _iam_s,
          'obs.execute( 调用 %d 次（应仅 PRAGMA 那一次）' % _iam_s.count("obs.execute("))
    check('K 批量单点仍重设连接级 foreign_keys', "PRAGMA foreign_keys=ON" in _iam_s, '')
    check('K 批量单点仍是「尽力而为」（OperationalError ⇒ pass）',
          "except sqlite3.OperationalError:" in _iam_s, '')
    check('K 空输入早退（不发连接）', "if not ids:" in _iam_s, '')

    _loop_ia = []
    for _fn in ast.walk(_tree):
        if not isinstance(_fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for _loop in ast.walk(_fn):
            if not isinstance(_loop, (ast.For, ast.While)):
                continue
            for _x in ast.walk(_loop):
                if isinstance(_x, ast.Call) and isinstance(_x.func, ast.Attribute) \
                        and _x.func.attr == "increment_access":
                    _loop_ia.append((_fn.name, _x.lineno))
    check('K 全仓循环体内不再出现 increment_access', not _loop_ia,
          f'残留 {_loop_ia}' if _loop_ia else '')

    _iam_calls = sum(
        1 for _fn in ast.walk(_tree)
        for _x in ast.walk(_fn)
        if isinstance(_x, ast.Call) and isinstance(_x.func, ast.Attribute)
        and _x.func.attr == "increment_access_many")
    check('K 批量单点有 ≥3 处调用（semantic_search / 检索结果 / 演练）',
          _iam_calls >= 3, f'实测 {_iam_calls} 处')

    _reh = next((n for n in ast.walk(_tree)
                 if isinstance(n, ast.FunctionDef) and n.name == "consolidate_cycle"), None)
    check('K0 已定位 consolidate_cycle', _reh is not None, '')
    if _reh is not None:
        _rs3 = ast.get_source_segment(SRC, _reh) or ""
        check('K 演练面仍只对 `importance >= rehearsal_threshold` 计数',
              "_rehearsed.append(n)" in _rs3, '')
        check('K 演练面四条过滤逐字未动',
              'MemoryLayer.ANCHOR, MemoryLayer.STRUCTURE' in _rs3
              and '"no_forget" in n.tags' in _rs3
              and '"pending_verification" in n.tags' in _rs3
              and 'n.importance >= rehearsal_threshold' in _rs3, '')
        check('K `new_count` 仍由快照 `n.access_count` 推出（不依赖 increment 返回值）',
              "(n.access_count or 0) + 1" in _rs3, '')
        check('K 降权/可压缩面仍在（`elif` 拆成显式 continue 后等价）',
              "n.importance < degrade_threshold" in _rs3
              and 'n.importance >= rehearsal_threshold:\n                continue' in _rs3, '')

print(f'\n=== N+1 优化守卫: {pass_n}/{pass_n + fail_n} 通过 ===')
sys.exit(0 if fail_n == 0 else 1)
