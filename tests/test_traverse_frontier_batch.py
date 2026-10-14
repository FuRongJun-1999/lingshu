# -*- coding: utf-8 -*-
"""test_traverse_frontier_batch.py · `traverse` 的**整层 frontier 批量取边**守卫

被优化处（`lingshu/core/core.py`）：
    旧实现**逐 frontier 节点**发查询（每层 1–2×|frontier| 条 SQL，扇出大时爆炸：
    实测扇出 200 / 深度 2 ⇒ 401 条）。改为按 `_SQL_IN_CHUNK`（#417 的 IN 绑定上限
    单点）分批把整层查回，再**按 frontier 原顺序**分组消费 ⇒ 每层
    1–2×ceil(|frontier|/chunk) 条（同例 **3** 条）。

本守卫三组：
  A 等价性（**真跑**）：内嵌旧实现作参照，在同一个种子库上对新实现逐例对比
    （含分叉/汇合/环/**同源同目标多边**/`out`/`in`/`both`/类型过滤/深度/阈值/截断）；
  B 查询条数：新实现的 SQL 条数**不随扇出线性增长**（= 2×深度 + 1）；
  C 静态面：新实现用 `IN (...)` 且沿用 `_SQL_IN_CHUNK`；`visited` 仍在 frontier
    顺序内即时更新（决定 `result` 的元素顺序）。

体例与 `tests/` 下其它 test_*.py 一致（`check()` + 退出码）。
"""
import sys
import os
import ast
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


# --------------------------------------------------------------- 参照实现（旧）
def traverse_old(c, start_id, relation_types=None, direction="out",
                 max_depth=5, min_importance=0.0, max_nodes=500, SQL_CHUNK=900):
    rel_clause, params = "", []
    if relation_types:
        ph = ",".join("?" * len(relation_types))
        rel_clause = f" AND e.relation_type IN ({ph})"
        params = list(relation_types)
    frontier, visited, result = [start_id], {start_id}, []
    for depth in range(1, max_depth + 1):
        if not frontier or len(visited) > max_nodes:
            break
        nxt = []
        for nid in frontier:
            if direction in ("out", "both"):
                for row in c.execute(
                        f"SELECT e.target_id AS tid, e.id AS eid FROM edges e "
                        f"WHERE e.source_id=?{rel_clause}", [nid] + params).fetchall():
                    if row["tid"] not in visited:
                        visited.add(row["tid"]); nxt.append(row["tid"])
                        result.append({"node_id": row["tid"], "depth": depth,
                                       "edge_id": row["eid"], "direction": "out"})
            if direction in ("in", "both"):
                for row in c.execute(
                        f"SELECT e.source_id AS sid, e.id AS eid FROM edges e "
                        f"WHERE e.target_id=?{rel_clause}", [nid] + params).fetchall():
                    if row["sid"] not in visited:
                        visited.add(row["sid"]); nxt.append(row["sid"])
                        result.append({"node_id": row["sid"], "depth": depth,
                                       "edge_id": row["eid"], "direction": "in"})
        frontier = nxt
    if result:
        ids = list({r["node_id"] for r in result}); imp = {}
        for i in range(0, len(ids), SQL_CHUNK):
            chunk = ids[i:i + SQL_CHUNK]; ph = ",".join("?" * len(chunk))
            imp.update({row["id"]: row["importance"] for row in c.execute(
                f"SELECT id, importance FROM nodes WHERE id IN ({ph})", chunk).fetchall()})
        result = [r for r in result if imp.get(r["node_id"], 0.0) >= min_importance]
        for r in result:
            r["importance"] = imp.get(r["node_id"], 0.0)
        result.sort(key=lambda x: (-x["importance"], x["depth"]))
    return result[:max_nodes]


SCHEMA = ("CREATE TABLE nodes (id TEXT PRIMARY KEY, importance REAL);"
          "CREATE TABLE edges (id TEXT PRIMARY KEY, source_id TEXT, target_id TEXT,"
          " relation_type TEXT);")


def seed(fanout=6, depth=2):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    ids = ["n0"] + [f"L{d}_{i}" for d in range(1, depth + 1) for i in range(fanout)]
    conn.executemany("INSERT INTO nodes VALUES (?,?)", [(i, 0.5) for i in ids])
    es, k = [], 0
    for i in range(fanout):
        es.append((f"e{k}", "n0", f"L1_{i}", "causal")); k += 1
    for d in range(1, depth):
        for i in range(fanout):
            es.append((f"e{k}", f"L{d}_{i}", f"L{d+1}_{i}", "causal")); k += 1
    conn.executemany("INSERT INTO edges VALUES (?,?,?,?)", es)
    return conn


def seed_tricky():
    """分叉 / 汇合 / 环 / **同源同目标多边** / 多类型。"""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.executemany("INSERT INTO nodes VALUES (?,?)", [(f"n{i}", 0.5) for i in range(12)])
    conn.executemany("INSERT INTO edges VALUES (?,?,?,?)", [
        ("e1", "n0", "n1", "causal"), ("e2", "n0", "n2", "causal"),
        ("e3", "n0", "n3", "other"), ("e4", "n1", "n2", "causal"),
        ("e5", "n1", "n4", "causal"), ("e6", "n2", "n5", "causal"),
        ("e7", "n3", "n6", "causal"), ("e8", "n4", "n5", "causal"),
        ("e9", "n5", "n0", "causal"),                      # 环
        ("e10", "n0", "n1", "causal"),                     # 同源同目标第二条
        ("e11", "n6", "n1", "causal"),
    ])
    return conn


# --------------------------------------------------------------- 被测新实现
_tp = os.path.join(_REPO, "lingshu", "core", "core.py")
_SRC = open(_tp, encoding="utf-8").read()
_TREE = ast.parse(_SRC)
_TV = next((n for n in ast.walk(_TREE)
            if isinstance(n, ast.FunctionDef) and n.name == "traverse"), None)
check('A0 已定位 traverse', _TV is not None, _tp)
if _TV is None:
    print(f'\n=== traverse 整层批量守卫: {pass_n}/{pass_n + fail_n} 通过 ===')
    sys.exit(1)
_TV_SRC = ast.get_source_segment(_SRC, _TV)
#: `traverse` 依赖的模块级常量（与 core.py 同值：`_SQL_IN_CHUNK = 900`）。
#: 从源码里**读**出来而不是硬编码，避免本守卫与被测文件各写一份。
_NS = {"List": list, "Dict": dict}
_m = __import__("re").search(r"^_SQL_IN_CHUNK\s*=\s*(\d+)", _SRC, __import__("re").M)
_NS["_SQL_IN_CHUNK"] = int(_m.group(1)) if _m else 900
exec("class _S:\n    def __init__(self, c):\n        self.conn = c\n"
     + "\n".join("    " + l for l in _TV_SRC.splitlines()), _NS)


class _Conn:
    def __init__(self, conn):
        self._c = conn

    def cursor(self):
        return self._c.cursor()


def traverse_new(conn, *a, **k):
    return _NS["_S"](_Conn(conn)).traverse(*a, **k)


CASES = [
    ("tricky out", seed_tricky, "n0", dict(direction="out")),
    ("tricky both", seed_tricky, "n0", dict(direction="both")),
    ("tricky in", seed_tricky, "n0", dict(direction="in")),
    ("tricky causal", seed_tricky, "n0", dict(direction="both", relation_types=["causal"])),
    ("tricky depth1", seed_tricky, "n0", dict(direction="both", max_depth=1)),
    ("tricky minimport", seed_tricky, "n0", dict(direction="both", min_importance=0.6)),
    ("tricky max3", seed_tricky, "n0", dict(direction="both", max_nodes=3)),
    ("fan6 out", seed, "n0", dict(direction="out")),
    ("fan6 both d2", seed, "n0", dict(direction="both", max_depth=2)),
]

print('— A 等价性（真跑：内嵌旧实现 vs 被测新实现）—')
for name, mk, start, kw in CASES:
    a = traverse_old(mk().cursor(), start, **kw)
    b = traverse_new(mk(), start, **kw)
    check(f'A {name}', a == b,
          '' if a == b else f'\n     旧={a}\n     新={b}')

print('— B 查询条数不随扇出线性增长 —')


class _Cur:
    def __init__(self, cur, box):
        self._c, self._b = cur, box

    def execute(self, *a, **k):
        self._b[0] += 1
        return self._c.execute(*a, **k)

    def fetchall(self):
        return self._c.fetchall()


class _CC:
    def __init__(self, conn):
        self._c, self.n = conn, [0]

    def cursor(self):
        return _Cur(self._c.cursor(), self.n)


for _fan in (6, 40, 120):
    _cc = _CC(seed(fanout=_fan, depth=2))
    _r = traverse_new(_cc, "n0", direction="out", max_depth=2)
    _n = _cc.n[0]
    check(f'B 扇出 {_fan}: 查询 {_n} 条（= 2×深度+1 = 5，与扇出无关）',
          _n <= 5 and len(_r) == _fan * 2, f'实测 {_n} 条 / 结果 {len(_r)} 条')

_cc2 = _CC(seed(fanout=40, depth=2))
traverse_old(_cc2.cursor(), "n0", direction="out", max_depth=2)
check('B 旧实现同例需 >40 条（扇出线性）', _cc2.n[0] > 40, f'实测 {_cc2.n[0]} 条')

print('— C 静态面 —')
_ts = _TV_SRC
check('C 用整层 `IN (...)` 取边', "e.source_id IN (" in _ts and "e.target_id IN (" in _ts, '')
check('C 沿用 `_SQL_IN_CHUNK`（#417 的 IN 上限单点）', "_SQL_IN_CHUNK" in _ts, '')
check('C 仍按 frontier 原顺序消费（`for nid in frontier`）',
      "for nid in frontier:" in _ts, '')
check('C `visited` 仍在循环内即时更新（决定 result 顺序）',
      "if row[\"tid\"] not in visited:" in _ts and "if row[\"sid\"] not in visited:" in _ts, '')
_sql_consts = [c.value for c in ast.walk(_TV)
               if isinstance(c, ast.Constant) and isinstance(c.value, str)
               and ("SELECT" in c.value or "FROM " in c.value)]
check('C **未**新增 `ORDER BY`（不改变行序口径）',
      not any("ORDER BY" in s.upper() for s in _sql_consts),
      f'SQL 常量里出现 ORDER BY：{[s.strip()[:60] for s in _sql_consts if "ORDER BY" in s.upper()]}')
check('C `rel_clause` / `params` 原样沿用（类型过滤语义未变）',
      "rel_clause" in _ts and "_chunk + params" in _ts, '')
check('C importance 补全仍按 `_SQL_IN_CHUNK` 分批（#417 未回退）',
      "for i in range(0, len(ids), _SQL_IN_CHUNK):" in _ts, '')

print(f'\n=== traverse 整层批量守卫: {pass_n}/{pass_n + fail_n} 通过 ===')
sys.exit(0 if fail_n == 0 else 1)
