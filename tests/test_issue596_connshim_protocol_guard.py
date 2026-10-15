# -*- coding: utf-8 -*-
"""test_issue596_connshim_protocol_guard.py · `_ConnShim` 协议面守卫（#596）

被修处（`lingshu/world/brain_store.py`）：`_ConnShim` 此前只有 execute/commit，
而 `scene_model.ingest_scene` 的状态批量写（#515）走 store.conn 的四个方法
（executemany→commit→[失败] rollback→逐条 execute）——脑侧 executemany/
rollback 恒 AttributeError（被 ingest_scene 的兜底 except 吞掉），静默回落
逐条路径，#515 的「省 N−1 次提交」对脑侧永不成立（#596）。

本守卫四组：
  G1 协议面：`_ConnShim` 实例四方法 hasattr 全 True
     （execute/commit/executemany/rollback）；
  G2 语义：executemany(sql, seq) 对多条参数**逐条生效**（条数与内容、次序）；
     空 seq 为 no-op；rollback() 可调用不抛；executemany 对真失败仍抛
     BrainError / 对非法 SQL 仍抛 NotImplementedError（回落路径不被堵死）；
  G3 防过修：execute/commit 既有行为零回归（翻译语义、record_state→False
     且有状态时抛 BrainError、非法 SQL 抛 NotImplementedError、commit no-op）；
  E  端到端：真实 ingest_scene + tracing 桩（`__getattr__` 只在正常查找失败
     时记录）——零缺失方法命中，N 条状态全入账、无告警（批量路径直通）。

体例与 `tests/` 下其它 test_*.py 一致（`check()` + 退出码）。
"""
import json
import os
import sys
import warnings

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.world.brain_store import _ConnShim           # noqa: E402
from lingshu.world import scene_model                     # noqa: E402

pass_n = fail_n = 0


def check(name, ok, detail=''):
    global pass_n, fail_n
    if ok:
        pass_n += 1
    else:
        fail_n += 1
    print(f'[{"✓" if ok else "✘"}] {name}{" — " + detail if detail else ""}')


class _StubStore:
    """最小桩：只提供 _ConnShim 消费的 record_state，记账可回读。"""

    def __init__(self, ok=True):
        self.calls = []
        self.ok = ok

    def record_state(self, node_id, state):
        self.calls.append((str(node_id), state))
        return self.ok


UPDATE_SQL = "UPDATE nodes SET state_attributes=? WHERE id=?"

# ---- G1 协议面 ----
shim = _ConnShim(_StubStore())
_face = {m: hasattr(shim, m) for m in ("execute", "commit", "executemany", "rollback")}
check('G1 四方法 hasattr 全 True（execute/commit/executemany/rollback）',
      all(_face.values()), f'实测 {_face}')

# ---- G2 语义：executemany / rollback ----
st = _StubStore()
shim = _ConnShim(st)
seq = [(json.dumps({"state": s}), nid)
       for nid, s in (("n1", "closed"), ("n2", "new"), ("n3", "broken"))]
try:
    shim.executemany(UPDATE_SQL, seq)
    _g2a = st.calls == [("n1", "closed"), ("n2", "new"), ("n3", "broken")]
    _g2a_d = f'实测 {st.calls}'
except AttributeError as exc:                # 变异（删 executemany）→ 干净红，不崩
    _g2a, _g2a_d = False, f'AttributeError: {exc}'
check('G2 executemany 多条参数逐条生效（3 条、内容与次序逐位）', _g2a, _g2a_d)

st0 = _StubStore()
try:
    _ConnShim(st0).executemany(UPDATE_SQL, [])
    _g2b, _g2b_d = st0.calls == [], f'实测 {st0.calls}'
except AttributeError as exc:
    _g2b, _g2b_d = False, f'AttributeError: {exc}'
check('G2 空 seq 为 no-op（与 sqlite3.executemany 口径一致）', _g2b, _g2b_d)

try:
    _ConnShim(_StubStore()).rollback()
    check('G2 rollback() 可调用不抛', True)
except Exception as exc:                                    # noqa: BLE001
    check('G2 rollback() 可调用不抛', False, f'{type(exc).__name__}: {exc}')

try:
    _ConnShim(_StubStore(ok=False)).executemany(
        UPDATE_SQL, [(json.dumps({"state": "x"}), "n1")])
    check('G2 executemany 对「状态未入账」仍抛 BrainError（回落路径不被堵死）',
          False, '未抛')
except Exception as exc:                                    # noqa: BLE001
    check('G2 executemany 对「状态未入账」仍抛 BrainError（回落路径不被堵死）',
          type(exc).__name__ == "BrainError", f'{type(exc).__name__}: {exc}')

try:
    _ConnShim(_StubStore()).executemany("DELETE FROM nodes", [("x",)])
    check('G2 executemany 对非状态写 SQL 仍抛 NotImplementedError（走 execute 同款契约）',
          False, '未抛')
except Exception as exc:                                    # noqa: BLE001
    check('G2 executemany 对非状态写 SQL 仍抛 NotImplementedError（走 execute 同款契约）',
          type(exc).__name__ == "NotImplementedError", f'{type(exc).__name__}: {exc}')

# ---- G3 防过修：execute / commit 既有行为零回归 ----
st = _StubStore()
shim = _ConnShim(st)
r = shim.execute(UPDATE_SQL, (json.dumps({"state": "happy"}), "nid-7"))
check('G3 execute 单条翻译语义不变（subject=节点 id、value=状态）',
      st.calls == [("nid-7", "happy")] and r is shim, f'实测 {st.calls}')

st_none = _StubStore(ok=False)
try:
    _ConnShim(st_none).execute(UPDATE_SQL, (json.dumps({}), "n1"))
    check('G3 execute 空 payload（无状态）不因 record_state→False 抛错（既有语义）',
          st_none.calls == [("n1", None)])
except Exception as exc:                                    # noqa: BLE001
    check('G3 execute 空 payload（无状态）不因 record_state→False 抛错（既有语义）',
          False, f'{type(exc).__name__}: {exc}')

try:
    _ConnShim(_StubStore(ok=False)).execute(
        UPDATE_SQL, (json.dumps({"state": "sad"}), "n1"))
    check('G3 execute 有状态而 record_state→False 仍抛 BrainError（不静默吞）',
          False, '未抛')
except Exception as exc:                                    # noqa: BLE001
    check('G3 execute 有状态而 record_state→False 仍抛 BrainError（不静默吞）',
          type(exc).__name__ == "BrainError", f'{type(exc).__name__}: {exc}')

try:
    _ConnShim(_StubStore()).execute("DROP TABLE nodes", ())
    check('G3 execute 对非状态写 SQL 仍抛 NotImplementedError（契约不变）',
          False, '未抛')
except Exception as exc:                                    # noqa: BLE001
    check('G3 execute 对非状态写 SQL 仍抛 NotImplementedError（契约不变）',
          type(exc).__name__ == "NotImplementedError", f'{type(exc).__name__}: {exc}')

try:
    check('G3 commit() 仍为可调用 no-op', _ConnShim(_StubStore()).commit() is None)
except Exception as exc:                                    # noqa: BLE001
    check('G3 commit() 仍为可调用 no-op', False, f'{type(exc).__name__}: {exc}')

# ---- E 端到端：真实 ingest_scene + tracing 桩 ----
_missing = []


class _TraceShim(_ConnShim):
    """__getattr__ 只在正常查找失败时触发——记录缺失方法命中（#596 复现面）。"""

    def __getattr__(self, name):
        _missing.append(name)
        raise AttributeError(name)


class _Node:
    def __init__(self, i):
        self.id = i


class _Agent:
    def __init__(self):
        self.n = 0

    def remember(self, content, importance, tags, entities):
        self.n += 1
        return _Node("n%d" % self.n)


class _Store:
    def __init__(self, conn):
        self.conn = conn


_est = _StubStore()
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    written = scene_model.ingest_scene(
        _Agent(), "门|fixture|0,0,1|closed\n桌|fixture|1,0,1|new\n椅|fixture|2,0,1|broken",
        store=_Store(_TraceShim(_est)))
check('E ingest_scene 零缺失方法命中（批量路径对脑侧直通，#596 前=[executemany,rollback]）',
      _missing == [], f'实测命中 {_missing}')
check('E 3 条实体状态全入账（节点 id 为键）',
      _est.calls == [("n1", "closed"), ("n2", "new"), ("n3", "broken")],
      f'实测 {_est.calls}')
check('E 无 RuntimeWarning（批量路径成功时不告警）', len(w) == 0,
      f'实测 {[x.category.__name__ for x in w]}')
check('E 返回值仍是节点 id 列表', written == ["n1", "n2", "n3"], f'实测 {written}')

print(f'\n=== _ConnShim 协议面守卫（#596）: {pass_n}/{pass_n + fail_n} 通过 ===')
sys.exit(0 if fail_n == 0 else 1)
