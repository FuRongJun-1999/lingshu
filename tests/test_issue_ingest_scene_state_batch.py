# -*- coding: utf-8 -*-
"""test_issue_ingest_scene_state_batch.py · `ingest_scene` 状态写入的**批量提交**守卫

被优化处（`lingshu/world/scene_model.py`）：
    旧实现每个实体一次
        store.conn.execute("UPDATE nodes SET state_attributes=? WHERE id=?", …)
        store.conn.commit()                       # ← N 次提交（N 次 fsync）
    改为「先登记 → 循环后一次 `executemany` + 一次提交」；**批量失败时回落到逐条
    路径**，保留原有的逐实体 `RuntimeWarning` 文案与「不阻断导入」语义。

本守卫五组：
  A 主循环不再逐实体提交：`ingest_scene` 的**第一个 `for`** 体内无 `execute`/`commit`；
  B 循环后恰有**一次** `executemany` + **一次** `commit`；
  C 回落路径仍在：`except` 分支里有逐条循环，且 `RuntimeWarning` 文案逐字保留；
  D 回落前先 `rollback()`（丢掉批量可能已开的隐式事务）；
  E 对照（不误杀）：`written` 仍逐实体 append、返回的仍是节点 id 列表。

体例与 `tests/` 下其它 test_*.py 一致（`check()` + 退出码）。
"""
import sys
import os
import ast

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


PATH = os.path.join(_REPO, "lingshu", "world", "scene_model.py")
SRC = open(PATH, encoding="utf-8").read()
TREE = ast.parse(SRC)

_fn = next((n for n in ast.walk(TREE)
            if isinstance(n, ast.FunctionDef) and n.name == "ingest_scene"), None)
check('A0 已定位 ingest_scene', _fn is not None, PATH)

if _fn is not None:
    _src = ast.get_source_segment(SRC, _fn) or ""
    _loops = [n for n in ast.walk(_fn) if isinstance(n, (ast.For, ast.While))]
    _main_loop = min(_loops, key=lambda n: n.lineno) if _loops else None
    check('A 找到主循环（按行号最小）', _main_loop is not None, '')

    if _main_loop is not None:
        _bad = []
        for _x in ast.walk(_main_loop):
            if isinstance(_x, ast.Call) and isinstance(_x.func, ast.Attribute) \
                    and _x.func.attr in ("execute", "commit", "executemany"):
                _bad.append((_x.func.attr, _x.lineno))
        check('A 主循环体内不再逐实体 execute/commit', not _bad,
              f'残留 {_bad}' if _bad else '')

    _em = [n.lineno for n in ast.walk(_fn)
           if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
           and n.func.attr == "executemany"]
    check('B 循环后恰有一次 executemany', len(_em) == 1, f'实测 {len(_em)} 处 @L{_em}')

    _cm = [n.lineno for n in ast.walk(_fn)
           if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
           and n.func.attr == "commit"]
    check('B commit 总数从「每实体一次」降到「批量一次 + 回落路径」',
          len(_cm) <= 2, f'实测 {len(_cm)} 处 @L{_cm}')

    check('C 回落路径的逐条告警文案逐字保留',
          '场景实体 %s 的状态 %r 未写入脑侧台账' in _src
          and '重建世界时该实体状态会回落为 neutral' in _src, '')
    check('C 仍是 RuntimeWarning（类别未变）', 'RuntimeWarning' in _src, '')
    check('C 仍有 `rollback()`（批量失败后丢掉隐式事务）',
          "store.conn.rollback()" in _src, '')
    check('C 回落路径是「逐条重试」的循环',
          "_pending_states" in _src and _src.count("for _nm, _st, _nid in _pending_states") >= 1, '')

    # E 对照：written 仍逐实体 append，返回类型不变
    check('E `written.append(node.id)` 仍在主循环内',
          "written.append(node.id)" in _src, '')
    check('E 仍 `return written`', "return written" in _src, '')
    check('E 三条过滤/解析分支未动（格式解析与 skip_dedup 选择面）',
          'parts = [p.strip() for p in ln.split("|")]' in _src
          and "skip_dedup=True" in _src, '')
    check('E 函数签名未变（agent, scene_desc, store=None, tag="spatial"）',
          "def ingest_scene(agent, scene_desc: str, store=None, tag: str = \"spatial\")" in _src, '')

print(f'\n=== ingest_scene 状态批量提交守卫: {pass_n}/{pass_n + fail_n} 通过 ===')
sys.exit(0 if fail_n == 0 else 1)
