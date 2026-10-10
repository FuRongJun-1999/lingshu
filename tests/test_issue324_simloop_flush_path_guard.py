# -*- coding: utf-8 -*-
"""#324 wm_simloop 门面 flush 的 out_path 无约束且 "w" 覆盖 —— 守卫

缺陷（lingshu issue #324 · [core/wm_simloop 门面][安全]）：
  `SpacetimeMemoryEngine.wm_simloop("flush", params)` 把模型驱动面给的
  `params["out_path"]` **原样**交给 `SimulationLoop.flush_payloads`，后者
  `with open(out_path, "w", encoding="utf-8")` ⇒ ①**任意路径可写**（无白名单、
  无根约束）；②**覆盖式且非原子**（先截断再写，中途失败留下截断坏档）。

修法：
  · 越界闸在**门面侧**（`_confine_simloop_out_path`，先判后写）：白名单根＝
    显式配置面 `LINGSHU_SIMLOOP_OUT_ROOT`（`os.pathsep` 多根）优先，未配置则回落
    认知图库根 `MDCG_ROOT`（本门面 `load` 分支的同一真源），两者皆无 ⇒ fail-closed。
    归一用 `realpath + normcase` ⇒ 软链/junction 别名与 `..` 逃逸不得绕过。
  · 原子写在下层 `SimulationLoop.flush_payloads`：同目录临时文件 + `os.replace`。

判据来源：经验标定（本件 #324）——口径取「显式面 → 认知图库根 → fail-closed」
（与 #85 三级回落链同形）＋「临时文件 + os.replace」；无理论章节规定该口径，
追不到更早出处。

断言组（抽掉任一修复即红）：
  A 未配置任何根 ⇒ 门面 flush 拒绝且**零落盘**（先判后写）
  B 显式根内 ⇒ 照旧写成功，`written` 与产物内容逐字正确（防误杀）
  C 越出显式根（含 `..` 逃逸）⇒ 拒绝且零落盘
  D 未配置显式根时回落 `MDCG_ROOT` ⇒ 根内可写、根外拒绝
  E 原子性：换入失败（`os.replace` 抛）⇒ 目标文件仍是**旧内容**（旧实现＝先截断成空）

运行（仓根）：python -X utf8 -m pytest tests/test_issue324_simloop_flush_path_guard.py -q --no-header
"""
from __future__ import annotations

import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.core.core import SpacetimeMemoryEngine  # noqa: E402
from lingshu.world import wm_simloop as wm_mod  # noqa: E402
from lingshu.world.wm_simloop import SimulationLoop  # noqa: E402

ENV = SpacetimeMemoryEngine.SIMLOOP_OUT_ROOT_ENV


def _engine_with_loop(td):
    eng = SpacetimeMemoryEngine(":memory:")
    r = eng.wm_simloop("setup", {"size": 12, "seed": 3,
                                 "wal_path": os.path.join(td, "wal.jsonl")})
    assert r["status"] == "ok", "前置：setup 失败 %r" % (r,)
    return eng


def _clear_roots(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv("MDCG_ROOT", raising=False)


def test_a_未配置任何根则拒绝且零落盘(monkeypatch):
    """A：连根都没有 ⇒ fail-closed，且目标路径**不存在**（先判后写）。"""
    td = tempfile.mkdtemp()
    _clear_roots(monkeypatch)
    eng = _engine_with_loop(td)
    out = os.path.join(td, "anywhere.md")
    r = eng.wm_simloop("flush", {"out_path": out})
    assert r["status"] == "error", "A 未配置根时仍放行写入：%r" % (r,)
    assert "#324" in r.get("error", ""), "A 拒绝理由未点名 #324：%r" % (r,)
    assert not os.path.exists(out), "A 被拒路径仍落盘（非先判后写）：%s" % out


def test_b_显式根内照旧写成功(monkeypatch):
    """B：防误杀——显式根内必须照旧写成功，且内容逐字正确。"""
    td = tempfile.mkdtemp()
    root = os.path.join(td, "out_root")
    os.makedirs(root, exist_ok=True)
    monkeypatch.setenv(ENV, root)
    monkeypatch.delenv("MDCG_ROOT", raising=False)
    eng = _engine_with_loop(td)

    out = os.path.join(root, "contextual", "run.md")
    r = eng.wm_simloop("flush", {"out_path": out})
    assert r["status"] == "ok", "B 根内写入被误拒：%r" % (r,)
    assert r["flush"]["written"] == out, "B written 不符：%r" % (r["flush"]["written"],)
    with open(out, encoding="utf-8") as f:
        got = f.read()
    assert got == r["flush"]["payload"]["body_md"], "B 落盘内容与载荷不一致"
    assert got.startswith("---\n") and "wm_simloop" in got, "B 产物形态不对"


def test_c_越出显式根被拒含点划逃逸(monkeypatch):
    """C：根外路径（含 `..` 逃逸）⇒ 拒绝且零落盘。"""
    td = tempfile.mkdtemp()
    root = os.path.join(td, "out_root")
    os.makedirs(root, exist_ok=True)
    monkeypatch.setenv(ENV, root)
    monkeypatch.delenv("MDCG_ROOT", raising=False)
    eng = _engine_with_loop(td)

    for name, out in (("根外同级", os.path.join(td, "outside.md")),
                      ("点划逃逸", os.path.join(root, "..", "escaped.md"))):
        r = eng.wm_simloop("flush", {"out_path": out})
        assert r["status"] == "error", "C %s 未被拒：%r" % (name, r,)
        assert not os.path.exists(os.path.abspath(out)), \
            "C %s 被拒后仍落盘：%s" % (name, out)


def test_d_回落_MDCG_ROOT(monkeypatch):
    """D：未配置显式根 ⇒ 回落认知图库根（根内可写、根外拒）。"""
    td = tempfile.mkdtemp()
    lib = os.path.join(td, "mdcg_lib")
    os.makedirs(lib, exist_ok=True)
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setenv("MDCG_ROOT", lib)
    eng = _engine_with_loop(td)

    inside = os.path.join(lib, "contextual", "ok.md")
    r = eng.wm_simloop("flush", {"out_path": inside})
    assert r["status"] == "ok", "D 回落库根内写入被误拒：%r" % (r,)
    assert os.path.exists(inside), "D 库根内未落盘"

    outside = os.path.join(td, "nope.md")
    r2 = eng.wm_simloop("flush", {"out_path": outside})
    assert r2["status"] == "error", "D 库根外未被拒：%r" % (r2,)
    assert not os.path.exists(outside), "D 库根外被拒后仍落盘"


class _OsShim:
    """包一层真 `os`，只把 `replace` 换成必失败（模拟「换入」这一步炸掉）。"""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def replace(self, *a, **k):
        raise OSError("模拟换入失败（#324 原子性探针）")


def test_e_换入失败不毁旧档(monkeypatch):
    """E：原子性——换入失败时目标文件仍是**旧内容**（旧实现先截断成空）。"""
    td = tempfile.mkdtemp()
    out = os.path.join(td, "existing.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("旧内容：不可被半截写毁掉")

    loop = SimulationLoop(size=10, seed=1, wal_path=os.path.join(td, "wal.jsonl"))
    loop.wm.perceive()
    monkeypatch.setattr(wm_mod, "os", _OsShim(os))

    raised = None
    try:
        loop.flush_payloads(out_path=out)
    except OSError as e:
        raised = e
    assert raised is not None, "E 换入失败未抛出（探针无效）"
    with open(out, encoding="utf-8") as f:
        got = f.read()
    assert got == "旧内容：不可被半截写毁掉", (
        "#324 换入失败后目标文件被毁：%r（旧实现＝先截断再写）" % (got,))
    leftovers = [n for n in os.listdir(td) if ".tmp-" in n]
    assert leftovers == [], "E 临时文件未清理：%r" % (leftovers,)


def test_f_正常落盘不留临时件(monkeypatch):
    """F：正常路径下不留 `.tmp-` 残件，且覆盖式语义保留（换入新内容）。"""
    td = tempfile.mkdtemp()
    out = os.path.join(td, "target.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("旧内容")

    loop = SimulationLoop(size=10, seed=1, wal_path=os.path.join(td, "wal.jsonl"))
    loop.wm.perceive()
    loop.flush_payloads(out_path=out)

    with open(out, encoding="utf-8") as f:
        got = f.read()
    assert got.startswith("---\n") and "旧内容" not in got, "F 覆盖式语义丢失"
    assert [n for n in os.listdir(td) if ".tmp-" in n] == [], "F 残留临时文件"
