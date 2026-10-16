# -*- coding: utf-8 -*-
"""#744 wm_simloop load 的 mdcg_root 零白名单根闸（读向任意目录枚举）—— 守卫

缺陷（lingshu issue #744 · [core/wm_simloop][安全][P1]）：
  `wm_simloop("load")` 把模型驱动面的 `params["mdcg_root"]` **原样**透传给
  `SimulationLoop.load_priors`，后者只做 `mdcg_root or MDCG_ROOT` 回落、无
  `realpath+normcase` 根判定 ⇒ 指向部署方任一可读目录即枚举其
  `contextual/*.md`，`- entity:` 行被原样 `scene.add_entity` 种进世界模型
  （对抗先验注入）。与写向 `flush` 经 `_confine_simloop_out_path`（#324）的
  白名单根闸**不对称**——本件实测写拒读放（同一根外树 flush 拒、load 放行
  seeded=2 毒实体进 scene_state）。

修法（读向根闸，与写向同款判定、读向语义）：
  门面 load 分支先判后读——`mdcg_root` 须落在 `MDCG_ROOT`（真源·设计的
  读取对象，恒在白名单）＋ `LINGSHU_SIMLOOP_IN_ROOT`（显式加挂面，
  `os.pathsep` 多根）的并集内；越界/零可读根 ⇒ fail-closed 返回 error。
  归一 `realpath + normcase`（与 `_confine_simloop_out_path` 同口径）。
  与写向（#480 起默认根不回落 `MDCG_ROOT`）刻意不同：读真源是 load 的本职。

断言组（抽掉任一修复即红）：
  G1 报告人场景：SUB + mdcg_root=根外树（内含毒实体）⇒ 拒，且毒实体不得
     进世界模型（state.nodes 保持 0——修前此腿 seeded=2、PWNED 进 scene）
  G2 显式加挂根（LINGSHU_SIMLOOP_IN_ROOT）之外的目录 ⇒ 拒
  G3 零可读根（MDCG_ROOT 与 LINGSHU_SIMLOOP_IN_ROOT 均缺）+ 非空 mdcg_root
     ⇒ fail-closed 拒
  G4 `..` 逃逸腿：字面在真源之下、realpath 解到根外 ⇒ 仍拒（归一口径）
  G5 显式加挂根之内 ⇒ 放行（加挂语义不误杀）
  G6（防误杀·反向腿）真源本尊 / 真源内子目录 / 空串回落 MDCG_ROOT ⇒ 照旧放行
  G7 分层口径零回归：直调库 API `load_priors`（调用方已持对象）不设白名单——
     与 `flush_payloads`/`_confine_simloop_out_path`（#324）同一分层判据

运行（仓根）：python -X utf8 -m pytest tests/test_issue744_simloop_load_root_guard.py -q --no-header
"""
from __future__ import annotations

import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.core.core import SpacetimeMemoryEngine, Role  # noqa: E402
from lingshu.world.wm_simloop import SimulationLoop  # noqa: E402

OUT_ENV = SpacetimeMemoryEngine.SIMLOOP_OUT_ROOT_ENV
IN_ENV = SpacetimeMemoryEngine.SIMLOOP_IN_ROOT_ENV

PWN_NOTE = ("---\ntitle: pwn\ntags: [wm_simloop]\n---\n\n"
            "- entity: PWNED_EXTERNAL id=evilA pos=(1.0, 1.5, 1.0)\n"
            "- entity: PWNED_EXTERNAL id=evilB pos=(2.0, 1.5, 2.0)\n")
OK_NOTE = ("---\ntitle: ok\ntags: [wm_simloop]\n---\n\n"
           "- entity: LEGIT_ENTITY id=ok1 pos=(1.0, 1.5, 1.0)\n")


def _mk_root(parent, name, note=OK_NOTE, fname="note.md"):
    """临时 mdcg 根：contextual/ 下放一篇 wm_simloop 先验笔记，返回根路径。"""
    root = os.path.join(parent, name)
    os.makedirs(os.path.join(root, "contextual"), exist_ok=True)
    with open(os.path.join(root, "contextual", fname), "w",
              encoding="utf-8", newline="\n") as f:
        f.write(note)
    return root


def _engine_with_loop(td, role=Role.PRIMARY):
    eng = SpacetimeMemoryEngine(":memory:", role=role)
    r = eng.wm_simloop("setup", {"size": 12, "seed": 3,
                                 "wal_path": os.path.join(td, "wal.jsonl")})
    assert r["status"] == "ok", "前置：setup 失败 %r" % (r,)
    return eng


def test_g1_报告人场景_根外毒实体不得注入世界模型(monkeypatch):
    """G1：SUB + mdcg_root 指向白名单根外树 ⇒ 拒，毒实体不进世界模型。

    修前实测（issue #744 probe）：`status: ok, seeded: 2`，实体
    `[PWNED_EXTERNAL evilA / evilB]` 被种进 scene——本腿连同世界模型终态一起
    断言（只断 error 会漏「拒了但已种入」的半截面）。
    """
    td = tempfile.mkdtemp()
    lib = _mk_root(td, "mdcg_lib")
    outside = _mk_root(td, "outside_lib", note=PWN_NOTE, fname="pwn.md")
    monkeypatch.setenv("MDCG_ROOT", lib)
    monkeypatch.delenv(IN_ENV, raising=False)

    eng = _engine_with_loop(td, role=Role.SUB)
    r = eng.wm_simloop("load", {"mdcg_root": outside})
    assert r["status"] == "error", "G1 根外 mdcg_root 仍放行：%r" % (r,)
    assert "#744" in r.get("error", "") and "mdcg_root" in r.get("error", ""), (
        "G1 被拒但理由不是读向根闸（可能被无关错误假绿）：%r" % (r.get("error"),))
    st = eng.wm_simloop("state")
    assert st["state"]["nodes"] == 0, (
        "G1 毒实体已进世界模型（nodes=%d）——修前缺陷形态"
        % st["state"]["nodes"])


def test_g2_显式加挂根之外的目录被拒(monkeypatch):
    """G2：配置了 LINGSHU_SIMLOOP_IN_ROOT 后，落在真源与加挂根**之外** ⇒ 拒。"""
    td = tempfile.mkdtemp()
    lib = _mk_root(td, "mdcg_lib")
    extra = _mk_root(td, "extra_lib")
    outside = _mk_root(td, "outside_lib", note=PWN_NOTE, fname="pwn.md")
    monkeypatch.setenv("MDCG_ROOT", lib)
    monkeypatch.setenv(IN_ENV, os.pathsep.join([extra, os.path.join(td, "extra2")]))
    _mk_root(td, "extra2")

    eng = _engine_with_loop(td)
    r = eng.wm_simloop("load", {"mdcg_root": outside})
    assert r["status"] == "error", "G2 加挂根之外仍放行：%r" % (r,)
    # 多根第二根（os.pathsep 拆分）之内 ⇒ 放行（多根口径不只在第一根生效）
    r2 = eng.wm_simloop("load", {"mdcg_root": os.path.join(td, "extra2")})
    assert r2["status"] == "ok", "G2b 第二加挂根被误拒：%r" % (r2,)


def test_g3_零可读根则fail_closed(monkeypatch):
    """G3：MDCG_ROOT 与 LINGSHU_SIMLOOP_IN_ROOT 均缺 + 非空 mdcg_root ⇒ 拒。

    无「已配置根」可言 ⇒ 任何模型给的根都不得放（fail-closed，不引入
    「未配置即放开」——#324 同口径）。
    """
    td = tempfile.mkdtemp()
    outside = _mk_root(td, "outside_lib", note=PWN_NOTE, fname="pwn.md")
    monkeypatch.delenv("MDCG_ROOT", raising=False)
    monkeypatch.delenv(IN_ENV, raising=False)

    eng = _engine_with_loop(td)
    r = eng.wm_simloop("load", {"mdcg_root": outside})
    assert r["status"] == "error", "G3 零可读根仍放行：%r" % (r,)
    assert "#744" in r.get("error", ""), "G3 拒绝理由不符：%r" % (r.get("error"),)


def test_g4_点号逃逸不得绕过根闸(monkeypatch):
    """G4：字面路径在真源之下、`realpath` 解到根外（`..` 逃逸）⇒ 仍拒。

    归一口径 `realpath + normcase`（与 `_confine_simloop_out_path` 同款）——
    字面包含性判定（startswith 原串）会被 `lib/../../outside` 这类写法绕过。
    """
    td = tempfile.mkdtemp()
    lib = _mk_root(td, "mdcg_lib")
    outside = _mk_root(td, "outside_lib", note=PWN_NOTE, fname="pwn.md")
    monkeypatch.setenv("MDCG_ROOT", lib)
    monkeypatch.delenv(IN_ENV, raising=False)

    eng = _engine_with_loop(td)
    sneaky = os.path.join(lib, "contextual", "..", "..", "outside_lib")
    assert os.path.normcase(os.path.realpath(sneaky)) != os.path.normcase(lib), \
        "G4 前置：逃逸串应解析到真源之外"
    r = eng.wm_simloop("load", {"mdcg_root": sneaky})
    assert r["status"] == "error", "G4 `..` 逃逸绕过根闸：%r" % (r,)


def test_g5_显式加挂根之内放行(monkeypatch):
    """G5（加挂语义）：真源之外、显式配置面之内 ⇒ 照旧放行（不误杀多语料库部署）。"""
    td = tempfile.mkdtemp()
    lib = _mk_root(td, "mdcg_lib")
    extra = _mk_root(td, "extra_lib")
    monkeypatch.setenv("MDCG_ROOT", lib)
    monkeypatch.setenv(IN_ENV, extra)

    eng = _engine_with_loop(td)
    r = eng.wm_simloop("load", {"mdcg_root": extra})
    assert r["status"] == "ok", "G5 加挂根内被误拒：%r" % (r,)
    assert r["load"]["seeded"] == 1 and r["load"]["entities"][0]["eid"] == "ok1", r


def test_g6_反向腿_真源本尊与子目录与空串照旧放行(monkeypatch):
    """G6（防误杀）：真源恒在白名单——本尊/子目录/空串回落三条腿都不得被误拒。"""
    td = tempfile.mkdtemp()
    lib = _mk_root(td, "mdcg_lib")
    sub = _mk_root(lib, "nested_sub")          # 真源之内的子树（恒可读）
    monkeypatch.setenv("MDCG_ROOT", lib)
    monkeypatch.delenv(IN_ENV, raising=False)

    eng = _engine_with_loop(td)
    for label, root in (("本尊", lib), ("子目录", sub)):
        r = eng.wm_simloop("load", {"mdcg_root": root})
        assert r["status"] == "ok", "G6 真源%s被误拒：%r" % (label, r)
        assert r["load"]["seeded"] == 1, "G6 真源%s载入数不符：%r" % (label, r)
    # 空串 ⇒ load_priors 内回落 MDCG_ROOT（#247 口径不变，真源属白名单）
    r3 = eng.wm_simloop("load", {"mdcg_root": ""})
    assert r3["status"] == "ok" and r3["load"]["seeded"] == 1, (
        "G6 空串回落 MDCG_ROOT 被误拒：%r" % (r3,))


def test_g7_直调库API不设白名单_分层口径零回归(monkeypatch):
    """G7：`SimulationLoop.load_priors` 直调（调用方已持对象）照旧不设白名单。

    分层口径与 #324 相同：模型驱动面经门面 `wm_simloop("load")` 进（闸在
    门面侧）；库 API 调用方本可任意 `open`，此层不设路径白名单。既有直调
    测试（#247/#399/#600）依赖该口径——本腿显式钉死，防有人把闸下沉进
    `load_priors` 造成直调面误杀。
    """
    td = tempfile.mkdtemp()
    outside = _mk_root(td, "outside_lib", note=PWN_NOTE, fname="pwn.md")
    monkeypatch.delenv("MDCG_ROOT", raising=False)
    monkeypatch.delenv(IN_ENV, raising=False)

    loop = SimulationLoop(wal_path=None)
    r = loop.load_priors(mdcg_root=outside, limit=10)
    assert r["status"] == "ok" and r["seeded"] == 2, (
        "G7 直调库 API 被门面闸误杀（分层口径回归）：%r" % (r,))


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
