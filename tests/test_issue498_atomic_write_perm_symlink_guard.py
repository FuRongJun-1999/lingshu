# -*- coding: utf-8 -*-
"""#498 原子写「不继承旧档权限 / 断符号链接」—— 守卫（三处落盘函数）

缺陷（lingshu issue #498 · [core/M13 导出 · nn/hex_cnn][P1]）：
  #330（`SpacetimeMemoryEngine.export_all`）/ #393（`hex_cnn.save_consolidated`）
  / #324（`SimulationLoop.flush_payloads`）的原子写修复都取
  「同目录临时件 + `os.replace`」。`os.replace` 把目标换成**新 inode** ⇒
    ①新档权限取 `0o666 & ~umask`，**不继承**旧档权限——运维刻意对备份设的
      `0600` 被静默放宽成 `0644`（全库导出含 `sensitivity=private` 条目、自我层
      快照、action_logs ⇒ 同机其他用户可读，信息泄露面）；
    ②目标是**符号链接**时替换的是**链接本身**——链接被换成普通文件，链接指向的
      真实备份位置不再更新（静默的灾备失效）。
  旧实现 `open(path, "w")` 截断写**同一 inode**，权限与符号链接都自然保留
  （对照旧基准 e18562f）。

修法（最小 · 本件 #498）：目标先 `os.path.realpath` 穿透符号链接（临时件建在
真实目标同目录 ⇒ 同卷原子性不丢）；`os.replace` 前
`os.chmod(tmp, os.stat(target).st_mode & 0o7777)` 继承旧档权限位（目标不存在 ⇒
`FileNotFoundError` ⇒ 跳过，保持 umask 默认）。

★平台可判性（本机 = Windows，如实标注，**不**编造 Linux 读数）：
  · **权限面**：Windows 上 `os.chmod(f, 0o600)` 后 `stat.S_IMODE(f) == 0o666`
    （只有只读位可表达）⇒「0600 备份变 0644」在本机**不可端到端复现**。
    故权限继承语义改用**跨平台可判**的观测量具取证：把 `os.chmod` / `os.replace`
    打桩记录、把 `os.stat(target).st_mode` 伪造成 `0o600`（A3/B3/C3、A4/B4/C4）
    —— 抽掉修复即红，Windows 上同样有判别力。
  · **符号链接面**：A6/B6/C6 在各自测试目录实际创建文件符号链接，成功就验证
    落盘后链接保留、真实件更新。仅 Windows 报缺少创建链接权限（WinError 1314）
    时跳过这三项中的对应测试；A4/B4/C4 仍通过带 `..` 的路径验证 `realpath`
    规范化，其他创建链接错误照常使测试失败。
  · **正对照**（P0/P1）：证明上面两条 SKIP 是**平台限制**、不是量具失效——
    P0 断言本平台确实能读写 `0o444`（posix 上再加 `0o600`）；P1 断言 symlink
    在本平台要么成功、要么以特权错失败（两种结局都断言 ⇒ 必然为正）。

断言组（抽掉修复 ⇒ A3/A4 等红；还原 ⇒ 全绿）：
  P0/P1/P2/P3 平台能力自证（必然为正的对照——「消融为负」本身也被消融）
  A1/A2 反向腿：export_all 目标不存在 / 普通目标 ⇒ 行为与改前逐字一致
  A3     export_all 继承旧档权限位（打桩：`chmod(tmp, 0o600)` 确实发生）
  A4     export_all 落盘目标取 realpath（带 `..` 的路径被规范化掉）
  A5     [posix] export_all 真权限面 0600 保持
  A6     [需文件符号链接能力] export_all 保留链接且更新真实件
  A7     换入失败 ⇒ 旧档不动且**不留残件**（跨平台；钉住「chmod 后删不掉」反向腿）
  B1..B7 save_consolidated 同构
  C1..C7 flush_payloads 同构

运行（仓根）：python -X utf8 -m pytest tests/test_issue498_atomic_write_perm_symlink_guard.py -q --no-header
"""
from __future__ import annotations

import json
import os
import stat
import sys
import tempfile

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from lingshu.core.core import SpacetimeMemoryEngine  # noqa: E402
from lingshu.nn.hex_cnn import save_consolidated  # noqa: E402
from lingshu.world.wm_simloop import SimulationLoop  # noqa: E402

POSIX = os.name == "posix"
SKIP_PERM = pytest.mark.skipif(
    not POSIX,
    reason="本机 %s：`os.chmod(0o600)` 后 st_mode 仍为 0o666（Windows 只表达只读位）"
           "⇒ 0600 权限面不可端到端复现；该语义由 A3/B3/C3 打桩取证" % sys.platform)


# ==================== 量具 ====================

def _spy(monkeypatch):
    """打桩 `os.replace` / `os.chmod`，记录 (src, dst) / (path, mode) 后转发真实实现。"""
    calls = {"replace": [], "chmod": []}
    real_replace, real_chmod = os.replace, os.chmod

    def _replace(src, dst, **kw):
        calls["replace"].append((str(src), str(dst)))
        return real_replace(src, dst, **kw)

    def _chmod(path, mode, **kw):
        calls["chmod"].append((str(path), mode))
        return real_chmod(path, mode, **kw)

    monkeypatch.setattr(os, "replace", _replace)
    monkeypatch.setattr(os, "chmod", _chmod)
    return calls


def _fake_mode(monkeypatch, target, mode):
    """让 `os.stat(target)` 报出指定 mode——跨平台取证「继承旧档权限位」语义。"""
    real_stat = os.stat
    key = os.path.normcase(os.path.abspath(target))

    def _stat(path, *a, **kw):
        st = real_stat(path, *a, **kw)
        try:
            hit = os.path.normcase(os.path.abspath(str(path))) == key
        except (TypeError, ValueError):
            hit = False
        if hit:
            vals = list(st)
            vals[0] = (vals[0] & ~0o7777) | (mode & 0o7777)
            return os.stat_result(vals)
        return st

    monkeypatch.setattr(os, "stat", _stat)


def _tmp_chmods(calls, target):
    """取落在 target 同目录、名为 `<target>.tmp-*` 的那些 chmod 的 (path, mode)。"""
    d = os.path.normcase(os.path.dirname(os.path.abspath(target)))
    out = []
    for path, mode in calls["chmod"]:
        if os.path.normcase(os.path.dirname(os.path.abspath(path))) != d:
            continue
        if ".tmp-" in os.path.basename(path):
            out.append((path, mode))
    return out


def _replace_dsts(calls):
    return [dst for _, dst in calls["replace"]]


def _same_path(a, b):
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def _try_file_symlink(target, link):
    """创建并验证真实链接；仅 Windows 缺创建链接权限时返回 False。"""
    try:
        os.symlink(target, link)
    except OSError as exc:
        winerror = getattr(exc, "winerror", None)
        privilege_error = (
            winerror == 1314
            or (winerror is None and exc.errno == 22
                and ("privilege" in str(exc).lower() or "特权" in str(exc)))
        )
        if sys.platform == "win32" and privilege_error:
            return False
        raise
    assert os.path.islink(link), "os.symlink 返回成功却未创建链接"
    assert os.path.samefile(target, link), "创建的链接未指向测试目标"
    return True


def _require_file_symlink(target, link):
    if not _try_file_symlink(target, link):
        pytest.skip("Windows 当前用户没有创建文件符号链接的权限")


# ==================== 三个被测落盘入口 ====================

def _engine():
    eng = SpacetimeMemoryEngine(":memory:")
    eng.add_perception("导出探针", skip_dedup=True)
    return eng


def _do_export_all(path):
    return _engine().export_all(path)


def _do_save_consolidated(path):
    return save_consolidated({"k": np.zeros(7, dtype=np.float32)}, {"note": "x"}, path)


def _do_flush_payloads(path):
    return SimulationLoop(size=4, seed=1).flush_payloads(path)


def _sentinel(path, body='{"old": true}'):
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    return path


# ==================== P：平台能力自证（必然为正的对照） ====================

def test_p0_权限量具在本平台有效():
    """P0：证明 SKIP_PERM 是平台限制而非量具失效——本平台确实能读写权限位。"""
    td = tempfile.mkdtemp(prefix="b498_p0_")
    f = os.path.join(td, "p.txt")
    with open(f, "w", encoding="utf-8") as fh:
        fh.write("x")
    os.chmod(f, 0o444)
    assert stat.S_IMODE(os.stat(f).st_mode) == 0o444, \
        "P0 量具失效：连只读位都读不出，SKIP_PERM 的理由不成立"
    os.chmod(f, 0o600)
    got = stat.S_IMODE(os.stat(f).st_mode)
    if POSIX:
        assert got == 0o600, "P0 posix 上 0600 必须可表达（实测 %s）" % oct(got)
    else:
        assert got == 0o666, (
            "P0 本机（%s）`chmod(0o600)` 后读到 %s——若该读数变了，"
            "SKIP_PERM 的理由需重核" % (sys.platform, oct(got)))


def test_p1_符号链接量具能力探测(tmp_path):
    """P1：在真实临时目录接受有效链接或已知的 Windows 权限不足。"""
    real = _sentinel(tmp_path / "real.txt")
    _try_file_symlink(real, tmp_path / "link.txt")


def test_p1_仅已知Windows权限错误可跳过(monkeypatch, tmp_path):
    """未知错误不能被当成平台能力缺失而隐藏。"""
    target = tmp_path / "real.txt"
    _sentinel(target)

    monkeypatch.setattr(sys, "platform", "win32")
    denied = OSError("symbolic link privilege required")
    denied.winerror = 1314
    monkeypatch.setattr(os, "symlink", lambda *_args: (_ for _ in ()).throw(denied))
    assert _try_file_symlink(target, tmp_path / "denied.txt") is False

    legacy_denied = OSError(22, "客户端没有所需的特权")
    monkeypatch.setattr(os, "symlink", lambda *_args: (_ for _ in ()).throw(legacy_denied))
    assert _try_file_symlink(target, tmp_path / "legacy-denied.txt") is False

    unexpected = OSError("unexpected link creation failure")
    monkeypatch.setattr(os, "symlink", lambda *_args: (_ for _ in ()).throw(unexpected))
    with pytest.raises(OSError, match="unexpected link creation failure"):
        _try_file_symlink(target, tmp_path / "unexpected.txt")


def test_p2_权限继承机制链在本平台可端到端观测():
    """P2：正对照——「chmod 临时件 → os.replace → 目标取该 mode」这条**机制链**
    在本平台用可表达的权限档（只读位 0o444）是**真能端到端读到**的。

    即：A5/B5/C5 在 Windows 上 SKIP 的成因是「0o600 这一档不可表达」，
    **不是**「这条通路在本机测不了 / 量具失效」。若本断言红了，说明本平台连
    只读位都传不过去 ⇒ SKIP_PERM 的覆盖范围必须重核。
    """
    td = tempfile.mkdtemp(prefix="b498_p2_")
    target = os.path.join(td, "fresh.txt")          # 目标**不存在**（Windows 上
    tmp = os.path.join(td, "fresh.txt.tmp")         # 不能对已存在只读档做 replace）
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("new")
    os.chmod(tmp, 0o444)
    assert stat.S_IMODE(os.stat(tmp).st_mode) == 0o444, "P2 前置：临时件 chmod 未生效"

    os.replace(tmp, target)

    assert stat.S_IMODE(os.stat(target).st_mode) == 0o444, (
        "P2 机制链在本平台不可观测：临时件 chmod(0o444) 后 os.replace 到目标，"
        "目标 mode=%s（期望 0o444）——SKIP_PERM 的覆盖范围需重核"
        % oct(stat.S_IMODE(os.stat(target).st_mode)))


def test_p3_realpath量具非恒等_本平台有效():
    """P3：正对照——证明 A4/B4/C4 不是「因 realpath 恒等于入参」而假绿。

    A4/B4/C4 用「带 `..` 的路径」跨平台代理「穿透符号链接」这条腿。若本平台的
    `os.path.realpath` 是恒等函数，那三条断言就会**恒绿**、毫无判别力。本断言
    钉住：①`realpath` 确实把含 `..` 的串规范化成不同串；②规范化结果稳定；
    ③`..` 规范化后的串**不含** `..`（A4 断言的前置条件本身可满足）。
    """
    td = tempfile.mkdtemp(prefix="b498_p3_")
    sub = os.path.join(td, "sub")
    os.makedirs(sub)
    raw = os.path.join(sub, "..", "f.json")
    real = os.path.realpath(raw)

    assert real != raw, (
        "P3 量具失效：本平台 realpath 对含 `..` 的路径是恒等函数 ⇒ "
        "A4/B4/C4 恒绿、无判别力（实测 %r -> %r）" % (raw, real))
    assert ".." not in real, "P3 realpath 未消掉 `..`：%r" % (real,)
    assert os.path.realpath(raw) == real, "P3 realpath 结果不稳定（非纯函数）"
    assert os.path.normcase(real) == os.path.normcase(os.path.join(td, "f.json")), (
        "P3 realpath 规范化结果不是预期目录下的 f.json：%r" % (real,))


# ==================== A：export_all ====================

def test_a1_目标不存在_正常创建且不继承(monkeypatch):
    """A1（反向腿）：目标不存在 ⇒ 照旧落盘、不误加 chmod（保持 umask 默认）。"""
    td = tempfile.mkdtemp(prefix="b498_a1_")
    path = os.path.join(td, "fresh.json")
    calls = _spy(monkeypatch)

    res = _do_export_all(path)

    assert os.path.exists(path), "A1 产物未落盘"
    with open(path, encoding="utf-8") as f:
        json.load(f)
    assert res["path"] == path and res["exported_nodes"] == 1, \
        "A1 返回值口径变了：%r" % (res,)
    assert _tmp_chmods(calls, path) == [], (
        "A1 目标不存在却对临时件做了 chmod ⇒ 新建备份不再取 umask 默认：%r"
        % (_tmp_chmods(calls, path),))
    assert [n for n in os.listdir(td) if ".tmp-" in n] == [], "A1 残留临时件"


def test_a2_普通目标_覆盖式语义与返回值不变(monkeypatch):
    """A2（反向腿）：普通已存在目标 ⇒ 覆盖、JSON 可解析、无残件（#330 D/#698 契约）。"""
    td = tempfile.mkdtemp(prefix="b498_a2_")
    path = _sentinel(os.path.join(td, "over.json"))
    _spy(monkeypatch)

    res = _do_export_all(path)

    raw = open(path, "rb").read()
    assert b"original" not in raw and b'{"old": true}' not in raw, "A2 覆盖未发生"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert len(data["nodes"]) == 1 and res["overwrote_existing"] is True, \
        "A2 覆盖/上报口径变了：%r" % (res,)
    assert [n for n in os.listdir(td) if ".tmp-" in n] == [], "A2 残留临时件"


def test_a3_继承旧档权限位_export_all(monkeypatch):
    """A3 ★：旧档权限位被拷到临时件（打桩 `os.stat` 报 0o600 ⇒ 须 `chmod(tmp, 0o600)`）。"""
    td = tempfile.mkdtemp(prefix="b498_a3_")
    path = _sentinel(os.path.join(td, "bk.json"))
    calls = _spy(monkeypatch)
    _fake_mode(monkeypatch, path, 0o600)

    _do_export_all(path)

    modes = [m for _, m in _tmp_chmods(calls, path)]
    assert modes == [0o600], (
        "#498 export_all 未继承旧档权限位：临时件 chmod 记录=%r（期望 [0o600]）"
        "——旧实现截断写同一 inode 会保留 0600，缺陷态则被放宽成 0644"
        % (modes,))


def test_a4_落盘目标是realpath_export_all(monkeypatch):
    """A4 ★：`os.replace` 的 dst 是 realpath（带 `..` 的路径被规范化）⇒ 穿透符号链接。"""
    td = tempfile.mkdtemp(prefix="b498_a4_")
    sub = os.path.join(td, "sub")
    os.makedirs(sub)
    path = os.path.join(sub, "..", "bk.json")           # 含 `..`：realpath 后不同串
    calls = _spy(monkeypatch)

    _do_export_all(path)

    want = os.path.realpath(path)
    assert ".." not in want, "A4 前置：探针路径未被规范化（%r）" % (want,)
    dsts = _replace_dsts(calls)
    assert dsts, "A4 未观测到 os.replace 调用"
    assert any(os.path.normcase(d) == os.path.normcase(want) for d in dsts), (
        "#498 export_all 落盘目标不是 realpath：os.replace dst=%r（期望 %r）"
        "——缺陷态直接把 `..` 原串交给 os.replace，目标是符号链接时链接本身被替换"
        % (dsts, want))


@SKIP_PERM
def test_a5_posix_0600目标写后仍0600():
    """A5 [posix]：真权限面——0600 备份经 export_all 后仍 0600（缺陷态变 0644）。"""
    td = tempfile.mkdtemp(prefix="b498_a5_")
    path = _sentinel(os.path.join(td, "bk.json"))
    os.chmod(path, 0o600)
    before = stat.S_IMODE(os.stat(path).st_mode)

    _do_export_all(path)

    after = stat.S_IMODE(os.stat(path).st_mode)
    assert before == 0o600 and after == 0o600, (
        "#498 export_all 把 0600 备份放宽成 %s（before=%s）——同机其他用户可读"
        % (oct(after), oct(before)))


def test_a6_符号链接目标保持且真实件更新(tmp_path):
    """A6：目标是符号链接 ⇒ 写后仍是链接，且链接指向的真实件被更新。"""
    td = str(tmp_path)
    real_dir = os.path.join(td, "offsite")
    os.makedirs(real_dir)
    real = _sentinel(os.path.join(real_dir, "latest.json"))
    link = os.path.join(td, "backup_link.json")
    _require_file_symlink(real, link)

    _do_export_all(link)

    assert os.path.islink(link), (
        "#498 export_all 把符号链接换成了普通文件 ⇒ 真实备份位置不再更新")
    with open(real, encoding="utf-8") as f:
        data = json.load(f)                     # 真实件必须是新的完整 JSON
    assert len(data["nodes"]) == 1, "A6 真实备份未被更新（仍是旧内容）"


def test_a7_换入失败_旧档不动且不留残件(monkeypatch):
    """A7 ★反向腿：`os.replace` 失败（权限继承 chmod 已发生）⇒ 旧档逐字不变、
    临时件被清理干净（**跨平台**：Windows 上只读临时件删不掉 ⇒ 缺陷态必红）。

    这一条钉住的是本件修复**自己**的反向腿：`chmod(tmp, 旧档mode)` 之后若清理
    仍按老办法直接 `os.remove`，Windows 上（旧档带只读位时）会留下 `.tmp-` 残件。
    """
    td = tempfile.mkdtemp(prefix="b498_a7_")
    path = _sentinel(os.path.join(td, "bk.json"))
    before = open(path, "rb").read()

    def _boom(src, dst, **kw):
        raise PermissionError(13, "模拟 os.replace 换入失败（#498 探针）")

    _spy(monkeypatch)
    _fake_mode(monkeypatch, path, 0o444)     # 只读位：Windows 上删不掉
    monkeypatch.setattr(os, "replace", _boom)

    with pytest.raises(PermissionError):
        _do_export_all(path)

    assert open(path, "rb").read() == before, "A7 换入失败却动了旧档"
    residue = [n for n in os.listdir(td) if ".tmp-" in n]
    assert residue == [], (
        "A7 换入失败后残留临时件 %r（权限继承 chmod 后未恢复可写再删）" % (residue,))


# ==================== B：save_consolidated ====================

def test_b1_目标不存在_正常创建且不继承(monkeypatch):
    """B1（反向腿）：save_consolidated 目标不存在 ⇒ 照旧落盘、不误加 chmod。"""
    td = tempfile.mkdtemp(prefix="b498_b1_")
    path = os.path.join(td, "kern.json")
    calls = _spy(monkeypatch)

    ret = _do_save_consolidated(path)

    assert ret == path, "B1 返回值口径变了：%r" % (ret,)
    with open(path, encoding="utf-8") as f:
        assert json.load(f)["consolidated"] is True
    assert _tmp_chmods(calls, path) == [], \
        "B1 目标不存在却 chmod 了临时档：%r" % (_tmp_chmods(calls, path),)
    assert [n for n in os.listdir(td) if ".tmp-" in n] == [], "B1 残留临时档"


def test_b2_普通目标_覆盖式语义不变(monkeypatch):
    """B2（反向腿）：普通已存在目标 ⇒ 覆盖、可 json.load、无残件。"""
    td = tempfile.mkdtemp(prefix="b498_b2_")
    path = _sentinel(os.path.join(td, "kern.json"))
    _spy(monkeypatch)

    _do_save_consolidated(path)

    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["consolidated"] is True and payload["note"] == "x", \
        "B2 覆盖式语义丢失：%r" % (sorted(payload),)
    assert [n for n in os.listdir(td) if ".tmp-" in n] == [], "B2 残留临时档"


def test_b3_继承旧档权限位_save_consolidated(monkeypatch):
    """B3 ★：旧档权限位被拷到临时档（打桩 0o600 ⇒ 须 `chmod(tmp, 0o600)`）。"""
    td = tempfile.mkdtemp(prefix="b498_b3_")
    path = _sentinel(os.path.join(td, "kern.json"))
    calls = _spy(monkeypatch)
    _fake_mode(monkeypatch, path, 0o600)

    _do_save_consolidated(path)

    modes = [m for _, m in _tmp_chmods(calls, path)]
    assert modes == [0o600], (
        "#498 save_consolidated 未继承旧档权限位：临时档 chmod 记录=%r（期望 [0o600]）"
        % (modes,))


def test_b4_落盘目标是realpath_save_consolidated(monkeypatch):
    """B4 ★：`os.replace` 的 dst 是 realpath（`..` 被规范化）。"""
    td = tempfile.mkdtemp(prefix="b498_b4_")
    sub = os.path.join(td, "sub")
    os.makedirs(sub)
    path = os.path.join(sub, "..", "kern.json")
    calls = _spy(monkeypatch)

    _do_save_consolidated(path)

    want = os.path.realpath(path)
    dsts = _replace_dsts(calls)
    assert dsts and any(os.path.normcase(d) == os.path.normcase(want) for d in dsts), (
        "#498 save_consolidated 落盘目标不是 realpath：os.replace dst=%r（期望 %r）"
        % (dsts, want))


@SKIP_PERM
def test_b5_posix_0600目标写后仍0600():
    """B5 [posix]：真权限面——0600 固化档经 save_consolidated 后仍 0600。"""
    td = tempfile.mkdtemp(prefix="b498_b5_")
    path = _sentinel(os.path.join(td, "kern.json"))
    os.chmod(path, 0o600)

    _do_save_consolidated(path)

    after = stat.S_IMODE(os.stat(path).st_mode)
    assert after == 0o600, "#498 save_consolidated 把 0600 放宽成 %s" % oct(after)


def test_b6_符号链接目标保持且真实件更新(tmp_path):
    """B6：符号链接目标 ⇒ 写后仍是链接且真实件已更新。"""
    td = str(tmp_path)
    real_dir = os.path.join(td, "offsite")
    os.makedirs(real_dir)
    real = _sentinel(os.path.join(real_dir, "kern.json"))
    link = os.path.join(td, "kern_link.json")
    _require_file_symlink(real, link)

    _do_save_consolidated(link)

    assert os.path.islink(link), "#498 save_consolidated 把符号链接换成了普通档"
    with open(real, encoding="utf-8") as f:
        assert json.load(f)["consolidated"] is True, "B6 真实档未被更新"


def test_b7_换入失败_旧档不动且不留残件(monkeypatch):
    """B7 ★反向腿：`os.replace` 失败 ⇒ 旧档逐字不变、临时档清理干净（跨平台）。"""
    td = tempfile.mkdtemp(prefix="b498_b7_")
    path = _sentinel(os.path.join(td, "kern.json"))
    before = open(path, "rb").read()

    def _boom(src, dst, **kw):
        raise PermissionError(13, "模拟 os.replace 换入失败（#498 探针）")

    _spy(monkeypatch)
    _fake_mode(monkeypatch, path, 0o444)
    monkeypatch.setattr(os, "replace", _boom)

    with pytest.raises(PermissionError):
        _do_save_consolidated(path)

    assert open(path, "rb").read() == before, "B7 换入失败却动了旧档"
    residue = [n for n in os.listdir(td) if ".tmp-" in n]
    assert residue == [], (
        "B7 换入失败后残留临时档 %r（权限继承 chmod 后未恢复可写再删）" % (residue,))


# ==================== C：flush_payloads ====================

def test_c1_目标不存在_正常创建且不继承(monkeypatch):
    """C1（反向腿）：flush_payloads 目标不存在 ⇒ 照旧落盘、不误加 chmod。"""
    td = tempfile.mkdtemp(prefix="b498_c1_")
    path = os.path.join(td, "out.md")
    calls = _spy(monkeypatch)

    res = _do_flush_payloads(path)

    assert res.get("written") == path, "C1 written 口径变了：%r" % (res.get("written"),)
    assert open(path, encoding="utf-8").read().startswith("---"), "C1 产物内容不对"
    assert _tmp_chmods(calls, path) == [], \
        "C1 目标不存在却 chmod 了临时件：%r" % (_tmp_chmods(calls, path),)
    assert [n for n in os.listdir(td) if ".tmp-" in n] == [], "C1 残留临时件"


def test_c2_普通目标_覆盖式语义不变(monkeypatch):
    """C2（反向腿）：普通已存在目标 ⇒ 覆盖（#324 覆盖式语义保留）、无残件。"""
    td = tempfile.mkdtemp(prefix="b498_c2_")
    path = _sentinel(os.path.join(td, "out.md"), body="OLD\n")
    _spy(monkeypatch)

    _do_flush_payloads(path)

    body = open(path, encoding="utf-8").read()
    assert body.startswith("---") and "OLD" not in body, "C2 覆盖式语义丢失"
    assert [n for n in os.listdir(td) if ".tmp-" in n] == [], "C2 残留临时件"


def test_c3_继承旧档权限位_flush_payloads(monkeypatch):
    """C3 ★：旧档权限位被拷到临时件（打桩 0o600 ⇒ 须 `chmod(tmp, 0o600)`）。"""
    td = tempfile.mkdtemp(prefix="b498_c3_")
    path = _sentinel(os.path.join(td, "out.md"), body="OLD\n")
    calls = _spy(monkeypatch)
    _fake_mode(monkeypatch, path, 0o600)

    _do_flush_payloads(path)

    modes = [m for _, m in _tmp_chmods(calls, path)]
    assert modes == [0o600], (
        "#498 flush_payloads 未继承旧档权限位：临时件 chmod 记录=%r（期望 [0o600]）"
        % (modes,))


def test_c4_落盘目标是realpath_flush_payloads(monkeypatch):
    """C4 ★：`os.replace` 的 dst 是 realpath（`..` 被规范化）。"""
    td = tempfile.mkdtemp(prefix="b498_c4_")
    sub = os.path.join(td, "sub")
    os.makedirs(sub)
    path = os.path.join(sub, "..", "out.md")
    calls = _spy(monkeypatch)

    _do_flush_payloads(path)

    want = os.path.realpath(path)
    dsts = _replace_dsts(calls)
    assert dsts and any(os.path.normcase(d) == os.path.normcase(want) for d in dsts), (
        "#498 flush_payloads 落盘目标不是 realpath：os.replace dst=%r（期望 %r）"
        % (dsts, want))


@SKIP_PERM
def test_c5_posix_0600目标写后仍0600():
    """C5 [posix]：真权限面——0600 目标经 flush_payloads 后仍 0600。"""
    td = tempfile.mkdtemp(prefix="b498_c5_")
    path = _sentinel(os.path.join(td, "out.md"), body="OLD\n")
    os.chmod(path, 0o600)

    _do_flush_payloads(path)

    after = stat.S_IMODE(os.stat(path).st_mode)
    assert after == 0o600, "#498 flush_payloads 把 0600 放宽成 %s" % oct(after)


def test_c6_符号链接目标保持且真实件更新(tmp_path):
    """C6：符号链接目标 ⇒ 写后仍是链接且真实件已更新。"""
    td = str(tmp_path)
    real_dir = os.path.join(td, "offsite")
    os.makedirs(real_dir)
    real = _sentinel(os.path.join(real_dir, "out.md"), body="OLD\n")
    link = os.path.join(td, "out_link.md")
    _require_file_symlink(real, link)

    _do_flush_payloads(link)

    assert os.path.islink(link), "#498 flush_payloads 把符号链接换成了普通文件"
    body = open(real, encoding="utf-8").read()
    assert body.startswith("---") and "OLD" not in body, "C6 真实产物未被更新"


def test_c7_换入失败_旧档不动且不留残件(monkeypatch):
    """C7 ★反向腿：`os.replace` 失败 ⇒ 旧档逐字不变、临时件清理干净（跨平台）。"""
    td = tempfile.mkdtemp(prefix="b498_c7_")
    path = _sentinel(os.path.join(td, "out.md"), body="OLD\n")
    before = open(path, "rb").read()

    def _boom(src, dst, **kw):
        raise PermissionError(13, "模拟 os.replace 换入失败（#498 探针）")

    _spy(monkeypatch)
    _fake_mode(monkeypatch, path, 0o444)
    monkeypatch.setattr(os, "replace", _boom)

    with pytest.raises(PermissionError):
        _do_flush_payloads(path)

    assert open(path, "rb").read() == before, "C7 换入失败却动了旧档"
    residue = [n for n in os.listdir(td) if ".tmp-" in n]
    assert residue == [], (
        "C7 换入失败后残留临时件 %r（权限继承 chmod 后未恢复可写再删）" % (residue,))
