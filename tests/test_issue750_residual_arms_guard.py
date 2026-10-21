# -*- coding: utf-8 -*-
"""test_issue750_residual_arms_guard · #750 残余面四臂矩阵（A/B/D/E 全拒）
============================================================================
背景（lingshu issue #750 残余 · EQRS #817 独立复跑口径）：
  #782（merge 171a665）只关掉「受控面有同名实现」的 B 臂；A/D/E 三臂经
  `_ComponentFinder.find_spec` 受控面未命中时的「宿主帧 `return None`」把整条
  `sys.path`（含 cwd / 脚本目录条目）交还标准 `PathFinder`，毒白名单组件仍在
  引擎进程内执行。四臂定义（#817 §一，与维护侧 MERGE-782-750-RESIDUAL §一
  的臂号一致）：

    A：逃生口开 + cwd == 脚本目录（victim）      —— 改前：毒件执行
    B：逃生口关 + cwd == 脚本目录                —— 改前：ModuleNotFoundError
    D：逃生口关 + 脚本目录 ≠ cwd（sys.path[0]=脚本目录）—— 改前：毒件执行
    E：逃生口开 + 脚本目录 ≠ cwd                 —— 改前：毒件执行

修复（EQRS #817 处置方向①，维护侧登记的「收口点」）：白名单名的解析面只剩
`safe_search_path()`——受控面未命中即按「组件缺失」失败（原生同文本
`ModuleNotFoundError` + 告警），不再交还 `PathFinder`，与调用方、与逃生口状态
均无关。方向②（脚本目录不解析白名单名）由本收口在组件层落实（`_pathguard`
对**非白名单名**仍维持 #421 边界——脚本目录≈宿主自有代码面，全面拒绝会拆掉
宿主相邻模块导入，G7 钉之）。方向③（`_brain_env` 黑名单拒传）见 G8。

判据来源：issue #750 正文 payload（毒 body.py 只写标记，不读密钥——同 #817
诚实边界）；#817 §一四臂读数（A/D/E `POISON_BODY_EXECUTED`、B 拒绝）；
维护侧 MERGE-782-750-RESIDUAL §三/§四（残余面逐条 + 「受控面未命中时宿主帧
该怎么办」收口点）。取舍披露：#275 的「宿主自有同名模块必须可解析」按名收窄
（两条支持径见 G5/G6/G9；tests/test_issue275_component_finder_host_scope.py
G1/G2 已按同口径改钉）。

断言组（回退 / 放宽即红）：
  G1-G4 四臂矩阵：修后 A/B/D/E 全部 `IMPORT_BODY ModuleNotFoundError`、
     MARKER 不存在（毒模块零执行）。
  G5 支持径①（受控面）：宿主自有 `body.py` 经 `LINGSHU_COMPONENT_ROOT` 声明
     ⇒ 宿主帧 `import body` ok owner=host（#275 收窄后的正路活着）。
  G6 支持径②（早导入）：宿主**先** `import body` 再 `import lingshu.core.core`
     ⇒ ok owner=host（sys.modules 天然最优先，finder 不干预）。
  G7 正常宿主导入不受影响：**非白名单**名（myhostmod）在脚本目录（≠cwd、
     逃生口关）仍可解析——证明本修复未把 #421 边界扩成全面脚本目录拒绝。
  G8 env 黑名单拒传（方向③）：`_brain_env` 不透传 LINGSHU_ALLOW_CWD_IMPORTS /
     LINGSHU_COMPONENT_ROOT；且 deny 优先——模拟未来白名单误收二者仍拒。
  G9 定点变异自证：复刻旧实现（受控面未命中 + 宿主帧 ⇒ return None）后，
     A/D/E 三臂毒模块**必执行**（MARKER 存在）——证明本守卫有判别力。

运行（仓根）：python -X utf8 tests/test_issue750_residual_arms_guard.py
退出码：0 = 全过；1 = 有断言失败
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

WL_NAME = "body"             # 白名单名（COMPONENT_NAMES 成员）
NON_WL = "myhostmod"         # 非白名单名（正常宿主导入对照）

_POISON = ("import os as _os\n"
           "POISON = True\n"
           "OWNER = 'poison'\n"
           "open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"
           " 'MARKER.txt'), 'w').write('EXECUTED')\n")

_MAIN = (
    "import lingshu.core.core\n"
    "try:\n"
    "    import body\n"
    "    print('IMPORT_BODY ok owner=%%s poison=%%s' %% (getattr(body, 'OWNER', None), getattr(body, 'POISON', None)))\n"
    "except ModuleNotFoundError:\n"
    "    print('IMPORT_BODY ModuleNotFoundError')\n"
    "try:\n"
    "    import %s\n"
    "    print('IMPORT_NONWL ok owner=%%s' %% getattr(%s, 'OWNER', None))\n"
    "except ModuleNotFoundError:\n"
    "    print('IMPORT_NONWL ModuleNotFoundError')\n"
) % (NON_WL, NON_WL)

# 变异体（G9）：复刻 #782 时代实现——受控面未命中 + 宿主帧 ⇒ return None
_BROKEN_PRELUDE = (
    "import sys\n"
    "import lingshu.core.core as _c\n"
    "import lingshu.core.component_resolver as _cr\n"
    "class _BrokenFinder(_cr._ComponentFinder):\n"
    "    def find_spec(self, fullname, path=None, target=None):\n"
    "        if fullname not in _cr.COMPONENT_NAMES:\n"
    "            return None\n"
    "        roots = _cr.safe_search_path()\n"
    "        spec = __import__('importlib.machinery', fromlist=['PathFinder']).PathFinder.find_spec(fullname, roots)\n"
    "        if spec is not None:\n"
    "            return spec\n"
    "        if not _cr._importer_is_lingshu():\n"
    "            return None\n"
    "        return _cr._ComponentFinder.find_spec(self, fullname, path, target)\n"
    "sys.meta_path[:] = [_BrokenFinder() if isinstance(f, _cr._ComponentFinder) else f\n"
    "                    for f in sys.meta_path]\n"
)


def _mkvictim(prefix, body_src=None, with_nonwl=False):
    """victim 目录：毒 body.py + main.py（+ 可选非白名单对照模块）。"""
    v = tempfile.mkdtemp(prefix=prefix)
    with open(os.path.join(v, WL_NAME + ".py"), "w", encoding="utf-8") as f:
        f.write(body_src if body_src is not None else _POISON)
    with open(os.path.join(v, "main.py"), "w", encoding="utf-8") as f:
        f.write(_MAIN)
    if with_nonwl:
        with open(os.path.join(v, NON_WL + ".py"), "w", encoding="utf-8") as f:
            f.write("OWNER = 'host'\n")
    return v


def _run(main_dir, hatch, same_cwd, extra_env=None, prelude=""):
    cwd = main_dir if same_cwd else tempfile.mkdtemp(prefix=main_dir.split(os.sep)[-1] + "_cwd_")
    env = {k: v for k, v in os.environ.items()
           if k not in ("LINGSHU_ALLOW_CWD_IMPORTS", "LINGSHU_COMPONENT_ROOT")}
    env.update({"PYTHONPATH": REPO, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    if hatch:
        env["LINGSHU_ALLOW_CWD_IMPORTS"] = "1"
    if extra_env:
        env.update(extra_env)
    main = os.path.join(main_dir, "main.py")
    if prelude:
        with open(main, "w", encoding="utf-8") as f:
            f.write(prelude + _MAIN)
    try:
        return subprocess.run([sys.executable, "-B", "-W", "ignore", main],
                              cwd=cwd, env=env, capture_output=True,
                              text=True, encoding="utf-8")
    finally:
        if prelude:
            with open(main, "w", encoding="utf-8") as f:
                f.write(_MAIN)


def _line(out, tag):
    for ln in out.splitlines():
        if ln.startswith(tag):
            return ln
    return "<missing>"


def _assert_arm(label, r, v, expect_refused=True):
    assert r.returncode == 0, "%s: rc=%d\nstderr: %s" % (label, r.returncode, r.stderr[-500:])
    ln = _line(r.stdout, "IMPORT_BODY")
    marker = os.path.exists(os.path.join(v, "MARKER.txt"))
    if expect_refused:
        assert "ModuleNotFoundError" in ln, (
            "%s: 白名单名未按「组件缺失」失败：%r" % (label, ln))
        assert not marker, "%s: 毒模块被执行（MARKER.txt 存在）——#750 残余未收口" % label
    else:  # 变异自证用：旧实现下必须执行
        assert ("owner=poison" in ln) or marker, (
            "%s: 变异体未复现缺陷（判别力不足）：%r marker=%s" % (label, ln, marker))


# ---------------------------------------------------------------- 四臂矩阵
def test_g1_arm_a_escape_open_cwd_eq_scriptdir():
    """A：逃生口开 + cwd==脚本目录 ⇒ 毒件不执行（改前：执行）。"""
    v = _mkvictim("f750r_A_")
    _assert_arm("A", _run(v, hatch=True, same_cwd=True), v)


def test_g2_arm_b_escape_closed_cwd_eq_scriptdir():
    """B：逃生口关 + cwd==脚本目录 ⇒ 维持拒绝（改前即拒绝，回归钉）。"""
    v = _mkvictim("f750r_B_")
    _assert_arm("B", _run(v, hatch=False, same_cwd=True), v)


def test_g3_arm_d_escape_closed_scriptdir_ne_cwd():
    """D：逃生口关 + 脚本目录≠cwd（sys.path[0]=脚本目录）⇒ 毒件不执行（改前：执行）。"""
    v = _mkvictim("f750r_D_")
    _assert_arm("D", _run(v, hatch=False, same_cwd=False), v)


def test_g4_arm_e_escape_open_scriptdir_ne_cwd():
    """E：逃生口开 + 脚本目录≠cwd ⇒ 毒件不执行（改前：执行）。"""
    v = _mkvictim("f750r_E_")
    _assert_arm("E", _run(v, hatch=True, same_cwd=False), v)


# ---------------------------------------------------------------- 支持径
def test_g5_controlled_root_host_body_importable():
    """G5 支持径①：宿主自有 body.py 经 LINGSHU_COMPONENT_ROOT 声明 ⇒ ok owner=host。"""
    v = _mkvictim("f750r_G5_", body_src="OWNER = 'host'\nPOISON = False\n")
    r = _run(v, hatch=False, same_cwd=True, extra_env={"LINGSHU_COMPONENT_ROOT": v})
    assert r.returncode == 0, r.stderr[-500:]
    ln = _line(r.stdout, "IMPORT_BODY")
    assert "owner=host poison=False" in ln, "G5: 受控面支持径被修坏：%r" % ln
    assert not os.path.exists(os.path.join(v, "MARKER.txt"))


def test_g6_early_import_host_body_survives():
    """G6 支持径②：先 import body 再 import lingshu.core ⇒ sys.modules 命中，不受影响。"""
    v = _mkvictim("f750r_G6_", body_src="OWNER = 'host_early'\nPOISON = False\n")
    prelude = "import body  # 早于 lingshu.core（sys.modules 天然最优先）\n"
    r = _run(v, hatch=True, same_cwd=True, prelude=prelude)
    assert r.returncode == 0, r.stderr[-500:]
    ln = _line(r.stdout, "IMPORT_BODY")
    assert "owner=host_early" in ln, "G6: 早导入径被修坏：%r" % ln


def test_g7_non_whitelist_scriptdir_module_unaffected():
    """G7：非白名单名在脚本目录（≠cwd、逃生口关）仍可解析——#421 边界未被扩成全面拒绝。"""
    v = _mkvictim("f750r_G7_", with_nonwl=True)
    r = _run(v, hatch=False, same_cwd=False)
    assert r.returncode == 0, r.stderr[-500:]
    ln = _line(r.stdout, "IMPORT_NONWL")
    assert "owner=host" in ln, (
        "G7: 正常宿主导入（非白名单名）被误伤——脚本目录被全面拒绝？%r\n%s" % (ln, r.stdout))


# ---------------------------------------------------------------- env 黑名单
def test_g8_brain_env_denies_hatch_and_component_root():
    """G8（方向③）：_brain_env 拒传两名字；deny 优先于（被模拟放宽的）allow。"""
    from lingshu.world import brain_store as BS
    old_allow = BS._ENV_ALLOW_NAMES
    old = {k: os.environ.get(k) for k in
           ("LINGSHU_ALLOW_CWD_IMPORTS", "LINGSHU_COMPONENT_ROOT", "LINGSHU_KB")}
    try:
        os.environ["LINGSHU_ALLOW_CWD_IMPORTS"] = "1"
        os.environ["LINGSHU_COMPONENT_ROOT"] = os.path.join(tempfile.gettempdir(), "evil")
        os.environ["LINGSHU_KB"] = os.path.join(tempfile.gettempdir(), "kb")
        env = BS._brain_env(None, None)
        assert "LINGSHU_ALLOW_CWD_IMPORTS" not in env, "G8: 逃生口被透传给脑侧子进程"
        assert "LINGSHU_COMPONENT_ROOT" not in env, "G8: 组件根被透传给脑侧子进程"
        assert "LINGSHU_KB" in env, "G8: 具名白名单（LINGSHU_KB）被误伤"
        # deny 优先：模拟未来白名单误收二者（前缀回归 / 具名误加）仍必须拒
        BS._ENV_ALLOW_NAMES = frozenset(set(old_allow) |
                                        {"LINGSHU_ALLOW_CWD_IMPORTS", "LINGSHU_COMPONENT_ROOT"})
        assert not BS._env_is_allowed("LINGSHU_ALLOW_CWD_IMPORTS"), (
            "G8: deny 未优先于 allow——黑名单形同虚设")
        assert not BS._env_is_allowed("LINGSHU_COMPONENT_ROOT")
        assert BS._env_is_allowed("LINGSHU_KB"), "G8: deny 面过宽（误伤正常白名单）"
    finally:
        BS._ENV_ALLOW_NAMES = old_allow
        for k, val in old.items():
            if val is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = val


# ---------------------------------------------------------------- 变异自证
def test_g9_mutation_old_host_fallback_executes_poison():
    """G9 定点变异：复刻旧实现（受控面未命中+宿主帧⇒return None）⇒ A/D/E 必执行。"""
    for label, hatch, same_cwd in (("A", True, True), ("D", False, False), ("E", True, False)):
        v = _mkvictim("f750r_G9%s_" % label)
        r = _run(v, hatch=hatch, same_cwd=same_cwd, prelude=_BROKEN_PRELUDE)
        assert r.returncode == 0, "G9-%s: rc=%d\n%s" % (label, r.returncode, r.stderr[-500:])
        _assert_arm("G9-%s" % label, r, v, expect_refused=False)


if __name__ == "__main__":
    _fails = 0
    for _name, _fn in sorted(list(globals().items())):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print("PASS", _name)
            except AssertionError as _e:
                _fails += 1
                print("FAIL", _name, str(_e)[:400])
    print("FAILED" if _fails else "ALL PASS")
    sys.exit(1 if _fails else 0)
