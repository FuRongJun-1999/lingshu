# -*- coding: utf-8 -*-
"""#194 守卫：make_multimodal_scene(n_objects>9) 必须显式报错，绝不许挂起。

判据（卡内三条验收的可执行化）：
  ① n=9 返回 9 个对象，位置互不重复；
  ② n=10 在超时内**抛明确 ValueError**（不是返回、更不是挂起）；
  ③ 多种子下 n=10 行为一致（排除「运气差」）；
  ④ 回归：仓内真实用法 n=1 / n=2 行为不变；
  ⑤ n=0 / 负数同样显式报错，不静默返回空场景。

★ 为什么 n>9 那几条要起子进程：修复前它会**无限挂起**，
在自己的进程里直接调用只会把 pytest 一起挂死（表现为「卡住」而不是「变红」）。
起子进程 + 超时，才能把它判成一次明确的失败。

★ 为什么子进程代码落在 `tests/_issue194_child_probe.py` 而不是内联字符串：
曾经内联成单行传 `python -c`，`try: ...; except ...:` 不能共处一个逻辑行
⇒ SyntaxError ⇒ 子进程立刻 rc=1 且 stdout 为空。若审判词匹配写得松，
守卫会在**回滚源码后也「通过」**（假通过）。本文件已按「真文件 + 显式审判词」重写。

判别力：只回滚源码（hex_text.py）而不动本文件 ⇒ 必须变红。
运行: python -X utf8 -m pytest tests/test_issue194_make_multimodal_scene_capacity.py -v
"""
import os as _os
import subprocess as _sp
import sys as _sys

_RP = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _RP)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lingshu.nn import hex_text as ht  # noqa: E402

TIMEOUT_S = 15
_PROBE = _os.path.join(_RP, 'tools', '_issue194_child_probe.py')

assert _os.path.isfile(_PROBE), '子进程探针缺失: %s' % _PROBE


def _run_child(n, seed=7):
    """在子进程里跑 n；返回 (rc, out, hung)。"""
    try:
        p = _sp.run([_sys.executable, '-X', 'utf8', _PROBE, str(n), str(seed)],
                    capture_output=True, text=True, timeout=TIMEOUT_S,
                    cwd=_RP, encoding='utf-8', errors='replace')
        return p.returncode, (p.stdout or '') + (p.stderr or ''), False
    except _sp.TimeoutExpired:
        return None, '', True


def test_probe_itself_is_usable():
    """元测试：探针必须能对 n=1 打印 NO_ERROR。

    否则「n=10 抛 ValueError」可能只是探针坏了（例如语法错误、导入失败），
    而不是被测行为正确 —— 这正是先前假通过的形态。
    """
    rc, out, hung = _run_child(1)
    assert not hung, '探针在 n=1 上挂起，说明探针或被测代码有问题'
    assert rc == 0, '探针 rc=%r 输出=%r' % (rc, out[-300:])
    assert 'NO_ERROR' in out, '探针未打印 NO_ERROR，输出=%r' % (out[-300:],)


def test_n9_returns_nine_objects():
    """① n=9 正常返回 9 个对象，位置互不重复。"""
    img, meta = ht.make_multimodal_scene(np.random.default_rng(7), 16, 9)
    labels = meta['labels']
    assert len(labels) == 9
    assert img.shape == (16, 16, 3)
    poses = [o['pos'] for o in labels]
    assert len(set(poses)) == 9, '位置必须互不重复: %r' % (sorted(poses),)


def test_n10_raises_instead_of_hanging():
    """② n=10 必须显式抛 ValueError，绝不挂起（子进程 + 超时判定）。"""
    rc, out, hung = _run_child(10)
    assert not hung, 'n=10 在 %ds 内未返回 ⇒ 挂起（#194 未修）' % TIMEOUT_S
    assert rc == 0, '探针本身没跑完 rc=%r 输出=%r' % (rc, out[-300:])
    assert 'RAISED_OTHER' not in out, '抛的应当是 ValueError: %r' % (out[-300:],)
    assert 'RAISED_VALUEERROR' in out, (
        'n=10 应当抛明确 ValueError，实际输出=%r' % (out[-300:],))
    assert 'MSG:' in out, '错误消息应当可读（便于调用方定位）: %r' % (out[-300:],)


@pytest.mark.parametrize('seed', [1, 42, 2024])
def test_n10_consistent_across_seeds(seed):
    """③ 多种子下 n=10 行为一致（排除「只是运气差」）。"""
    rc, out, hung = _run_child(10, seed=seed)
    assert not hung, 'seed=%d 的 n=10 挂起' % seed
    assert 'RAISED_VALUEERROR' in out, 'seed=%d 未抛 ValueError: %r' % (seed, out[-300:])


def test_in_repo_usage_unchanged():
    """④ 回归：仓内真实用法 n=1 / n=2 行为不变。"""
    for n in (1, 2):
        img, meta = ht.make_multimodal_scene(np.random.default_rng(11), 48, n)
        assert len(meta['labels']) == n
        assert img.shape == (48, 48, 3)
        assert meta['description'], '描述不应为空'


def test_invalid_low_n_rejected():
    """⑤ 边界：n=0 / 负数也应当显式报错，而不是静默返回空场景。"""
    for bad in (0, -1):
        with pytest.raises(ValueError):
            ht.make_multimodal_scene(np.random.default_rng(3), 16, bad)