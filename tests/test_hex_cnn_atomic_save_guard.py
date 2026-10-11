#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""守卫：hex_cnn.save_consolidated 落盘必须是「原子替换」，不得「先截断再写」。

缺陷形态（修复前 lingshu/nn/hex_cnn.py:296-297）::

    with open(path, "w", encoding="utf-8") as f:   # "w" 先截断既有目标档
        json.dump(payload, f, ensure_ascii=False, indent=1)

`"w"` 在写之前就把已存在的目标档截断为 0 字节；一旦写入中途失败（典型：meta
含不可 JSON 序列化的对象 ⇒ json.dump 抛 TypeError），目标档已毁且只留下半截 /
空档 —— 旧数据丢失且新数据不完整（本件 #393：非原子写、先毁后写）。

判据（经验标定）：本件 #393（分诊表 P0·数据丢失）。本守卫断言三件事：
  ① 写失败后，既有目标档内容逐字节不变（旧档未被先毁）；
  ② 写失败后，目标档仍是完整的合法 JSON（绝不停留在空档 / 半截坏档）；
  ③ 写失败后，同目录不残留临时档（原子写的中间产物必须被清理）。
另附正向断言：正常写入产出可 json.load 的档，且 os.replace 已生效。

判别力说明：本守卫不覆盖「进程被 kill -9 恰在 os.replace 前」的断电场景
（需真故障注入，本件不做）；它钉住的是同一进程内、异常路径下的先毁后写。
"""
import json
import os
import sys
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from lingshu.nn.hex_cnn import DEFAULT_KERNELS, load_consolidated, save_consolidated


def _sentinel_dir():
    tmpd = tempfile.mkdtemp(prefix="hex_cnn_atomic_")
    path = os.path.join(tmpd, "kernels.json")
    sentinel = {"sentinel": "must-survive", "kernels": {"k": [0.0] * 7}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sentinel, f)
    return tmpd, path, sentinel


def _failed_write(path):
    """用不可序列化的 meta 触发写入中途失败，返回异常类型名。"""
    try:
        save_consolidated(DEFAULT_KERNELS, {"bad": object()}, path)
    except (TypeError, ValueError) as e:
        return type(e).__name__
    raise AssertionError("不可序列化 meta 竟未抛错——写入路径未走到失败分支")


def test_existing_file_survives_failed_write():
    """① 写失败后既有目标档逐字节不变（非原子写会先截断 ⇒ 变空/半截）。"""
    tmpd, path, sentinel = _sentinel_dir()
    before = open(path, "rb").read()
    _failed_write(path)
    after = open(path, "rb").read()
    assert after == before, (
        f"写失败后目标档被改动（先毁后写）：before={before[:80]!r} "
        f"after={after[:80]!r}")
    assert json.loads(after.decode("utf-8")) == sentinel


def test_no_truncated_corrupt_file_left():
    """② 写失败后目标档仍是完整合法 JSON（绝不留 0 字节 / 半截坏档）。"""
    tmpd, path, sentinel = _sentinel_dir()
    _failed_write(path)
    raw = open(path, "rb").read()
    assert len(raw) > 0, "写失败后目标档成了 0 字节空档（先截断后写）"
    payload = json.loads(raw.decode("utf-8"))       # 半截 JSON 会在此抛错
    assert payload["sentinel"] == "must-survive"


def test_no_temp_file_residue_after_failure():
    """③ 失败后同目录不残留临时档（原子写的中间产物须被清理）。"""
    tmpd, path, sentinel = _sentinel_dir()
    _failed_write(path)
    residue = [n for n in os.listdir(tmpd) if n != "kernels.json"]
    assert residue == [], f"失败后残留临时档：{residue}"


def test_successful_write_is_atomic_and_loadable():
    """正向：正常写入产出可 json.load 的完整档，且内容与 load_consolidated 一致。"""
    tmpd, path, sentinel = _sentinel_dir()
    save_consolidated({**DEFAULT_KERNELS}, {"epochs": 3}, path)
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["consolidated"] is True and payload["epochs"] == 3
    assert set(load_consolidated(path)) >= set(DEFAULT_KERNELS)
    residue = [n for n in os.listdir(tmpd) if n != "kernels.json"]
    assert residue == [], f"成功写入后残留临时档：{residue}"
