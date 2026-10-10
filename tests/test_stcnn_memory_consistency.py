# -*- coding: utf-8 -*-
"""守卫 · 时空记忆图自校验不得恒真（issue #302）

缺陷（HEAD 修复前，`lingshu/nn/stcnn.py` verify_consistency）：
    `recalled = self.recall(); return len(recalled) == len(self.events)`
    —— 无查询时 recall() 就是 `self.events` 的逐条 append，两侧长度是**同一个数的
    两次读数**，恒真。于是「记了却回忆不出」「回忆出没记过的」等记忆幻觉永远
    发现不了——「记忆无幻觉」这句承诺在这条自校验里没有判别力。

判据来源：
    · 理论/契约来源：本文件 docstring 与 `lingshu/nn/stcnn.py` 模块头声明的
      「看见→记住→回忆 白箱闭环」——闭环的三个可观测环节（写入条数 / 回忆条数 /
      逐条可达性）必须能相互印证，否则自校验无意义。
    · 断言形态为**定点变异标定**：每条 test 先证「健康记忆判 True」，再注入一种
      具体的不一致形态，证「该形态被判 False」。恒真实现（旧式
      `len(recall())==len(events)`）在注入形态下仍判 True ⇒ 必红。

运行：python -X utf8 -m pytest tests/test_stcnn_memory_consistency.py -q --no-header
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lingshu.nn.stcnn import SpatiotemporalMemory  # noqa: E402


def _rolling():
    return {"direction": "向右", "speed": 2.0, "period": None,
            "moving": True, "trajectory_len": 5}


def _blinking():
    return {"direction": "静止", "speed": 0.0, "period": 3,
            "moving": True, "trajectory_len": 4}


def _healthy():
    mem = SpatiotemporalMemory()
    mem.remember(_rolling(), label="球")
    mem.remember(_blinking(), label="灯")
    return mem


def test_healthy_memory_is_consistent():
    """基线：正常写入的时空记忆，自校验须判 True（否则守卫误伤健康路径）。"""
    assert _healthy().verify_consistency() is True


def test_legacy_length_only_formula_would_be_true_here():
    """标定恒真式在健康数据上的读数——证明「注入后仍 True」即旧缺陷未修。"""
    mem = _healthy()
    assert len(mem.recall()) == len(mem.events)


def test_detects_event_missing_spatiotemporal_anchor():
    """注入①：events 里混入一条缺时空锚点字段的损坏事件 ⇒ 必须判 False。

    恒真实现只看条数（recall 仍能 append 该条）⇒ 旧式判 True ⇒ 本条红。
    """
    mem = _healthy()
    mem.events.append({"label": "幻象"})  # 缺 t_start/t_end/direction/speed/period/moving
    # 恒真盲区自证：此时条数两侧仍然相等（recall 照样把这条 append 出来），
    # 所以旧式 `len(recall())==len(events)` 判 True —— 这就是缺陷的形态。
    assert len(mem.recall()) == len(mem.events)
    assert mem.verify_consistency() is False


def test_detects_unrecallable_event():
    """注入②：已记事件在回忆侧丢失（记了却回忆不出）⇒ 必须判 False。"""
    mem = _healthy()

    class LosingRecall(SpatiotemporalMemory):
        def recall(self, query=None):
            out = super().recall(query)
            return out[:-1] if out else out  # 丢掉最后一条

    losing = LosingRecall()
    losing.events = list(mem.events)
    assert losing.verify_consistency() is False


def test_detects_phantom_recalled_event():
    """注入③：回忆凭空多出未记过的事件（幻觉）⇒ 必须判 False。"""
    mem = _healthy()

    class PhantomRecall(SpatiotemporalMemory):
        def recall(self, query=None):
            out = list(super().recall(query))
            if query is None:
                out.append({"label": "不存在的事件"})
            return out

    phantom = PhantomRecall()
    phantom.events = list(mem.events)
    assert phantom.verify_consistency() is False
