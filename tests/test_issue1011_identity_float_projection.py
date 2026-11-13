"""#1011 守卫：validate_history 的 accepted 侧与快照侧同一数值投影。

缺陷（修复前实测，锚定 e83529c）：accepted 存原始 int 测量值，快照侧经
perceive 逐值 float() 化——>2^53 的合法整数坐标（validate_measurements 放行）
使 `projected != frame["entities"]` 恒真 ⇒ v2 恢复门卫（wm_simloop
seed_from_prior）对合法存档永久 ValueError。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.world.identity import validate_history  # noqa: E402
from lingshu.world.world_model import UnifiedWorldModel  # noqa: E402


def test_g1_big_int_coordinate_survives_validation():
    m = UnifiedWorldModel(size=20, identity_policy="conservative")
    m.perceive([{"category": "rock", "pos": [2 ** 60 + 1, 0, 0], "eid": "big1"}], tick=1)
    validate_history(m.history, m.nodes)  # 修复前：ValueError disagree


def test_g2_validator_still_catches_real_disagreement():
    """防过修正向：篡改历史后校验仍须拒绝。"""
    m = UnifiedWorldModel(size=20, identity_policy="conservative")
    m.perceive([{"category": "rock", "pos": [1.0, 0, 0], "eid": "r1"}], tick=1)
    m.history[-1]["entities"]["r1"][0] += 1.0
    try:
        validate_history(m.history, m.nodes)
    except ValueError:
        pass
    else:
        raise AssertionError("篡改后的历史必须仍被拒绝")
