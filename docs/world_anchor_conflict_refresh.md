# 锚点冲突的即时确认状态

AnchorVerification.channel_conflict_detect 及语义图、core.world3d("verify_conflict")
在登记真实冲突并清零该通道证据后，立即运行既有确认判定。
返回值、anchor_state 与 verification_summary 同步反映当前资格，稳定轮数和排序依据由原判据更新。

本件沿用全称量闸与全部阈值；一致观测仍为空操作，未知锚点仍由图层拒绝。
该刷新不计为新通道观测，不增加注册表 hit/miss，也不产生新稳定轮数。
冲突内容由调用方提供，刷新状态不证明现实中的矛盾或传感器结论。

[固定提交复跑、具名变异及未测边界](testing/world-anchor-conflict-refresh-20261027.md)。
