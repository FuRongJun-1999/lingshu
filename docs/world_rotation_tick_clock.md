# 旋转预测的世界时钟

`UnifiedWorldModel.perceive(..., tick=...)` 记录每条四元数观测的时刻。
`generate(horizon=h)` 的旋转角预测与位置预测对齐到 `model.tick + h`。
core 的 world_model("perceive"/"generate") 和隔离 counterfactual 复用同一条链。

旋转缺测后，外推时长是“目标时刻 − 最后旋转观测时刻”。只有位置的新观测不会
把旧旋转样本变成新测量；空帧也不会向旋转缓冲补造观测。

角速度按真实 tick 间隔求平均，单位 rad/tick。加速度使用两个滑窗的时间中点间距，
再将窗口平均速度推进到最后旋转观测的时刻，按局部等角加速度外推到目标时刻。
因此稀疏或不等间隔采样不会被当作连续的单位 tick 样本。

周期检测器仍按采样索引工作：只对等间隔旋转样本启用。检测到的周期乘采样间隔后
作为 `rot_P` 输出，单位为世界 tick；目标时刻换回采样格后调用共享相位插值。
跨多个周期先回绕相位，不均匀采样则记录 nonuniform_sampling，保留局部速度/加速度分支。
内部 sample_period/P_used 保留检测器的采样格口径。

内部旋转诊断记录 last_observation_tick、target_tick、observation_age、
horizon_from_observation、sampling_step 和速度/加速度单位。
没有旋转历史时输出保持原位置预测链的形状；位置预测算法、解缠规则和既有阈值不变。

本件只修时间换算，不把单轴标量角升级为完整多轴刚体姿态，也不解决采样混叠、未知转数、
噪声标定或运动突变。旋转角的测试读数由独立合成轨迹给出；现有 verify 的位置评分
不作为旋转命中的证据。真实设备未测。

复跑：`python -X utf8 -m pytest tests/test_world_rotation_tick_clock.py -q`。
改前/改后、具名变异及未测范围见[验证报告](testing/world-rotation-tick-clock-20261026.md)。
