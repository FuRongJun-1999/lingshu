# 非整数周期的多周期预测

功能入口沿用 UnifiedWorldModel.generate 与 core 的 world_model("generate")。
默认 noninteger=False；显式 set_period_caliber(noninteger=True) 或既有环境变量开启后，
检测器可返回浮点周期，位置通道按历史相位线性插值。

horizon 超过该周期时，先把跨度回绕到最近一周期；保留浮点周期的小数部分，
整周期回到最后观测锚点。整数周期、周期内预测和默认开关沿用既有行为。
此规则同样作用于复用生成链的 counterfactual 候选，不写回真实实例或长期记忆。

本修复只处理已检测周期的取址，不保证周期检测正确或未来继续遵守同一周期，
也不校准多步置信度/范围。不处理稀疏采样的时间换算、设备接入或执行行动。
真实设备和在线脑/DSH 未测试。

复跑：python -X utf8 -m pytest tests/test_world_period_phase_wrap.py -q。
[固定提交验证与具名变异](testing/world-period-phase-wrap-20261027.md)。
