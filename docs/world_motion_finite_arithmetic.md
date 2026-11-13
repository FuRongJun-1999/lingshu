# 运动估计的有限数值边界

ObservationWindow.motion/motion_detail、estimate_motion 和最近方向查询沿用真实观测，
不补造帧。三轴坐标须可转换为有限值；坏形状、转换溢出和非有限坐标不作可用证据。

差值先溢出而真实间隔下速度可恢复时，尝试先除间隔再相减；不可表示的运算配对拒绝采用。
有效速度之和溢出时用缩放均值，正常数值与长间隔权重保持原规则。
motion_detail 仅在运算拒绝发生时增加 arithmetic_pairs_dropped；无估计时说明
invalid_motion_arithmetic，保持 estimate=None/adopted=False，不伪造零速。

原四元组、估计器三参替换口、缺失帧与新证据计数语义保持；最近方向不返回非有限向量。
该修复不提供任意精度、极值预测精度或性能保证，也不改写原始历史。
[复跑、变异、首轮失败与清理残留](testing/world-motion-finite-arithmetic-20261028.md)。
