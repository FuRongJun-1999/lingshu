# 旋转测量的归一化与拒绝回执

`UnifiedWorldModel.perceive` 和 core 的 world_model("perceive") 共用旋转包装链。
rot_quat 的顺序为 [w,x,y,z]；未提供或为 null 表示本帧没有旋转测量。

四元数须为四项 list/tuple，数值可转换为有限 float，整体非零；布尔值不作为旋转分量。
保留旧有数值字符串的转换兼容。归一化先除以最大绝对分量，再计算模长，避免大数平方溢出、
小数平方下溢，且不再用单位姿态替代零测量。输入已丢失的浮点分量不能由归一化恢复。

无效旋转不追加到旋转缓冲、不增加已接受四元数计数；同帧有效位置继续走原观测流程。
该次回执增加 rotation_rejections，元素包含 observation_index、已解析 eid 和
reason=invalid_quaternion。legacy 使用本帧顺序，conservative 使用已确认观测的原始序号；
上游自带的索引字段不能覆盖 legacy 的本帧顺序。只有发生拒绝时才增加这个字段。

原始测量仍保留在既有 observations/身份审计中，可能含原来的非有限数值。
严格 JSON 编码的验证只覆盖后续预测，不覆盖原始观测回执。
位置校验、身份关联、旋转预测时钟、主轴判据和既有阈值均不由本件修改。

本件防止新无效值污染旋转轨迹；旧实例中已有的坏缓冲不作迁移修复。
缺少有效旋转历史时保持原位置预测输出，不声称已取得旋转预测。
真实设备、在线脑/DSH 和生产数据未测。

复跑：`python -X utf8 -m pytest tests/test_world_rotation_quaternion_validity.py -q`。
基线、具名变异与隔离组合记录见[验证报告](testing/world-rotation-quaternion-validity-20261026.md)。
