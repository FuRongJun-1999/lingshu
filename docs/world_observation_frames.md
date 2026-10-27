# 整帧观测提交边界

`UnifiedWorldModel.perceive` 和 `SpacetimeMemoryEngine.world_model("perceive", params)`
接收一帧外部测量。生效包装链先检查整帧，再更新 tick、身份随机数、节点、置信、
观测历史和预测缓冲；格式检查失败时保持原运行态，修正数据后可重送同一个 tick。

帧容器为 list/tuple，每条为 dict，包含 category 和三维 pos；conditions 如提供必须为
dict。显式 tick 是非负整数（不接受 bool），已有观测后必须严格递增，第一帧可用 0。
默认传感器入口先读取一次整帧，再走相同检查，不重复读取设备。

默认 legacy 继续支持可转为 float 的坐标、可转为字符串的类别和 ID；NaN/Inf
保留既有运动诊断口径，不在本修复中改成严格拒绝。conservative 复用既有测量校验，
要求三个有限数值，保留 ACCEPT/DEFER/REJECT 关联回执和未决观测隔离规则。

通过格式检查不代表测量为真。上层仍须提供可信的来源、坐标标定、时序和身份；
后续独立观测才能核验原预测。这里不提供并发事务快照，不回滚自定义扩展代码或
任意运行时故障；并发修改输入、外部对象的转换副作用及真实设备均未验收。

复跑：`python -X utf8 -m pytest tests/test_world_observation_atomicity.py -q`。
完整基线、具名变异、环境与未测项见[修复验证记录](testing/world-observation-atomicity-20261026.md)。
