# 示教迁移目标角色绑定互斥校验的调用约束

对应 docs/world_process_plan.md 的相对轨迹示教迁移与角色重新绑定；不能以一个实体键表示两条不同角色轨迹。只是目标名称互斥，不限制复用或交换不同的原始名称。

在原生 transfer_demonstration 入口拒绝 actor==ref，抛出与 complete_process 同口径的 ValueError。core 直接记录和 memory_id 恢复两条路径都受此约束。不同目标角色的原坐标回放、移动参照物、平移/旋转/尺度与时刻偏移继续保留两条轨迹，source 仍为 generated。

新守卫实际调用模块、临时 SQLite 引擎及 core 的 extract/load/transfer，核对记忆原文不变。沿用 transfer 对非法参数抛 ValueError 的既有契约，没有把它改成 error dict。没有写入真实观测或改动脑端。未测 DSH、真实设备或生产数据。
