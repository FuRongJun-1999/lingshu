# 世界模型功能补全 · 第一批

范围：F01 推演经验入脑与恢复、F02 场景关系往返、F03 显式实例身份、F05 外部视觉观测。
基线为 `cb3a6ea2dbd9771d685b2bf5c1fe398abe69e419`。对照 2026-10-09 的缺口报告重新读了
当前源码：同类多物体融合、同帧一一追踪已在基线，本批沿用；图文检测两条旧红也已修复。

## 数据通路

```text
像素检测 bbox + eid + tick
  → VPrim / MultiViewFusion → World3D.observations(tick)
  → UnifiedWorldModel.perceive → generate → 下一帧 perceive → verify
  → SimulationLoop.flush_payloads(brain=...) → 脑端 MCP
  → 新实例 load_priors(brain=..., memory_id=...) → 生成端继续使用经验
```

沿用现有模块和 MCP 客户端，不增加数据库、运行依赖、调度器或跨轮缓存。

## F03 / F05：同一份观测身份与采样时刻

- `VPrim.eid`、`Object3D.eid`、`MultiViewFusion.add_observation(..., eid=...)` 接受检测/跟踪器
  的实例身份。未给 ID 的路径保留已有几何关联。
- `World3D.add_vprim/add_view(..., tick=...)` 保存真实帧号。融合只组合一个 tick 内同一实例的
  多相机射线；下一帧开始新一组三角化，旧帧不会拉偏运动物体的位置。
- `World3D.observations(tick=...)` 只导出该帧实际重建成功的对象，遮挡对象不被重新写成静止观测。
- `UnifiedWorldModel.perceive(..., tick=..., timestamp=...)` 使用采样 tick 估计真实 dt，保留
  来源与时间。采样 tick 必须递增，第一帧可以从 0 开始；缺省仍是一轮加一。
  timestamp 是溯源字段，速度单位仍为位置/tick。
- `perceive` 返回已关联 ID 的 `observations`；视觉门面回填同一关联结果。匿名关联优先使用
  目标 tick 正好匹配的生成先验，继续遵循现有类别、两体素半径和一一匹配约束。

核心门面：

```python
engine.world3d("add", {"category": "chair", "eid": "chair-1", "bbox": bbox,
                       "screen_w": 640, "screen_h": 480,
                       "tick": 10, "timestamp": capture_time, "source": "camera"})
engine.world_model("perceive_visual", {"tick": 10, "timestamp": capture_time})
engine.world_model("generate", {"horizon": 3})
# 实际第 13 帧再次 add / perceive_visual，然后 verify。
```

`world3d("add_view")` 也转发 eid/tick。已有相机参数传递范围保持原状；真实相机元数据的
记忆重建属于第二批 F04。外部三维观测可直接经 `world_model("perceive", {"observations": ...})`。

身份边界：显式 ID 是上游关联证据，不能从不可辨别的交叉/完全遮挡像素中凭空推导。
匿名最近邻与生成先验是启发式，没有新增通用外观重识别器，也不保证任意匿名交叉都正确。

## F02：场景关系与条件往返

`ingest_scene` / `ingest_scene_to_brain` 现在可直接接收 `scene_model.WorldModel`。
仍逐实体调用现有感知写入与状态事件通路，返回脑侧节点 ID 列表；旧文本格式保持可用，
另可传 `relations=[(source_name, kind, target_name), ...]`。

```python
scene.add_entity("杯子", "cup", (4, 1, 5), eid="cup-1",
                 conditions={"source": "camera", "timestamp": capture_time})
scene.relation("杯子", "right_of", "桌子")
ingest_scene_to_brain(brain, scene)
restored = load_world_from_brain(brain)
restored.entities["桌子"].pos = (6, .45, 5)
restored.build()  # 已恢复的关系继续约束布局。
```

实体卡的 `lingshu-scene` JSON 块保存 eid、呈现类型、条件和关系。恢复完成实体选择后再接边，
关系来自被选中的同一份最新实体观测；脑侧 created_at 现在供原有时新度选择使用。
空间关系沿用 `left_of/right_of/near/far/in_front/behind`，不新增物理接触求解。
本阶段关系保存在既有实体记忆卡中，可由场景模型查询；未增加脑端通用派生边检索。
静态场景接口仍以场景内唯一名称索引实体，eid 用于跨模块身份；同类别实例须使用不同名称。

## F01：脑侧经验恢复成内部先验

```python
loop = SimulationLoop(world_model=wm)  # 与正在接收视觉观测的模型共享对象。
saved = loop.flush_payloads(brain=brain)
brain.store.client.close()
# 使用同一隔离/目标根重新 connect，创建新模型。
resumed = SimulationLoop()
resumed.load_priors(brain=brain, memory_id=saved["memory_id"])
prediction = resumed.wm.generate()
# 只有下一次真实 perceive 才能提供 verify 的 actual。
```

经验块 `lingshu-world` / `lingshu.world-prior.v1` 保存实体 ID、观测位置、推断关系及其依据、
条件、模型时刻/空间边界、完整观测历史、运动统计和已确认拓扑假设。未确认假设不进入可恢复
先验。恢复不会把推断行为安装成模拟器的隐藏规则，也不会把记忆位置当成新的观测快照。

写入走 `BrainEngine.add_perception → mdcg_remember`，role 为 `tool-output`；自动发现使用脑端
现有 `receipt` 视图，不把机器推断改标为用户知识。指定 memory_id 时经 MCP 精确读取；建议
多场景应用持有这个 ID。正文按脑端 next_offset 续读，50 KiB 预览不会截坏经验 JSON。
核心 `wm_simloop("flush"/"load")` 也转发 Python 调用方提供的已连接 brain 对象及 memory_id。
`wm_simloop("attach")` 把推演循环接到核心当前统一模型；视觉观测、推断与经验保存因此使用同一个对象。
成功执行 `world_model("init")` 会替换当前模型，同时结束附着在旧模型上的循环。
如需保留旧模型的经验，请在重新初始化前执行 `wm_simloop("flush")`；重新初始化不会修改已写入的 WAL。
新模型需显式 `wm_simloop("attach")` 后才能继续推演，建议使用新的 WAL 路径。
`wm_simloop("setup")` 创建的独立循环不受统一模型替换影响。

边界：自动发现只从可见的最多 200 条候选中选 saved_at 最新的快照，仍受脑端召回与权限约束；
旧 mdcg_root 文件加载保留原位置/类别沙盒种入语义。本批不改变该旧入口的行为。完整历史随
经验保存，体积随观测数增长，没有实现历史压缩；恢复后应沿用原 tick 坐标，再输入新观测。

## 验证

六个聚焦用例覆盖原 store / 真实脑端的关系往返、真实 MCP 关闭再连接后的经验恢复与续读、
同类显式 ID 交叉与遮挡、受控图像像素 bbox 到三维预测与下一帧验证、核心外部观测转发。
其中经验用例串起两个同类实例的像素观测、第一轮 2/2 验证、空间关系入脑、重连恢复及
第二轮 2/2 验证；生成模式为 exact，确认恢复的 seek 关系实际参与预测。
它们使用隔离数据库、隔离脑根和辅助目录；不写在役记忆。缺脑包时两项 MCP 用例显式 skip。

```bash
# MDCG_BRAIN_PYTHONPATH 指向脑包；项目 CI 已提供固定公开提交。
python -X utf8 -B -m pytest tests/test_world_memory_first_batch.py -q -p no:cacheprovider
python -X utf8 -B tools/verify_manifest.py --strict --quiet
```

受控像素检测器只读取单个白色矩形的图像 bbox，用来检验数据通路；没有使用模拟器位置
充当图像观测，也不将该结果当成真实场景识别精度或匿名重识别能力的证明。
