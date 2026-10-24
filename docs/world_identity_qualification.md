# W2 第一增量：保守身份关联

同类别实体靠近、交叉或暂时遮挡时，最近邻返回一个 ID 不等于身份已确定。
本增量在统一世界图的理解入口增加显式可选的保守关联，沿用现有预测、
观测历史和世界经验恢复通路。默认 `legacy` 保留原有最近邻行为。

## 使用

```python
from lingshu.world.world_model import UnifiedWorldModel

wm = UnifiedWorldModel(identity_policy="conservative")
wm.perceive([
    {"eid": "a", "category": "person", "pos": [0, 1.5, 2]},
    {"eid": "b", "category": "person", "pos": [2, 1.5, 2]},
])
frame = [{"category": "person", "pos": [0.5, 1.5, 2]}]
preview = wm.inspect_identity(frame)  # 只读：不推进 tick，不分配 ID
result = wm.perceive(frame, tool="camera")
assert result["deferred"] == 1
assert result["observations"] == []
```

引擎入口对应：

```python
engine.world_model("init", {"identity_policy": "conservative"})
engine.world_model("identity", {"observations": frame})
engine.world_model("perceive", {"observations": frame})
engine.world_model("identity_policy", {"policy": "legacy"})
```

`identity` 与 `identity_policy` 在未初始化时返回错误，不会隐式创建世界。
策略查询省略 `policy`。非法策略、非有限三维位置或无效 tick 在写事实前拒绝；
切换策略不重写历史。原 `perceive` 签名及其包装链转发契约保持不变。

## 判据和回执

关联只读取输入的位置、类别、显式 ID 和模型既有观测/目标 tick 的预测。
不读取模拟器的隐藏身份、行为、目标或评测器真值。候选半径沿用现有
`distance < 2.0`，单位为模型坐标；候选不截断，以免截断后伪造唯一性。

| 输入条件 | 关联状态 | 写入已确认世界图 |
| --- | --- | --- |
| 全帧内唯一显式 ID | ACCEPT | 按显式 ID 更新/创建 |
| 匿名观测只有一个候选，且该候选无其它观测竞争 | ACCEPT | 更新该候选 |
| 匿名观测有多个候选，或竞争同一候选 | DEFER | 暂缓 |
| 同一显式 ID 在本帧重复出现 | REJECT | 两条都不写入 |
| 匿名观测没有可用旧轨迹候选 | ACCEPT | 分配新观测轨迹 ID |

全帧显式 ID 优先预留；匿名新 ID 也检查本帧预留与既有节点碰撞。
“新轨迹”只表示当前候选集合没有可关联对象，不断言该物理对象从未出现。
内部隐藏目标假设不作为匿名观测的身份候选。

`result.identity.readings` 按原输入序号保存 `observation_index`、目标 tick、
原始测量、候选距离/参考位置/参考来源、`association_state`、原因和关联结果。
只有 ACCEPT 有 `resolved_eid`；匿名新轨迹的 `matched_eid` 为 null。
ACCEPT/DEFER/REJECT 在此仅限定**身份关联**，不升级知识真实性、权限或脑侧资格。

未决测量保留在 `history[*].identity_readings`，含观测工具和可选时间戳；
`history[*].entities`、真实观测快照、置信更新、轨迹、配对误差和旋转缓冲只接收
已确认关联。结果中的 `observations` 是已确认子集，并含原始观测序号；
视觉回填按此序号写回对象 ID。

验证仍只用新鲜真实观测，缺少可比较数据时 `total=0`、`hit_rate=None`。
保守模式下，SimLoop 的假设主体没有本帧已确认观测时，候选验证窗口、
确认/回退及已确认假设置信均等待；episode WAL 记本帧接受/暂缓/拒绝数量。

## 保存与恢复

保守策略或存在关联回执历史时，`SimulationLoop.flush_payloads` 导出
`lingshu.world-prior.v2`，保存策略与审计历史。全程 legacy 且无回执时仍导出 v1。
新加载器接受 v1（恢复为 legacy）和 v2；旧加载器会拒绝 v2，避免静默降级策略。
恢复前验证 v2 策略及回执与已确认历史的一致性，任何失败不发布候选运行态。
恢复的内容是内部先验，不是新观测；误差/旋转缓存不从另一段旧运行态沿用。
有 `limit` 截断时仅选定实体进入事实历史，关联回执保留完整的历史审计引用。

## 验证与边界

```text
python -X utf8 -m pytest tests/test_world_identity_qualification.py -q
python -X utf8 tools/testing/world_identity_scenarios.py
python -X utf8 tools/testing/replay_save_guards.py --spec tests/fixtures/world_identity_qualification_mutations.json --baseline c6139540461aa20f72bb6aa15e1b31e83c772af3 --mutate
```

场景评分器把真值 ID 留在模型外，只传匿名测量。必须同时报告误关联和接受覆盖率，
不能把全部 DEFER 当作成功。新 API 在基线上由具名断言报告缺失，不计作导入错误。
回放恢复三个既有生产文件为基线；新增纯函数模块在沙箱中存在，但基线未引用它。
旧 ID 分配变异的定位随分配函数迁移，断言和碰撞条件不改。

本增量是确定性合成场景与本地文件协议验收，不代表真实摄像头、在线脑/MCP、
DSH、私有场景资产或通用多目标追踪效果。当前不做外观/ReID、全局匹配、
跨遮挡多假设重关联、轨迹淘汰或坐标单位标定。输入显式 ID 仍依赖观测适配器的
可信追踪；错误的显式 ID 不会由本增量自动纠正。候选时间和诊断空间开销均为
O(观测数 × 已知实体数)，未声明大规模性能收益。它为后续 W2 消歧提供可审计入口，
不表示 W2 全部完成或七层世界模型已收口。
