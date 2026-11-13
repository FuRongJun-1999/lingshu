# 隔离关系条件试演

## 用途与启用条件

在已观测的 UnifiedWorldModel 上比较不同 seek/flee 关系条件，沿用当前生效的预测链。只在调用 `counterfactual` 时启用，普通 `generate/perceive/verify` 的行为保持原契约。

```python
from lingshu.world.world_model import UnifiedWorldModel

model = UnifiedWorldModel(size=80)
for x in range(6, 12):
    model.perceive([
        {"eid": "actor", "category": "person", "pos": [x, 1.5, 2]},
        {"eid": "goal", "category": "marker", "pos": [30, 1.5, 2]},
    ])
result = model.counterfactual(horizon=1, relations=[
    {"source": "actor", "relation": "flee", "target": "goal"},
])
print(result["baseline"]["predictions"]["actor"]["predicted"])
print(result["alternative"]["predictions"]["actor"]["predicted"])
```

core 门面为 `engine.world_model("counterfactual", {"horizon": 1, "relations": [...]})`，返回 `status=ok` 与 `counterfactual`。参数原样转发，非法值不通过整数转换变成另一种有效请求。

## 输入和执行

- `horizon` 是严格整数，范围 1..64；这是计算预算，不是经实验标定的预测视界。布尔、浮点、字符串和越界值抛 ValueError。
- `relations` 缺省为无条件对照；提供时必须是列表，每条恰含 source/relation/target。两端为已知且不同的实体 ID，关系限 seek/flee，每个来源至多一条。
- 两份副本分别承载原预测与条件候选；条件只替换候选中相同来源的关系。候选复用当前速度、轨迹、旋转、模式守卫及显式物理环境，仍受现有预测模式优先级约束。一个关系条件不保证改变每个场景的预测，也不证明实体将采取该行为。
- 整个模型状态深拷贝；world 及指向它的别名被隔断。不会读取模拟器隐藏的目标、行为、随机种子结果或推进物理世界。实例上的显式 physics_env/environment 可供现有可选物理包装层使用。
- 副本的生成、模式推断、关系、诊断缓存、随机状态与异常均在副本内。失败直接抛出异常，副本丢弃，无提交操作；不把“试演结束”变成观测、记忆或真实预测。

## 返回与可信边界

`wm.counterfactual/1` 返回 source_tick、target_tick、horizon、assumptions、baseline、alternative 及每个实体的位移差/距离。外层 `provenance=counterfactual`、`verification=unverified`、`committed=false` 必须随结果保留；其中的 confidence/bound 沿用预测模型语义，不是对候选条件的确认。

结果中的两份预测与真实 `_last_prediction` 分离。可用于候选比较或呈现；不可当作 actual 送回 perceive 或宣称已验证。程序无法阻止调用方主动丢弃来源标记并伪装输入，采集端仍须负责真实观测来源。此接口不自动选择或执行行动，不写长期记忆，也不做因果归因。

本轮为 W3 第一增量，只做模型关系条件的隔离比较。没有实现完整物理分支模拟、动作序列搜索或跨分支经验学习；没有改变 SevenLayerLoop 与统一模型的实例关系。

## 验证方式与不适用条件

```text
python -X utf8 tools/testing/world_counterfactual_scenarios.py
python -X utf8 -m pytest tests/test_world_counterfactual_isolation.py -q
python -X utf8 tools/testing/replay_save_guards.py --spec tests/fixtures/world_counterfactual_mutations.json --mutate
```

固定场景只测候选差异与来源状态保留，不作为一般精度/性能基准。守卫覆盖旧生成对照、成功/失败隔离、真实预测后续核验、别名、core 数据库零写入、参数闸、显式物理包装链及具名变异。

调用方必须串行提供稳定的模型；深拷贝不提供并发事务快照。成本随模型状态内存增长，本轮未做大规模性能验证。支持仓库原生可深拷贝的模型状态；自行挂载不可复制的连接/文件/自定义对象可能使复制失败。普通 Python 扩展可执行任意代码，本接口不提供不可信扩展代码的安全沙箱。

真实摄像头、在线脑/MCP、DSH、机器人、生产观测、完整物理层开启后的全套兼容及一般因果识别均未由本件验证。实际验收和理论口径由维护者审核。
