# W1 第一增量：行动残差诊断与定向修订

本功能落在现有 `SevenLayerLoop → ActionDecision → SceneSimulator` 行动闭环。
它分别记录“执行后自身位置是否偏离动作模型”和“目标观测是否偏离目标预测”，
只把合格的新鲜目标证据用于该行动的 x/z 预测偏移修订。

它与 `UnifiedWorldModel` 是当前仓库已有的不同运行实例。本 PR 基于最新 main
独立交付，不包含尚待审核的 #905 身份关联实现，也不把两套实例宣称为已统一。
这里的观测使用既有稳定实体 ID；匿名身份资格、真实传感器可靠性仍需各自验证。

## 启用

```python
loop.configure_action(actor, goal, feedback_policy="qualified")
# 核心门面
engine.seven_layer_loop("configure_action", {
    "entity_id": actor, "target": goal, "feedback_policy": "qualified"
})
trace = loop.step()["L7_action"]
diagnosis = trace["outcome"]["diagnosis"]
```

`feedback_policy` 支持 `legacy`（默认）与 `qualified`。省略参数时，旧配置键、
行动留痕、目标误差更新公式及旧适配器调用保持兼容。非法策略在替换配置、清除
待执行命令之前拒绝。新策略与行动选择策略、好奇观测策略分别配置。

## 读数与资格

新策略在决策时冻结行动目标、主体、步长、动作策略、决策 tick 和预期观测 tick。
结算时使用观察者节点的 `last_seen`，分别判断自身和目标观测是否为本帧新证据。
目标/行动配置不匹配、时刻不匹配、重复或乱序结算不能写入修订。
重复调用不会改写之前已保存的成功审计对象。

诊断中的 `execution` 和 `target_forecast` 分别给出资格、原因、三维残差与范数：

| 状态 | 含义 |
| --- | --- |
| ACCEPT | 该分量有可比较的新鲜有限三维观测 |
| DEFER | 无预测、缺测、陈旧或重复证据 |
| REJECT | 上下文、时刻或数值不合法 |
| BLINDSPOT | 配置坐标是常量，不是新的目标观测 |

`classification` 标记执行残差、目标预测残差、两者并存、无实测残差或证据不足。
配置中的目标确实变化时标记 `goal_context_changed`，拒绝旧目标的反馈。
**位置残差不能单独证明目标意图变化、执行器故障或身份/观测错误**，因此 `cause`
保留 `unknown`。组件资格只描述可比较证据，不提升知识事实或权限。

自身陈旧时不计算 `self_error`，也不把旧自身位置与新目标拼成当前距离；
目标新鲜时仍可独立修订目标偏移。反过来，只有自身的新鲜证据不能修订目标。
坐标目标不会新增“目标观测误差”样本。垂直残差如实记录，但现有修订量只覆盖
x/z；`unsupported_correction_axes=["y"]`，不宣称已修正垂直误差。

## 修订公式

已有预测为 `base + correction_applied`，观察残差为 `actual - prediction`。
新策略先还原对基础模型的偏差：

```text
innovation = actual - prediction + correction_applied
next_correction = retain * current_correction + alpha * innovation
alpha = 0.5, retain = 1 - alpha
```

平滑计算复用项目统一时间核 `cred_blend`；未新增时间核或阈值豁免。
旧策略保持原公式。持续基础模型偏差下，新策略逐步补足偏移；基础预测改善后，
修订量能回落。修订只作用于该决策器的目标偏移，不改执行器、实体位置、关系图、
物理规则或长期记忆。坐标单位沿用模型坐标。

`diagnosis.revision` 保留资格、作用对象、修改前后值、基础预测偏差和是否发生数值
改变；`corrected` 仅在偏移确有改变时为 true。诊断状态与数值增益不是因果结论。

## 验证与边界

```text
python -X utf8 -m pytest tests/test_world_qualified_feedback.py -q
python -X utf8 tools/testing/world_feedback_scenarios.py
```

固定场景包括官方 `decide(predictions=...)` 入口输入的受控基础预测偏差，以及
真实模拟器碰撞箱挡住受控动作。预测偏差场景先修订六帧，再冻结参数读取两帧
留出误差；不边评估边更新，不把碰撞或新目标位置传给决策器作隐藏真值。

本轮是模拟世界与公开 Python/核心门面验收，不代表真实摄像头、在线大脑、
DSH 或实际机器人效果；单一受控偏差不代表普遍预测收益。未实现一般因果归因、
自动修改物理规律、三维执行器学习或跨重启反馈去重。SevenLayerLoop 目前没有
此配置的独立持久化格式，本件不新增恢复承诺。需要维护者技术审核与理论裁决。
