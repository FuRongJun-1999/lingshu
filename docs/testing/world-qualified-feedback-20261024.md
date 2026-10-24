# W1 第一增量：行动残差诊断与定向修订

日期：2026-10-24

现有 F08 已分开记录自身与目标误差，但未提供可用的资格诊断/定向修订模式。持续基础预测偏差为 1 时，旧公式对已经加过 correction 的预测残差再次平滑，修订量固定在 0.5。旧行为是兼容对照，本件不改写默认路径或已交付 F08 的原判据。

新增显式可选的 `feedback_policy="qualified"`：冻结行动目标和预期观测 tick，分别诊断执行残差、目标预测残差及证据缺失；旧目标上下文、时刻不符、陈旧、重复及非有限读数不推进修订。新鲜目标证据只修订该行动的 x/z 预测偏移，基础预测偏差先由 `actual - corrected_prediction + correction_applied` 还原，再经统一时间核 `cred_blend` 平滑。真实碰撞的执行残差不混入目标偏移。

`cause` 保持 unknown：位置残差本身不能证明目标意图变化或执行器故障。明确的配置目标变化可按上下文识别，不能用旧目标证据修订新目标。坐标目标是常量；垂直残差记录为未支持的修订轴，不冒充完整三维学习。

## 基线、守卫与定点变异

当前 main 冻结为 `50ee03b2c17aa15c5e6859ee208ef2a4aedf6ee5`；双平台源码验证提交 `e40acb72002e4ac0177b82b55562fb600c676c45`。基线的 22 条失败以具名断言报告新策略缺失，1 条旧反馈兼容对照通过；不把新增接口缺失算作导入错误或高优先级漏洞。

| 平台 | 同一守卫在基线 | 功能实现 | 反面对照 |
| --- | --- | --- | --- |
| Windows / Python 3.12.10 | 22 failed, 1 passed in 1.21s | 23 passed, 9 warnings in 0.82s | 5/5 具名变异各点红一条用例 |
| Linux / Python 3.13.12 | 22 failed, 1 passed in 2.32s | 23 passed, 9 warnings in 1.16s | 5/5 具名变异各点红一条用例 |

五组具名变异分别为：把已修正残差当基础偏差、接纳陈旧目标、复用旧目标上下文、计分陈旧自身位置、核心门面丢反馈策略。每组只跑指定用例，恰好 1 failed / 0 errors / 0 skipped，核对指定断言。有效功能批次、基线和变异均无超时或 SKIP。

- Windows：`90 passed, 122 warnings in 7.51s`，0 skipped。
- Linux：`90 passed, 122 warnings in 36.17s`，0 skipped。

相关回归包括既有 F08 全部行动/门面、非法参数隔离、世界反馈与文档计数/隐私守卫。严格清单 `entries=64 match=64 mismatch=0`，时间核没有新增违规。每平台只跑一个有效定向批次，不跑本地全套，不为整理材料重复稳定测试。

## 真实入口的固定场景

场景通过官方 `ActionDecision.decide(predictions=...)` 接受受控基础预测，随后真实 `SceneSimulator.set_command/step`、`CuriosityExplorer.observe` 和反馈结算；不替换感知、执行或反馈方法。偏差场景的目标实际静止，基础预测固定少 1 个模型坐标单位。先修订 6 帧，再冻结参数取 2 帧留出读数：

| 固定场景指标 | 原始 main / legacy | qualified |
| --- | --- | --- |
| 第 6 帧后冻结的 x 偏移 | 0.5 | 0.9844 |
| 两帧留出平均点误差 | 0.5 | 0.015600000000000058 |
| 固定阈值 `<0.5` 的留出命中 | 0/2 | 2/2 |
| 真实碰撞：自身执行误差 | 1.0 | 1.0 |
| 真实碰撞：目标误差 / 目标偏移 | 0.0 / 0.0 | 0.0 / 0.0 |
| 碰撞残差诊断 / 因果结论 | 原接口无此读数 | execution_residual / unknown |

两平台均与此表一致；新提交的 legacy 场景 JSON 与 git archive 导出的原始 main 完全一致。碰撞由现有碰撞箱内核实际产生，contact_count=1。此处只证明受控偏差修订与分量隔离，不代表一般预测性能或真实设备收益。

## 与上一增量的组合

上一份 [PR #905](https://github.com/FuRongJun-1999/lingshu/pull/905) 仍待人审；本 PR 基于最新 main 独立开发，不包含其代码或重复申报身份关联能力。现有 SevenLayerLoop 与 UnifiedWorldModel 是不同运行实例，本件不声称它们已统一。

Windows 隔离源码组合：以共同祖先做 core.py 三方文本合并，左侧保留本件行动反馈，右侧取 #905 的身份/恢复实现；两份具名守卫共 45 passed / 0 skipped，临时副本清理完成。该项不是正式 Git 合并、清单或全套组合验收，Linux 组合未跑。

## 使用与复跑

```text
python -X utf8 tools/testing/world_feedback_scenarios.py
python -X utf8 -m pytest tests/test_world_qualified_feedback.py -q
python -X utf8 tools/testing/replay_save_guards.py --spec tests/fixtures/world_qualified_feedback_mutations.json --baseline 50ee03b2c17aa15c5e6859ee208ef2a4aedf6ee5 --mutate
python -X utf8 -m pytest tests/test_third_batch_actions.py tests/test_third_batch_core_action.py tests/test_issue691_z3_action_facade_guard.py tests/test_world_feedback.py tests/test_doc_module_counts_guard.py tests/test_public_docs_privacy_guard.py -q
python -X utf8 tools/verify_manifest.py --strict --quiet
python -X utf8 tools/time_core_lint.py --json
python -X utf8 tools/testing/replay_world_increments.py --left e40acb72002e4ac0177b82b55562fb600c676c45 --right 17e10c30352662a86c28d21624243a2c7b465bb0 --output <新输出目录>
```

组合工具随本件新增并实际执行一次，保存了工具指纹和合成源码指纹。组合引用两个本地已取到的提交，缺对象时应先获取对应贡献分支。输出目录必须不存在；工具仅删除自己创建的临时子目录。使用说明见[固定验证提交的功能说明](https://github.com/STARDUSTLC666/lingshu/blob/e40acb72002e4ac0177b82b55562fb600c676c45/docs/world_qualified_feedback.md)。同名证据目录提供双平台原始读数、场景 JSON、组合日志与来源指纹；公开日志仅替换工作区前缀及归一化换行/行尾空白，完整原文保留本地。

## 既有贡献和验证边界

沿用 @rinDBeans 在 [PR #516](https://github.com/FuRongJun-1999/lingshu/pull/516) 交付的 F08 行动、执行分离、观测新鲜度和核心门面工作，不把旧功能计作本件贡献。目标仓及 bounty 的所有状态条目已查重；#516 与已结算的 bounty #160 属原功能，本件只申报新增诊断资格和可用的定向偏移修订。

判定为部分验证：已验公开 Python、核心门面、真实模拟器碰撞、冻结修订后的受控留出读数；未验真实摄像头、在线大脑/MCP、DSH、机器人、生产数据或一般因果归因。新策略仍依赖输入适配器的稳定实体 ID，不提供匿名重识别；没有修改物理规律、实体位置、关系图、执行器或长期记忆。没有独立持久化/跨重启去重承诺，完整 W1 仍待后续。默认 legacy 与旧适配器兼容；未新增运行依赖、门禁豁免或阈值。维护者技术审核和理论裁决待完成，human review pending。
