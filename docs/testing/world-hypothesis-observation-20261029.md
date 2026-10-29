# 修复：缺主体观测时暂停假设验证

基线 `7a8ef6bd71b570eadf69c3ef311db3f433cd1a34`；固定功能验证源码 `2fc6434b21fc6b1d9784dbd59facf9888cb0153b`。

默认 `legacy` 路径曾在主体缺观测时继续消费假设验证窗口，候选会仅因遮挡回退，
已确认假设会在 `verify.total=0` 时自动增加置信。既有 `conservative` 的缺观测暂停规则没有覆盖默认策略。
现在 `_growth_tick` 对两种策略都检查主体是否在本帧 `_obs_snapshot` 中，其他算法与参数保持。
旧 identity 守卫断言保持，只将其定点变异定位更新为同一分支的新文本；manifest 同步 wm_simloop.py 指纹。

## 原始证据与集中结果

[全部证据](world-hypothesis-observation-20261029/)。Windows / Python 3.12.10 / SQLite 3.49.1。

| 验证 | 读数 |
| --- | --- |
| 基线，同一新守卫 | 5 failed、6 passed；0 errors、0 skipped |
| 修复后 | 11 passed；0 errors、0 skipped |
| 新增定点变异 | 4/4，各恰好 1 个指定断言失败 |
| 既有 identity 变异 | 1 个指定断言失败，旧守卫仍有效 |
| 邻接回归 | 112 passed、2 skipped |
| 严格清单 | 64/64 MATCH |
| 时间核 | 0 新违规；23 条既有登记 |

新变异为 restore_legacy_bypass、pause_candidates_only、accept_other_entity_evidence、freeze_observed_subjects。
覆盖两种策略、候选和已确认状态、其他实体观测、重新出现后的确认/加置信、有效观测下的正常回退，以及实际 core.wm_simloop("step") 门面。
候选由既有 `_grow` 固定准备，后续经过真实 SceneSimulator、观察过滤、step、verify、WAL。
因此是软件合成闭环证据，不宣称是新生长算法效果或物理设备验收。

复跑（仓根，安装 .[full,dev]）：

```text
python -X utf8 -m pytest tests/test_world_hypothesis_observation.py tests/test_world_identity_qualification.py tests/test_wm_simloop_guards.py -q
python -X utf8 tools/testing/replay_save_guards.py --spec tests/fixtures/world_hypothesis_observation_mutations.json --baseline 7a8ef6bd71b570eadf69c3ef311db3f433cd1a34 --mutate --output <新证据目录>
python -X utf8 tools/verify_manifest.py --strict --quiet
```

严格重现应检出上面的固定验证提交。replay 只替换规格中的 wm_simloop.py，其余代码来自固定检出。
首轮新守卫比较完整 WMNode，将 generate 的正常 behavior_inferred 从 unknown→wander 也误判为证据变化，
产生 7 failed、4 passed；其原始日志保存为 initial-guard-failure.log。
最终只比较假设验证写入的状态、位置、置信、last_seen、条件与事件；生产修复未因该量具修订改变。
旧断言、确认阈值与 CI 门禁均未放宽。公开日志仅归一路径、换行和行尾空白，本地保留原始日志。

## 边界与查重

本地跳过 test_scene_relations_round_trip_real_brain、test_experience_real_brain_restart_preserves_predictive_relations（未配置脑端）；不计 passed。
本地 Linux 重放/变异、DSH、真实设备、在线脑/MCP、生产数据和并发未测；远端 CI 另记。
初轮与最终批次的测试副本均已清理。没有删除系统文件或其他任务产物。

全状态归档查重 692 个 Issue、366 个 PR。#1 是将内部假设位置用作 verify 的 actual，
#192 是生长阈值/其他展示与统计残面；本件是已存在假设在主体缺观测时推进验证的默认策略残面，根因和判据不同。
一个独立修复对应一个 PR/intake；定级、验收、结算由维护者裁定，human review pending。
