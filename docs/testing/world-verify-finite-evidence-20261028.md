# 世界模型验证的数值资格与分母隔离

日期：2026-10-28

## 问题与修复

legacy 观测路径保留非有限坐标供诊断；verify 却把 NaN/Inf 当作可用 actual，
计入 total 并标 verified。NaN 的 distance/mean_distance 不可严格 JSON 编码，
还会将未知结果算成一次失败，污染有效实体的命中率分母。坏预测、坏 bound 与距离溢出同样可误计分。

本件只改 UnifiedWorldModel.verify：三轴坐标有限、bound 有限且非负、比较距离可表示时才评分。
缺失、时刻不符等原闸继续保留；不可验证项标 pending，hit/distance 为空，
unverifiable_reason 区分 invalid_observation、invalid_prediction、invalid_bound、unrepresentable_distance。
返回诊断中的不可用坐标/bound 置 None；原预测、原观测快照、节点和历史不被重写。
有限的距离/半径之和溢出时，用缩放均值，保留原正常范围内的求均值路径。

## 固定源码与集中验证

main 基线 `02caf8bfe6d53fb33981d2f51c0ebceef9a0957f`；固定源码验证提交 `cb7ec0117d59dda49e0a31a87700c5754da1607d`。
Windows / Python 3.12.10 / SQLite 3.49.1，一次集中定向批次；未跑本地全套。

| 项目 | 实测 |
| --- | --- |
| main，同一守卫 | 13 failed、5 passed；0 errors / 0 skipped |
| 修复后 | 18 passed；0 failed / 0 errors / 0 skipped |
| 具名变异 | 4/4 各恰好 1 failed，命中指定断言；0 errors / 0 skipped |
| 相关回归 | 231 passed、2 skipped |
| 派生清单 | 64/64 MATCH |
| 时间核 | 0 新违规；23 条既有登记 |

具名变异：score_nonfinite_actual、accept_nonfinite_bound、score_overflowed_distance、sum_metric_values_before_scaling。第一个变异同时移除 actual 输入闸与距离兜底闸，
其余分别放过无效 bound、不可表示距离或恢复先求和的溢出均值。
每个只运行指定用例、核对指定断言，错误/未收集不计成功。

18 条守卫覆盖 NaN/正负 Inf、有效与无效实体混合、预测/bound 资格、有限端点的距离溢出、
两个有限 1e308 距离的平均值、core 门面、缺失/目标时刻/陈旧快照以及只读源状态。
基线经真实 core 输入合法 JSON 数值字符串 "NaN"：verify 返回 total=1、pending=0、
status=verified、mean_distance=NaN。修复守卫要求 total=0、pending=1，评分率为 None，
严格 JSON 编码通过，SQLite 写入计数不变。混合样本只让可验证实体进入评分分母。

## 复跑

仓根安装 .[full,dev] 后：

```text
python -X utf8 -m pytest tests/test_world_verify_finite_evidence.py -q
python -X utf8 tools/testing/replay_save_guards.py --spec tests/fixtures/world_verify_finite_evidence_mutations.json --baseline 02caf8bfe6d53fb33981d2f51c0ebceef9a0957f --mutate --output <新证据目录>
python -X utf8 tools/verify_manifest.py --strict --quiet
```

重放只恢复规格中列出的生产源码，其余依赖取运行检出；严格重现应检出固定验证提交。
临时副本已清理，[公开原始读数](world-verify-finite-evidence-20261028/)。公开日志仅归一工作区/解释器前缀、换行和行尾空白。
原始特殊浮点读数在 JSON 探针中以 NaN/+Infinity/-Infinity 字符串展示，strict_json 字段记录原编码是否成功。

## SKIP 与未测边界

test_scene_relations_round_trip_real_brain；test_experience_real_brain_restart_preserves_predictive_relations。相关脑端依赖跳过项不计 passed。本轮未做本地 Linux 重放/变异，远端 Linux CI 单列。
真实设备、在线脑/MCP、DSH、生产数据、性能和并发未测。

这里只修验证结果的资格与有限统计，不改变 legacy 的原始观测接收或修复已污染的预测/轨迹缓冲，
也不把 pending 说成命中或失败。严格 JSON 断言仅覆盖本件验证返回，不覆盖保留原始坏读数的 perceive/history。
预测生成、旋转逻辑、关系反馈写入、阈值、CI、依赖及脑端代码均未改；human review pending。
