# 示教迁移目标角色绑定互斥校验

示教记录原本包含 actor/ref 两条不同轨迹。transfer_demonstration(record, 'same', 'same') 构造 entities={ref:ref_pos,actor:actor_pos}，两个键相同，后写的 actor 静默覆盖 ref；公开 world_process('transfer') 仍返回 status=ok，但元数据声称两角色都完成迁移。

在原生 transfer_demonstration 入口拒绝 actor==ref，抛出与 complete_process 同口径的 ValueError。core 直接记录和 memory_id 恢复两条路径都受此约束。不同目标角色的原坐标回放、移动参照物、平移/旋转/尺度与时刻偏移继续保留两条轨迹，source 仍为 generated。

对应 docs/world_process_plan.md 的相对轨迹示教迁移与角色重新绑定；不能以一个实体键表示两条不同角色轨迹。只是目标名称互斥，不限制复用或交换不同的原始名称。

基线 `891c28a7471a31275cc6d37a4468b62e49fe0694`；固定功能验证源码 `5fd13c811a6da11afe5f26c361411da9ec6839f6`。建议 P2/medium/B，最终定级、验收与结算由维护者决定。

## 集中验证

[原始证据](world-demonstration-distinct-roles-20261104/)；Windows / Python 3.12.10 / SQLite 3.49.1。

| 检查 | 实测 |
| --- | --- |
| 基线同一新守卫 | 4 failed、3 passed；0 errors、0 skipped |
| 修复后新守卫 | 7 passed；0 errors、0 skipped |
| 具名定点变异 | 2/2，各恰好一个指定断言失败；0 errors、0 skipped |
| 邻接回归 | 97 passed、1 skipped |
| 严格清单 | 64/64 MATCH |
| 时间核 | 0 新违规，23 条既有登记 |

具名变异：allow_output_role_collision、reject_distinct_source_name_reuse。邻接跳过：test_h1_record_round_trips_through_brain；跳过不计 passed。
只更新导出清单中 world_process.py 的现有指纹；未新增模块或修改 core、CI、依赖、过程识别阈值。同步调整 test_process_ops 的 M9：在隔离副本中同时撤掉 compose 与 transfer 的同名闸，才能验证同名保护机制被破坏后确实判红；M1 至 M8 仍用真实 transfer，全部原有断言保留。首批相关回归发现 M9 被新增的下层闸兜住，原始失败日志保留本地；最终读数来自修正变异范围后冻结源码的一批。

## 复跑与边界

```text
python -X utf8 -m pytest tests/test_demonstration_distinct_roles.py -q
python -X utf8 tools/testing/replay_save_guards.py --spec tests/fixtures/demonstration_distinct_roles_mutations.json --baseline 891c28a7471a31275cc6d37a4468b62e49fe0694 --mutate --output <新证据目录>
python -X utf8 tools/verify_manifest.py --strict --quiet
```

新守卫实际调用模块、临时 SQLite 引擎及 core 的 extract/load/transfer，核对记忆原文不变。沿用 transfer 对非法参数抛 ValueError 的既有契约，没有把它改成 error dict。没有写入真实观测或改动脑端。未测 DSH、真实设备或生产数据。
全状态 705 Issues / 471 PRs 已核对。#1136/#1143 处理非有限 scale 及 scene_simulator 读数；同名角色限制原先仅在 process_ops.complete_process，本件补齐 transfer_demonstration 的独立入口，不重复申报已修尺度问题。
本地 Linux 重放与变异未跑；远端 CI 单列。未运行压力测试或真实设备验收。
本批临时数据库和隔离副本已清理；公开日志仅脱敏路径并归一换行/行尾空白，本地保留原始结果。
一个独立 PR 对应一个 intake，human review pending。
