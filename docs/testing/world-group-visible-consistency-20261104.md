# 运动组合预测尊重当前可见偏移冲突

先由共同平移历史检测 a/b/c 组，再给当前观测 a.x=5、b.x=20。历史 b 相对 a 的偏移是 2，当前已经变成 15；公开 predict_groups 仍任选 a 作锚点，返回缺失 c.x=9、actual=False，却不报告无法确定。选择 b 会得到另一个位置，两者矛盾。

预测前按组模型保存的 tol_pos 两两核对当前可见成员的水平偏移。存在矛盾时缺失成员进入 unresolved，reason=conflicting_observed_anchors，附冲突成员对及偏差，不给坐标。单可见锚点、容差内观测、异高成员与 generated/actual=False 语义保持。六位小数偏移序列化每轴至多 1e-6 的双偏移差另按 sqrt(2)*1e-6 补偿，仅用于预测阶段的表示误差；未放宽学习判据。

对应 docs/world_process_plan.md 的成员独立运动后重算组合及只按真实可见锚点补位；旧组与新观测矛盾时应保留不确定，不能任选一项生成确定位置。

基线 `891c28a7471a31275cc6d37a4468b62e49fe0694`；固定功能验证源码 `38a0a1fc6a047f55990691d9c4531bd9ce8229b4`。建议 P2/medium/B，最终定级、验收与结算由维护者决定。

## 集中验证

[原始证据](world-group-visible-consistency-20261104/)；Windows / Python 3.12.10 / SQLite 3.49.1。

| 检查 | 实测 |
| --- | --- |
| 基线同一新守卫 | 4 failed、5 passed；0 errors、0 skipped |
| 修复后新守卫 | 9 passed；0 errors、0 skipped |
| 具名定点变异 | 3/3，各恰好一个指定断言失败；0 errors、0 skipped |
| 邻接回归 | 46 passed、1 skipped |
| 严格清单 | 64/64 MATCH |
| 时间核 | 0 新违规，23 条既有登记 |

具名变异：ignore_visible_conflict、check_canonical_anchor_only、forget_serialization_precision。邻接跳过：test_h1_record_round_trips_through_brain；跳过不计 passed。
只更新导出清单中 motion_groups.py 的现有指纹；未增加模块或修改既有测试、CI、依赖、学习容差。

## 复跑与边界

```text
python -X utf8 -m pytest tests/test_motion_group_anchor_consistency.py -q
python -X utf8 tools/testing/replay_save_guards.py --spec tests/fixtures/motion_group_anchor_consistency_mutations.json --baseline 891c28a7471a31275cc6d37a4468b62e49fe0694 --mutate --output <新证据目录>
python -X utf8 tools/verify_manifest.py --strict --quiet
```

新守卫实际调用 core.world_process('predict_groups') 与原生模块，使用临时 SQLite 引擎和合成观测。只处理当前多个可见成员之间的水平矛盾，不声称用一个可见成员就能检测隐藏成员独立运动，不添加竖直方向分组判据。未测真实相机、DSH 或生产轨迹。
全状态 705 Issues / 471 PRs 已核对。#1078 处理 world_learner 的结构锚点确认状态，是不同模块与契约；既有 motion_groups 判据和组形成后的多锚点预测是两个阶段，本件只修后者。
本地 Linux 重放与变异未跑；远端 CI 单列。未运行压力测试或真实设备验收。
本批临时数据库和隔离副本已清理；公开日志仅脱敏路径并归一换行/行尾空白，本地保留原始结果。
一个独立 PR 对应一个 intake，human review pending。
