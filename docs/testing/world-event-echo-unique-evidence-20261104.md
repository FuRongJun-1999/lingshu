# 事件回声去除重放导致的伪支持与评分重复

事件回声已有 _echo_key 将 tick/eid/event 定义为事件身份，但 _echo_stream 仅排序不去重。一组 stop@1 → turn@3 本来不足 min_support=2；把同一组行复制一次即可返回 count=2、consistency=1 的关联。重复行还改变按事件数切分的留出边界和评分分母，并不代表新增观测。

在共用事件流入口按完整 (tick,eid,event) 去重并排序，同一身份保留首条正文。学习、切分与两种模型评分共用这一流；evaluate_event_echo 的 dev_events 同步按去重后的身份计数。同 tick 的不同实体或不同事件仍分别保留，不把同一事件类型在不同时刻的真实重复发生合并。

对应 docs/world_process_plan.md 第 2 组的延迟关联、开发/留出对照及已有事件身份契约；支持数必须来自不同逻辑事件，重放不能制造额外证据。

基线 `891c28a7471a31275cc6d37a4468b62e49fe0694`；固定功能验证源码 `8d0937751ae1a6f2b52508628b740a85037aadb9`。建议 P2/medium/B，最终定级、验收与结算由维护者决定。

## 集中验证

[原始证据](world-event-echo-unique-evidence-20261104/)；Windows / Python 3.12.10 / SQLite 3.49.1。

| 检查 | 实测 |
| --- | --- |
| 基线同一新守卫 | 6 failed、1 passed；0 errors、0 skipped |
| 修复后新守卫 | 7 passed；0 errors、0 skipped |
| 具名定点变异 | 3/3，各恰好一个指定断言失败；0 errors、0 skipped |
| 邻接回归 | 147 passed、1 skipped |
| 严格清单 | 64/64 MATCH |
| 时间核 | 0 新违规，23 条既有登记 |

具名变异：count_replayed_rows_again、report_raw_development_count、collapse_distinct_same_tick_events。邻接跳过：test_h1_record_round_trips_through_brain；跳过不计 passed。
只更新 world_process.py 的现有导出指纹；未增加模块、改动既有测试、CI、依赖、支持数门槛或滞后判据。

## 复跑与边界

```text
python -X utf8 -m pytest tests/test_event_echo_unique_evidence.py -q
python -X utf8 tools/testing/replay_save_guards.py --spec tests/fixtures/event_echo_unique_evidence_mutations.json --baseline 891c28a7471a31275cc6d37a4468b62e49fe0694 --mutate --output <新证据目录>
python -X utf8 tools/verify_manifest.py --strict --quiet
```

实际运行原生公开分析函数 echo_lag_associations/echo_split/evaluate_event_echo，输入为合成事件流。core.echo 从运动历史正常提取的事件已是唯一身份，本件不声称它支持上传任意 raw events，也不声称该正常通路已观察到重复行。门槛、lag、留出比例公式和因果边界保持；两段各自去重，不新增跨段泄漏检测。未测 DSH、真实设备或生产日志。
全状态 705 Issues / 471 PRs 已核对，未见相同三键事件重放计数根因。#724 处理历史 tick 非递增除零，事件流行重复与原始观测时刻非法是两件不同问题。另一本轮角色绑定 PR 改同文件的迁移函数，是独立机制；导出清单的同一文件指纹在合并时须按最终源码重登记。
本地 Linux 重放与变异未跑；远端 CI 单列。未运行压力测试或真实设备验收。
本批临时数据库和隔离副本已清理；公开日志仅脱敏路径并归一换行/行尾空白，本地保留原始结果。
一个独立 PR 对应一个 intake，human review pending。
