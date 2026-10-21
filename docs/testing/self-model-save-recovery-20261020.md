# 自我模型保存失败后的合并基线恢复

日期：2026-10-20

基线：`40fceb69e79d0f481455f05ea8581b6f6f5e5816`。生产仅改
`lingshu/core/core.py` 的 `_save_self_model` / `_merge_self_model_value`，LF 归一
SHA256 为 `641df4b1edc90a77ff790ce8efbdb96217af897b5fd04090bc4a1cdfe48b43d4`。
同名目录保存双平台基线/修复/变异日志、逐条结果和真实 API 读回。

## 缺陷与修复范围

建议优先级 P1，由维护者裁定。#483 的三方合并依赖 `_self_model_baseline` 判断字段
是否是本地新改动，但原回调在 SQL 写入前就把它更新为 local。真实 SQLite 拒绝写入
或 COMMIT 后，盘上仍是旧快照、内存仍有待保存改动，基线却已当成保存成功。下一次
正常保存因此把待保存字段判为未改动，改采盘上旧值；这次保存成功后，重启仍丢数据。

双平台实际 API：一次价值修正遇真实触发器拒写；移除故障，正常更新另一字段后重启。
基线仍读到默认 values、value_evolution=0；修复后读到本地修正、value_evolution=4，
后一次正常更新的 goal 也保留。该后果与“写失败的当下盘上不更新”不同：故障解除后的
下一次成功保存仍无法保存此前的内存修正，才是本件修复点。

修复让合并回调只算载荷，捕获本次 local 输入，在真实写入/自主提交成功后才推进
相同输入的基线。先取得 SQLite 写事务，再进入原有 store 锁，沿用 merge_meta 锁顺序，
不在持 store 锁时等待首次 BEGIN。成功基线仍为 local，不能改成 merged，否则会复活
#483 的跨实例覆盖问题。没有新增重试线程：待保存改动随下一次正常保存重试。

保留 `_persist` 的既有旁路失败策略（失败不阻断内存模型变更）、三方字段归属规则、
同字段后写者胜、合并不回填实例内存。仅保证本件独立保存的语句/提交失败后不误推进
基线；调用方已开事务、随后自行回滚的运行态同步不在本件范围，不新增该保证。

## 可复跑

检出修复提交，安装 `.[full,dev]`，在仓根运行：

```text
python -X utf8 -m pytest tests/test_self_model_save_recovery.py -q
python -X utf8 tools/testing/replay_self_model_guards.py --spec tests/fixtures/self_model_save_mutations.json --baseline 40fceb69e79d0f481455f05ea8581b6f6f5e5816 --mutate
```

| 平台 | 同一守卫在基线 | 修复后 | 具名变异 |
|---|---|---|---|
| Windows / Python 3.12.10 | 6 failed, 3 passed, 73 warnings in 1.67s | 9 passed, 73 warnings in 1.67s | 4/4 命中 |
| Linux / Python 3.13.12 | 6 failed, 3 passed, 73 warnings in 4.48s | 9 passed, 73 warnings in 4.12s | 4/4 命中 |

9 个新守卫无 SKIP；6 个红例是 values/value_evolution、trust_state/trust_history、
current_goal 分别遭遇语句拒绝与真实 COMMIT 拒绝。另三条是成功路径和 #483 跨实例
保真对照，旧基线本来应通过，不作为新增缺陷计数。

语句失败使用 SQLite 触发器 `RAISE(ABORT)`；COMMIT 失败使用真实 DEFERRABLE 外键，
不替换 sqlite3 的写入/提交实现。确认失败快照逐位未变、无事务遗留；移除故障后
正常写入另一字段，再另开引擎读回，检查待保存字段与历史逐位保持。

## 固定变异

| 变异 | 指定用例 | 必须命中的断言 |
|---|---|---|
| publish_baseline_before_snapshot_write | test_pending_change_survives_rejected_save_and_later_recovery[values-statement] | pending self-model change was dropped after recovery |
| publish_baseline_before_commit | test_pending_change_survives_rejected_save_and_later_recovery[values-commit] | pending self-model change was dropped after recovery |
| never_advance_successful_baseline | test_successful_local_baseline_preserves_later_remote_values | successful baseline was not advanced before a later remote update |
| advance_baseline_to_merged_disk_payload | test_adopted_disk_fields_do_not_become_local_baseline | adopted disk values were overwritten on the next save |

每个变异在新副本只改 AST 指定方法中的唯一片段，只跑指定用例，要求指定用例精确失败
并出现指定文本；error、超时、SKIP 不计作命中。完整源码/守卫指纹与退出码随记录交付。
工具、守卫和日志属于修复验证，不单列贡献。

## 相关回归

```text
python -X utf8 -m pytest tests/test_issue483_self_model_merge_guard.py tests/test_issue33_self_model_restore_guard.py tests/test_self_model_history.py tests/test_issue182_self_model_gate_guard.py tests/test_issue206_trust_state_guard.py tests/test_core_thread_safety.py tests/test_issue665_nested_tx_savepoint.py tests/test_issue317_import_tx_ownership_guard.py tests/test_manifest_integrity.py tests/test_public_docs_privacy_guard.py -q
python -X utf8 tools/verify_manifest.py --strict --quiet
```

- Windows：74 passed, 337 warnings in 20.08s；无 SKIP。
- Linux：74 passed, 345 warnings in 55.59s；无 SKIP。
- 登记严格核验 64/64 MATCH；组件分类 D 组 6 条断言通过。
- 日志仅替换临时输出根、规范化行尾空白并去掉 WSL 启动诊断；完整原始记录保留本地。
- 未验：本地全量门禁、生产自我模型、线上触发频率、DSH 宿主、磁盘满/断电，以及上述
  调用方稍后回滚的运行态同步。远端全量门禁以 PR 当前提交检查为准。

## 查重与最小性

对照 #33（持久化/恢复）、#483（写前读回、三方合并）、#182（字段写闸）、#206
（信任参数），本件是 #483 修复中基线早于 SQL 成功推进的残余面，保留这些既有判据。
PR #813 是 core 事务出口回滚，#820 是账本/TMS，#835 是 activation；它们均不管理
本地模型合并基线，也不是本件前置依赖。本件不按字段或失败类型拆单。
