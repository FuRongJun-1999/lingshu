# 可选证据账本与真值维护的事务归属验证

日期：2026-10-19

基线：main `40fceb69e79d0f481455f05ea8581b6f6f5e5816`。

## 问题与改动

`EvidenceLedger` 和 `TruthMaintenance` 原先直接调用连接的 commit，绕过 core 的
事务归属机制。开启功能后，普通感知写入或依据登记会提前提交调用方尚未完成的工作；
外层子图替换失败也可能留下部分新节点。记账后同步或传播失败，则可能部分提交并持锁。

现在这些写入统一复用 `LayeredStore._structure_event_transaction`：无外层事务时
独立提交，已有事务时只释放自己的 SAVEPOINT。证据、置信度同步、依据登记与传播属于
同一工作单元，失败一并回滚。TMS 的传播缓存通过 finally 在成功和失败时都清空。

来源去重、Beta 置信度、TMS min/max 规则、角色限制和默认关闭策略不变。

## 首次初始化边界

首次启用功能需要在调用方事务之外完成，否则在建表和缓存运行对象之前抛 RuntimeError，
不会提交或回滚调用方事务。建表本身使用独立原子事务，建索引失败不留下半套表。
完成初始化后的记账、同步、依据登记和传播支持调用方事务及回滚。

这个明确拒绝需要维护者确认。它避免了另一种错误：允许建表参与调用方事务、却把对象
缓存到引擎中；调用方 rollback 撤掉表后，缓存对象仍被当作可用。

## 改前与改后

新增守卫：`tests/test_extension_transaction_ownership.py`，15 个用例。

| 检查 | Windows / Python 3.12 | Linux / Python 3.13 |
|---|---:|---:|
| main 基线运行同一组新守卫 | 14 failed / 1 passed | 14 failed / 1 passed |
| 修复后新守卫 | 15 passed | 15 passed |
| 新守卫与证据语义、TMS 传播、角色限制、治理闭环、嵌套事务的既有回归 | 84 passed | 84 passed |

关键可观察结果：

- 子图替换包含合法新根和非法子节点时，改前报错后残留 1 个新节点，改后为 0；旧根仍在。
- 调用方先删旧记忆，再调用感知写入、记证据、sync、依据登记或 propagate：改前提前
  提交、rollback 无法恢复旧记忆；改后调用方事务保留，rollback 能恢复旧记忆并撤回新写入。
- 传播中途或最终 COMMIT 被拒时，改后证据、依据和节点置信度一并回滚，连接不留悬挂事务。
- 首次初始化的索引创建失败时不残留部分表，关闭故障后可重试。

相关回归文件：

- `tests/test_evidence_ledger.py`
- `tests/test_tms.py`
- `tests/test_tms_truth_maintenance_guard.py`
- `tests/test_issue765_774_evidence_ledger_role_guard.py`
- `tests/test_issue769_780_governance_write_role_guard.py`
- `tests/test_governance_write_surface_closure_guard.py`
- `tests/test_issue665_nested_tx_savepoint.py`

运行新守卫：`python -X utf8 -m pytest tests/test_extension_transaction_ownership.py -q`。
基线复跑时，把同一个测试文件放入基线 checkout 的 tests 目录，再从该 checkout 运行；
若外置测试，需在收集前固定 core、evidence、tms 三个模块的实际源码归属。

## 范围与相关修复

本件是 #665 / PR #743 事务归属修复的跨模块漏口。PR #813 处理 core 的事务出口提交或
释放失败，本件处理可选模块在方法内部主动 commit，修改位置和触发路径不同。
本补丁直接基于 main，不依赖 #813。

仅在启用对应功能时影响上述业务写入。验证使用临时合成 SQLite 库和真实引擎入口，
没有 DSH、脑端、外部模型、生产数据或完整 CI 通过的结论。

## 具名变异自证

```text
python -X utf8 tools/testing/replay_transaction_guards.py --spec tests/fixtures/extension_transaction_mutations.json --baseline 40fceb69e79d0f481455f05ea8581b6f6f5e5816 --mutate
```

Windows 和 Linux 各 7/7 条变异命中。每次只运行预先指定用例，要求恰好该用例失败、指定断言文本出现，且无错误或跳过；收集失败和超时不算命中。

| 变异 | 指定断言 |
|---|---|
| `ledger_record_owns_caller_transaction` | extension committed the caller transaction |
| `ledger_sync_owns_caller_transaction` | extension committed the caller transaction |
| `tms_registration_owns_caller_transaction` | TMS committed the caller transaction |
| `tms_propagation_owns_caller_transaction` | TMS committed the caller transaction |
| `synchronize_after_record_commit` | failed record committed evidence |
| `allow_first_enable_inside_caller_transaction` | DID NOT RAISE |
| `retain_tms_cache_after_failure` | failed propagation left a stale runtime cache |

双平台逐条记录、源码/守卫 SHA256 和日志见同名目录。日志将临时输出目录前缀替换为 `<output>` 并移除行尾空白，保留失败正文、原始计数和退出码。完整未替换输出可由上述 driver 重新生成。

通用复跑 driver 与另一事务修复 PR 内容相同，属于配套验证工具，不作为额外独立贡献重复计算；各 PR 的变异规格和业务修复独立。
