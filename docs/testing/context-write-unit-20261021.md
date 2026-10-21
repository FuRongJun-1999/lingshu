# 情境写入以持久化容量和一次事务为准

日期：2026-10-21

修前 main `049dd79d902fcbb5d0b097fc5e1f7cf3197dae44`；最终守卫验证提交 `33c55085e31309dd4baa7b6608266fd452837f7a`。生产源码 `lingshu/core/core.py`，LF SHA256 `6ff4f5b9126d0d019d53e189ed147a01441679449f9886f43e0be211dc883882`。

复核已有 #246 的存活实例场景：实例 B 已在容量 2 时打开，A 调为 5 并存 4 条，B 正常写入一条仍按陈旧副本裁到 2 条；反向缩容后 B 则超过已提交的上限。第二个真实 Python 进程同样复现。原报告者 cyd471 的贡献保留，本轮不将原缺陷算作新发现。

另在真实 SQLite 删除触发器拒绝 FIFO 时，修前新节点已先提交：调用报错，独立连接仍读到 3 条而非原来的 2 条。在调用方事务中，本次失败的插入也留在其事务里。

## 修复与边界

只修改 `SpacetimeMemoryEngine.add_context`：复用既有 `store._tx()`，插入、读取持久化容量和 FIFO 共用写事务。已有调用方事务时只撤回自己的 SAVEPOINT。`_context_max` 兼容观察值、容量设置入口、归一化、FIFO 插入顺序和保护优先的既有规则不改。

判定为部分验证，建议优先级 P1。使用公开 Python API、隔离文件库、真实独立进程与真实 SQLite；未修改数据库实现。未验 DSH 宿主、生产记忆、线上发生频率、断电/磁盘满、吞吐或所有其它锁的正确性。未修改大脑区，没有新增运行时依赖。

## 固定读数与复跑

检出上述固定验证提交，安装 `.[full,dev]`，在仓根运行：

```text
python -X utf8 -m pytest tests/test_context_write_unit.py -q
python -X utf8 tools/testing/replay_save_guards.py --spec tests/fixtures/context_write_mutations.json --baseline 049dd79d902fcbb5d0b097fc5e1f7cf3197dae44 --mutate
```

| 平台 | 同一守卫在基线 | 修复后 | 具名变异 |
|---|---|---|---|
| Windows / Python 3.12.10 | 5 failed, 2 passed in 1.48s | 7 passed in 1.47s | 4/4 指定用例及断言命中 |
| Linux / Python 3.13.12 | 5 failed, 2 passed in 3.76s | 7 passed in 3.35s | 4/4 指定用例及断言命中 |

基线两条通过项是调用方主动回滚与浮点容量/受保护节点的对照，不算新缺陷。最终新增守卫、基线对照及变异均无 error、SKIP 或超时。每条变异只运行指定用例，必须恰好该用例失败并出现指定断言。

| 变异 | 指定用例 | 必须出现的断言 |
|---|---|---|
| reuse_stale_instance_capacity | test_existing_instance_uses_current_database_capacity[increased] | context write used a stale or default capacity |
| ignore_smaller_persisted_capacity | test_existing_instance_uses_current_database_capacity[decreased] | context write used a stale or default capacity |
| commit_insert_before_fifo | test_fifo_failure_undoes_the_new_context_on_disk | failed context write was committed before FIFO failed |
| omit_fifo_trimming | test_successful_context_call_can_be_rolled_back_by_caller | successful context write did not trim FIFO inside caller transaction |

## 相关回归

```text
python -X utf8 -m pytest tests/test_issue246_context_cap_persistence_guard.py tests/test_context_capacity_atomicity.py tests/test_issue493_context_cap_after_import_guard.py tests/test_issue259_context_fifo_insertion_order_guard.py tests/test_issue36_forget_protection_guard.py tests/test_issue665_nested_tx_savepoint.py tests/test_transaction_exit_rollback.py tests/test_issue317_import_tx_ownership_guard.py tests/test_issue768_engine_meta_governance_guard.py tests/test_core_thread_safety.py tests/test_manifest_integrity.py tests/test_public_docs_privacy_guard.py -q
python -X utf8 tools/verify_manifest.py --strict --quiet
python -X utf8 tools/time_core_lint.py --json
```

- Windows：`100 passed, 488 warnings in 22.33s`。
- Linux：`100 passed, 492 warnings in 92.50s (0:01:32)`。

两平台相关回归均无 SKIP、failure 或 error，64/64 严格登记一致，时间核无新增违规；未改登记阈值或豁免。相关回归与守卫均验证生产提交 `33c55085e31309dd4baa7b6608266fd452837f7a`；最终守卫采用诊断文件与有界就绪检查，详见 `source-provenance.json`。

初次 Windows 独立进程夹具发生输出管道等待超时，已改为诊断文件与有界就绪检查后完整复跑；原超时和清理异常保留本地，不计通过、不改生产断言。公开日志只规范化换行/行尾空白并替换工作区前缀，完整原始记录保留本地。

## 查重

对照 #246（原容量持久化）、#493（导入后的本实例刷新）、PR #819（容量设置原子性）、#813（底层事务失败回滚）、#841（结构写锁顺序）。本件修复普通情境写入消费陈旧容量及插入/FIFO 分离的问题；复用这些已合入的事务原语，未重报它们。两个问题属于同一情境写入工作单元，一份 PR 交付，不按入口或平台拆单。维护者人审待完成。
