# 子树替换预检与写入共用事务快照

日期：2026-10-21

子树替换只把挂入和删除放进事务，父子范围、保护和后代集合仍先在事务外读取。另一实例在预检后、写事务开始前正常保护子节点，替换仍会删除其未保护的父节点并挂入新树；旧子树若刚移到另一父节点下，原调用仍会删除范围外节点；新加入的后代则被遗漏、成为残留。

只给 `SpacetimeMemoryEngine.subgraph_replace` 使用已有 `@_transactional` 入口，使独立调用先取写锁再读取范围、保护与后代；已有调用方事务则参与其快照并保留提交权。复用既有嵌套 SAVEPOINT 和内部挂入/删除事务，不新增锁或事务实现。

## 复现与判据

建议优先级 P1，判定为部分验证。使用两个正常引擎实例和真实文件库；SQLite trace 仅在第一条 BEGIN IMMEDIATE 执行前协调另一实例提交，不替换查询、结果或写方法。观察已提交保护/边、剩余节点和调用方回滚。
修前 main 固定为 `f5dcd58165cbf64393d1f2d4dfc61f781485ec29`；两平台验证提交 `657cbd982a19da5f8b6256b2f18ed365056b6828`。

| 平台 | 同一守卫在修前 | 修复后 | 具名变异 |
|---|---|---|---|
| Windows / Python 3.12.10 | 3 failed, 2 passed in 1.02s | 5 passed in 1.03s | 4/4 指定用例及断言命中 |
| Linux / Python 3.13.12 | 3 failed, 2 passed in 3.04s | 5 passed in 2.84s | 4/4 指定用例及断言命中 |

全部最终守卫、修前对照与变异无 error、SKIP 或超时；变异恰好命中指定单条用例及断言。

| 变异 | 指定用例 | 指定断言 |
|---|---|---|
| skip_entry_transaction | test_new_protection_rejects_the_whole_replacement | subtree protection was checked before write ownership |
| ignore_protected_subtree | test_new_protection_rejects_the_whole_replacement | subtree protection was checked before write ownership |
| ignore_current_parent_scope | test_moved_subtree_is_rechecked_before_deletion | subtree replacement deleted outside its current parent scope |
| omit_current_descendants | test_descendants_are_collected_after_write_ownership | subtree replacement used stale descendants |

- Windows：`25 passed, 153 warnings in 3.27s`，0 skipped。
- Linux：`25 passed, 153 warnings in 15.18s`，0 skipped。

严格登记 64/64 MATCH，时间核无新增违规；未改门禁、阈值、断言或豁免。本地仅跑上述受影响面，完整门禁交由最终 PR head 的远端 CI。不为整理材料重复跑稳定测试；子树项曾因上游 #821 改到其感知调用入口而补跑一次定向批，旧读数保留本地，本报告只统计对齐后结果。

安装 `.[full,dev]` 后，在上述固定验证提交的仓根执行：

```text
python -X utf8 -m pytest tests/test_subtree_write_snapshot.py -q
python -X utf8 tools/testing/replay_save_guards.py --spec tests/fixtures/subtree_write_snapshot_mutations.json --baseline f5dcd58165cbf64393d1f2d4dfc61f781485ec29 --mutate
python -X utf8 -m pytest tests/test_issue139_subgraph_replace_guard.py tests/test_issue139_subgraph_replace_validate_guard.py tests/test_issue665_nested_tx_savepoint.py tests/test_transaction_exit_rollback.py -q
python -X utf8 tools/verify_manifest.py --strict --quiet
python -X utf8 tools/time_core_lint.py --json
```

同目录公开两平台逐腿原始日志及来源校验。公开日志只替换工作区前缀、统一换行和行尾空白，完整原始输出保留本地；时间核完整 JSON 保留本地，其 SHA256 见相关日志。

## 来源与边界

原始范围/保护问题来自 cyd471 的 [lingshu #139](https://github.com/FuRongJun-1999/lingshu/issues/139)；LarryE135 的字段类型残留已由维护者在 [既有回执](https://github.com/FuRongJun-1999/lingshu/issues/139#issuecomment-6223385466)修复并在 bounty #684 交付。本件只补仍在 main 的预检/写入快照竞态，不将原报告或已修字段失败再次计分，不关闭覆盖更广的旧 Issue。对照 bounty #973（底层事务退出失败）和 #1051（证据/TMS 事务归属），本件不修改这些原语或重报其已交付工作。

保留既有深度 64、遍历默认 500 的边界，未修大子树截断、共享后代策略或运行态计数回滚；未宣称子图替换所有缺口均已闭环。 未修改 dsh-memory 大脑区；未验证 DSH 宿主、生产记忆、线上发生频率、吞吐、断电/磁盘满或所有其它锁。维护者人审待完成。
