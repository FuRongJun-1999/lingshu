# 写事务退出失败后的回滚验证

基线：`40fceb69e79d0f481455f05ea8581b6f6f5e5816`。

## 问题与修复边界

`LayeredStore._tx` 原先只捕获 `yield` 内的方法体异常，正常退出的
`conn.commit()` / `RELEASE SAVEPOINT` 放在 `try ... else` 中。
SQLite 拒绝这两步时不会执行回滚，失败写入和锁留在连接上。

从真实 `SpacetimeMemoryEngine.subgraph_replace` 入口注入一次 COMMIT 拒绝后：
旧子树在本连接中已被删除、独立连接中仍存在；`in_transaction=True`。
随后 `add_context` 返回节点 ID，但独立连接看不到它。
原因是后续操作被当成遗留事务的 SAVEPOINT，未提交到磁盘。

本修复只将正常 COMMIT/RELEASE 移入原有回滚保护范围，不改提交权判据。
嵌套失败仍 `ROLLBACK TO` 本层 SAVEPOINT，调用方已有的事务仍由调用方处理。
相关历史：#184（方法体异常）、#665（嵌套提交）、#317（调用方提交权）；
本件补的是这些修复之后仍遗漏的**正常退出阶段本身失败**。

## 可复现证据

SQLite 的真实 authorizer 一次性拒绝 COMMIT 或 RELEASE；不替换连接，不模拟
事务实现。SQLite 负责产生异常和持有/释放真实锁，独立连接核验落盘结果。
仅使用测试临时库及合成节点。

```text
python -X utf8 -m pytest tests/test_transaction_exit_rollback.py -q
python -X utf8 tools/testing/check_transaction_exit_rollback.py --baseline 40fceb69e79d0f481455f05ea8581b6f6f5e5816 --mutate
```

第二条命令把当前包复制到临时目录，以 `git show` 的原始 core.py 替换基线副本，
不修改工作树。输出原始 pytest 日志、JUnit 和带源码 SHA256 的 summary.json。
命令会检查失败用例名称、失败断言和退出码；超时/收集错误不算杀死变异。

| 环境 | 基线守卫 | 修复守卫 | 定点变异 |
| --- | --- | --- | --- |
| Windows / Python 3.12.10 | 3 failed / 3 passed | 6 passed | 4/4 被指定断言杀死 |
| WSL Linux / Python 3.13.12 | 3 failed / 3 passed | 6 passed | 4/4 被指定断言杀死 |

Windows 相关回归：新守卫及 `test_issue665_nested_tx_savepoint.py`、
`test_issue317_import_tx_ownership_guard.py`、`test_issue139_subgraph_replace_guard.py`、
`test_issue139_subgraph_replace_validate_guard.py`、`test_core_thread_safety.py`，
合计 **43 passed，0 skipped**。该数字包含新守卫，不与上表相加。
另同步既有导出登记中 core.py 的 LF 归一指纹；登记守卫 **20 passed**，
`tools/verify_manifest.py --strict --quiet` 为 **64 MATCH，0 MISMATCH，exit 0**。

| 变异 | 指定守卫 | 必须失败的断言 |
| --- | --- | --- |
| `commit_outside_rollback_boundary` | G1 公共子图替换 | failed COMMIT left a transaction pending |
| `release_outside_rollback_boundary` | G2 调用方持有事务 | failed RELEASE left its inner write pending |
| `rollback_caller_instead_of_savepoint` | G2 调用方持有事务 | caller transaction was ended by inner recovery |
| `transaction_depth_not_reset` | G4 方法体抛错 | transaction depth was not reset after a body failure |

G3 另守卫“外层捕获内层 RELEASE 失败后继续提交”，G5 是正常子图替换的落盘正对照；
G4 同时覆盖普通 Exception 与 KeyboardInterrupt。具名守卫会随 tests/ 自动进入门禁。

同目录的 `windows-summary.json`、`linux-summary.json` 记录逐项结果和源码哈希；
`windows-*.txt` 是原始执行输出。Linux 使用 Windows 工作树时只需为本次命令
指定映射后的 `GIT_DIR`；普通 Linux clone 无此需要。

## 限制与待验收项

- 这是确定性提交故障注入，不是磁盘满、掉电或真实生产负载测试。
- 现有回滚清理仍为 best effort；若 SQLite 连回滚也拒绝，本件不承诺连接已恢复。
- 本机仅运行上述相关回归；全仓矩阵以 PR 当前 head 的 CI 为准。
- 未运行真实 DSH 宿主、Blender 或可选重载模型；它们不是这个 SQLite 边界的前置。
- 人工技术审查、理论裁定和最终优先级/难度/影响等级待维护者决定。
