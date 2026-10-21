# 激活工作态的事务归属与失败回滚

日期：2026-10-20

基线为 `40fceb69e79d0f481455f05ea8581b6f6f5e5816`。生产改动仅在
`lingshu/core/activation.py`，LF 归一 SHA256 为
`218c8ab92525e58651f8e09af4d3afaee51e0b64103ac3325723a38ec0e38ddd`。
同名目录保存双平台逐条结果、实际 API 读回和原始测试输出。

## 问题与修复

建议优先级为 P1，最终由维护者裁定。`ActivationEngine(store=...)` / `conn=...`
借用的是调用方正在使用的 SQLite 连接，但构造期建表和正常 `activate()` 写工作态
仍直接 `commit()`。调用方先开启事务并删除一个普通节点，再构造或激活，随后
`rollback()` 已无法恢复该节点；另开连接读回也确认删除已永久提交。

工作态替换还会先 DELETE 再 INSERT。真实 SQLite 触发器拒绝 INSERT 后，旧工作态
从两行变成零行，借用连接停在未结束事务中。`import_workset` 已修复语句异常，
但 COMMIT 和 RELEASE 原先在 `try` 的 `else` 分支，出口失败仍绕过回滚。

三条写路径统一使用模块内的 `_transaction`：没有事务时 BEGIN IMMEDIATE，
已有事务时只管理自己的 SAVEPOINT；语句、COMMIT、RELEASE 失败均回滚本次工作，
保留调用方原有写入与提交权。独立文件连接仍由本模块关闭，借用连接仍不关闭。
不依赖尚未合并的 core / 证据账本修复。

首次建表也参与调用方事务。若调用方随后回滚首次建表，须重新构造激活引擎；
守卫实际验证回滚后重建并激活成功。不新增运行时依赖，不改变传播公式、来源/hop、
工作集返回体或 JSONL 审计格式。JSONL 审计不参与 SQLite 事务，未新增跨介质原子承诺。

## 可复跑

检出修复提交，安装 `.[full,dev]`，在仓根运行：

```text
python -X utf8 -m pytest tests/test_activation_transaction_ownership.py -q
python -X utf8 tools/testing/replay_activation_guards.py --spec tests/fixtures/activation_transaction_mutations.json --baseline 40fceb69e79d0f481455f05ea8581b6f6f5e5816 --mutate
```

| 平台 | 同一守卫在基线 | 修复后 | 指定变异 |
|---|---|---|---|
| Windows / Python 3.12.10 | 11 failed, 3 passed in 1.14s | 14 passed in 0.91s | 5/5 命中 |
| Linux / Python 3.13.12 | 11 failed, 3 passed in 3.31s | 14 passed in 3.44s | 5/5 命中 |

新增守卫与变异均无错误、超时或 SKIP。基线的三条通过项为已存在的 import 语句
异常回滚（独立/外层事务）与正常独立文件往返，不把它们写成新增缺陷。
实际 API 三场景在双平台相同：

| 场景 | 基线 | 修复 |
|---|---|---|
| 调用方删除 → 构造 → rollback → 新连接读回 | 原节点 0 | 原节点 1 |
| 调用方删除 → activate → rollback → 新连接读回 | 原节点 0 | 原节点 1 |
| INSERT 被真实触发器拒绝 | 工作态 2→0，事务遗留 | 工作态 2→2，无事务遗留 |

COMMIT 故障使用真实 DEFERRABLE 外键，不替换 sqlite3 的 commit 实现；RELEASE
故障由连接 authorizer 定点拒绝首次 RELEASE，后续回滚语句仍执行真实 SQLite。
初始化故障同样由 authorizer 拒绝索引创建，验证新表整体回滚并能重试。

## 具名变异

| 变异 | 指定用例 | 指定断言 |
|---|---|---|
| commit_borrowed_caller_transaction | test_activate_preserves_caller_rollback[store] | activation committed caller deletion |
| leave_statement_failure_unrolled_back | test_failed_replacement_restores_previous_workset[activate-standalone] | failed activation discarded the previous workset |
| commit_failure_bypasses_rollback | test_commit_failure_rolls_back_workset_and_trigger_rows[import] | activation commit failure left staged changes |
| retain_failed_savepoint_replacement | test_release_failure_undoes_only_activation_work[import] | failed activation release kept partial replacement |
| initialize_without_atomic_schema | test_failed_schema_initialization_rolls_back_created_table | failed activation initialization left a partial schema |

工具在新副本中按 AST 定位类/方法，每个替换要求唯一命中；每个变异只跑指定用例，
要求恰好该用例失败、出现指定文本且没有 error/SKIP。源码和守卫指纹随记录保存。
不修改工作树，也不将“其他异常导致测试红”计作命中。

## 相关回归与范围

```text
python -X utf8 -m pytest tests/test_activation_empty_graph.py tests/test_activation_graph_refresh.py tests/test_issue7_161_293_314_53_activation_contract.py tests/test_issue229_280_activation_suppression_bounded.py tests/test_issue709_cond_match_shape_guard.py tests/test_issue714_missing_content_tables_guard.py tests/test_issue771_778_import_atomicity_guard.py tests/test_manifest_integrity.py tests/test_public_docs_privacy_guard.py tests/test_import_direction_lint.py -q
python -X utf8 tools/verify_manifest.py --strict --quiet
```

- Windows：156 passed, 16 skipped, 211 warnings in 21.86s。
- Linux：172 passed, 211 warnings in 107.57s (0:01:47)，无 SKIP。
- Windows 的 16 条 SKIP 全部因缺可选 SciPy 稀疏后端；逐条名称和原因见
  `related-results.json`，对应 Linux 稀疏后端实跑通过。
- 指纹严格核验 64/64 MATCH；既有组件分类 D 组 6 条断言通过。
- 时间核登记保持原 R3 条目和表达式，只将 activation.py 的登记行号 298→335。
  新增事务代码使原位置移动；初轮 CI 因陈旧位置有 5 项相关失败，未跳过断言或新增豁免。
  补同步后 `test_time_core_lint.py` 与 `test_issue607_time_core_shapes_guard.py`
  合跑 74 passed in 161.96s (0:02:41)；全仓时间核扫描 new=0、registered=23、stale=0。
  原始输出见 `time-core-related.log`，结构结果见 `time-core-results.json`。
- 与独立自我模型保存修复 PR #836 合用：将两个提交的真实包文件和原封不动的两份
  守卫放入同一隔离副本，23 passed, 73 warnings in 1.97s，无 SKIP；源文件 SHA256
  和组件提交在 `joint-activation-self-model.json`，临时副本已关闭并移除。
  这只是兼容性联测，不构成额外贡献或 DSH 宿主验收。
- 公开日志仅替换临时输出根、规范化行尾空白，并去掉 WSL 启动诊断；完整原始记录
  在本地保留。失败正文、读数、断言不改写。
- 未验：本地全量门禁、DSH 宿主、生产记忆库、线上触发频率和磁盘满/断电。
  远端完整门禁以 PR 当前提交的实际检查结果为准。

## 查重与边界

本件是激活模块自身漏接的事务边界。对照 core 的 #184、#665、PR #743 / #813，
以及证据账本/TMS 的 PR #820，它们均不修改 activation 的三条写路径。
已有 #293 是连接复用，#314 是来源/hop，仓内 #771 守卫已覆盖 import 的语句异常；
本件保留该既有行为，只补构造/正常激活和退出失败。当前远端 #771/#778 返回 404，
不据此声称已读到这些讨论的全文。传播算法方面的 PR #830 与本件正交。
维护者可以将其归为同族问题后续收口；不按三个入口拆成三份贡献。
