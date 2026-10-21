# 结构事件与普通写入统一锁顺序

日期：2026-10-21

普通 add_node 先取得 SQLite 写事务再等 Python 锁，结构工作单元反向持 Python 锁等 BEGIN IMMEDIATE，形成等待环。record_event、注意力配置更新、验证标准终裁三个正常入口均可超时失败；结构事件未落库。

建议优先级 P1；最终优先级与验收由维护者裁定。
修前固定基线 `77c738235846839c33b1932d11e7e61778d427d0`；生产验证提交 `2084a3f94baadae8a2a88a616f07123e386c49d5`。
生产文件 `lingshu/core/core.py`，LF SHA256 `2711fa7949e4ed504f4ec705f918b66bdc7132a35a65120af1dede9c6cbeb1fe`。

## 修复及边界

仅修改 _structure_event_transaction、adjust_attention_weight、引擎 adjudicate_verifier_standard。统一先预留 SQLite 写事务，再取得 Python 锁；两个配置入口的真实提交仍放在持锁及内存补偿 try 内，提交失败恢复运行态。调用方已有事务仍只用 SAVEPOINT，不提交调用方工作。

判定：部分验证。真实 SQLite 与公开写入 API；仅在锁入口及 SQLite trace 回调调度一个合法线程交错，不替换数据库或写方法。测试将各连接 busy_timeout 从默认 30000ms 调为 300ms，仅缩短复现等待。提交失败由真实 DEFERRABLE 外键产生。未验线上触发频率、生产库、DSH 宿主、多宿主 OS 账户隔离或所有其它锁的全局正确性；不声称修复其它事务出口缺陷。

## 可复跑

检出修复提交，安装 `.[full,dev]`，在仓根执行：

```text
python -X utf8 -m pytest tests/test_structure_write_lock_order.py -q
python -X utf8 tools/testing/replay_save_guards.py --spec tests/fixtures/structure_write_lock_mutations.json --baseline 77c738235846839c33b1932d11e7e61778d427d0 --mutate
```

| 平台 | 同一守卫在基线 | 修复后 | 具名变异 |
|---|---|---|---|
| Windows / Python 3.12.10 | 3 failed, 4 passed in 1.87s | 7 passed in 0.64s | 5/5 指定用例与断言命中 |
| Linux / Python 3.13.12 | 3 failed, 4 passed in 2.77s | 7 passed in 1.83s | 5/5 指定用例与断言命中 |

基线通过的对照项不算新增缺陷。每个变异仅运行指定用例，要求精确用例失败并出现指定断言；error、超时、SKIP 不算命中。源码与守卫指纹、退出码、逐条读数及原始失败正文见同名目录。

| 变异 | 指定用例 | 必须出现的断言 |
|---|---|---|
| take_structure_lock_before_sqlite | test_structural_and_ordinary_writers_do_not_invert_locks[record_event] | structural writer failed under concurrent ordinary write |
| attention_lock_before_sqlite | test_structural_and_ordinary_writers_do_not_invert_locks[attention] | structural writer failed under concurrent ordinary write |
| attention_commit_after_compensation_scope | test_commit_failure_restores_runtime_configuration[attention] | attention runtime was published before a failed commit |
| verifier_lock_before_sqlite | test_structural_and_ordinary_writers_do_not_invert_locks[verifier] | structural writer failed under concurrent ordinary write |
| verifier_commit_after_compensation_scope | test_commit_failure_restores_runtime_configuration[verifier] | verifier runtime was published before a failed commit |

## 相关回归与显式跳过

```text
python -X utf8 -m pytest tests/test_structure_writer.py tests/test_structure_events.py tests/test_structure_anchor.py tests/test_structure_record_extra_tags.py tests/test_issue153_verifier_standard_replay_guard.py tests/test_issue768_engine_meta_governance_guard.py tests/test_adjudicate_identity_guard.py tests/test_issue118_governance_state_identity_guard.py tests/test_issue665_nested_tx_savepoint.py tests/test_issue317_import_tx_ownership_guard.py tests/test_core_thread_safety.py tests/test_manifest_integrity.py tests/test_public_docs_privacy_guard.py -q
python -X utf8 tools/verify_manifest.py --strict --quiet
```

- Windows：`118 passed, 1 skipped, 468 warnings in 22.15s`。
- Linux：`119 passed, 482 warnings in 65.22s (0:01:05)`。

Windows 1 条 SKIP：环境缺 OpenSSL CLI，TLS 证书与受认证进程用例未运行；对应 Linux 使用原生文件系统临时目录真实通过。初次在 Windows 挂载盘跑该 TLS 用例因 0700 权限语义不满足而 setup error，记录保留本地；未放宽权限守卫。新守卫和五条变异均无 SKIP、error 或超时。时间核扫描无新增违规，未改其登记、阈值或断言。

两平台严格登记核验 64/64 MATCH。全部相关回归的条数、跳过原因与临时目录已清理状态见 `related-results.json`。本地未跑全仓门禁，远端 CI 以 PR 最终 head 为准。

公开日志仅替换工作区/临时目录根、规范化换行和行尾空白；完整原始记录保留本地。Linux 驱动的初次 HEAD 探针误读共享 Git 目录，保留原观测并另以源码、守卫、规格和验证提交逐字节复核；见 `source-provenance.json`，不拿错误探针作提交证据。

## 查重与交付

对照 #184/#665、PR #813/#820 与已合入 #835/#836：本件修锁顺序形成的等待环，不把事务归属、退出失败回滚或自我模型基线恢复重新申报。三个入口按同一根因合并交付，不按入口拆单。

两项独立修复各一份 PR；本文的工具、规格、日志只属于该修复的验证，不另算贡献。维护者独立人审待完成。
