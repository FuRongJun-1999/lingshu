# 容量设置失败与 FIFO 回滚验证

日期：2026-10-19

基线：main `40fceb69e79d0f481455f05ea8581b6f6f5e5816`。

## 问题与改动

`SpacetimeMemoryEngine.set_context_cap` 原先先更新运行值，再独立保存配置并逐条淘汰。
数据库写锁使保存失败时，运行值已经生效；下一次正常 `add_context` 会按这个未保存的
小容量删除旧记忆。裁剪中断还可能留下已经提交的配置和部分删除。

现在保存配置和 FIFO 裁剪复用 `LayeredStore._structure_event_transaction`，共用一条
事务，提交成功后才发布运行值。角色检查、数值归一、FIFO 顺序和保护节点判据保持原语义。

## 接口边界

调用方已有事务时，在任何改动前抛 `RuntimeError`，要求先结束调用方事务；不会提交或
回滚调用方的工作。这与已有 `adjudicate_verifier_standard` 的独立配置事务做法一致。
原因是调用方稍后的 SQLite rollback 无法撤回 Python 属性；单纯调换两行不能解决该路径。
这一明确拒绝是兼容性边界，需要维护者审查确认。

## 改前与改后

新增守卫：`tests/test_context_capacity_atomicity.py`，7 个用例。

| 检查 | Windows / Python 3.12 | Linux / Python 3.13 |
|---|---:|---:|
| main 基线运行同一组新守卫 | 6 failed / 1 passed | 6 failed / 1 passed |
| 修复后新守卫 | 7 passed | 7 passed |
| 新守卫与容量、导入刷新、数值归一、治理角色、保护、FIFO、提案存活的既有回归 | 51 passed | 51 passed |

新守卫覆盖真实第二连接占用写锁、第二次删除时触发 SQLite 错误、最终 COMMIT 被拒，
以及 BEGIN/SAVEPOINT 调用方事务。失败后检查运行值、盘上配置、原有节点、连接事务状态，
并用独立连接确认持久状态。正常成功检查只在最终提交后发布新值，保留既有保护节点策略。

锁失败的公开入口场景：容量 10、已有 3 条记忆，设置为 1 时保存报错。改前下一次普通
写入会删光原有 3 条；改后容量仍为 10，正常写入后共 4 条，原有 3 条均在。

相关回归文件：

- `tests/test_issue246_context_cap_persistence_guard.py`
- `tests/test_issue493_context_cap_after_import_guard.py`
- `tests/test_issue655_context_max_overflow_guard.py`
- `tests/test_issue768_engine_meta_governance_guard.py`
- `tests/test_issue36_forget_protection_guard.py`
- `tests/test_issue259_context_fifo_insertion_order_guard.py`
- `tests/test_issue167_promotion_node_liveness_guard.py`

运行新守卫：`python -X utf8 -m pytest tests/test_context_capacity_atomicity.py -q`。
基线复跑时，把同一个测试文件放入基线 checkout 的 tests 目录，再从该 checkout 运行，
避免跨工作树收集测试时混入不同版本的惰性模块。

## 验证范围

使用临时合成 SQLite 库及真实引擎入口，无 DSH、脑端或外部模型依赖。没有模型效果、
生产数据、完整 CI 或完整回归通过的结论。本补丁不改变多实例配置同步策略，也不保证
所有受保护节点都能被裁到容量之内；沿用既有 FIFO 判据。

## 具名变异自证

```text
python -X utf8 tools/testing/replay_transaction_guards.py --spec tests/fixtures/context_capacity_mutations.json --baseline 40fceb69e79d0f481455f05ea8581b6f6f5e5816 --mutate
```

Windows 和 Linux 各 4/4 条变异命中。每次只运行预先指定用例，要求恰好该用例失败、指定断言文本出现，且无错误或跳过；收集失败和超时不算命中。

| 变异 | 指定断言 |
|---|---|
| `publish_runtime_before_commit` | runtime cap was published before the single final commit |
| `save_metadata_outside_fifo_transaction` | failed capacity change changed stored metadata |
| `allow_existing_caller_transaction` | DID NOT RAISE |
| `omit_fifo_trimming` | successful capacity change did not trim FIFO |

双平台逐条记录、源码/守卫 SHA256 和日志见同名目录。日志将临时输出目录前缀替换为 `<output>` 并移除行尾空白，保留失败正文、原始计数和退出码。完整未替换输出可由上述 driver 重新生成。

通用复跑 driver 与另一事务修复 PR 内容相同，属于配套验证工具，不作为额外独立贡献重复计算；各 PR 的变异规格和业务修复独立。
