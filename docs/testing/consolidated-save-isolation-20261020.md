# 固化模型并发保存的临时文件隔离

日期：2026-10-20

基线：`40fceb69e79d0f481455f05ea8581b6f6f5e5816`。
本件只修改身体侧 `lingshu.nn.hex_cnn.save_consolidated` 的临时文件归属，
生产代码指纹为 `f94d2cbfdd88781c879b1412810ab8a9b40cbff4a0180f4cb7b62c8a61e72926`。

## 缺陷与实际入口

旧实现只用目标路径和 PID 命名临时文件。同进程两个线程保存同一个模型时，
一个线程的 rename 或异常清理会消费另一线程的临时文件。

以真实 `save_consolidated`、`load_consolidated` 和临时模型文件复现：
线程 A 写完后暂停在 rename 前，让线程 B 完成，再释放 A。屏障只控制先后顺序，
JSON 序列化、fsync、rename、清理与模型读回均为真实实现。

| 情境 | 改前 | 改后 |
|---|---|---|
| A、B 都提交合法模型 | B 成功，A 抛 FileNotFoundError | 两者成功，最终为最后完成 rename 的 A |
| B 序列化失败 | B 抛 TypeError，A 也抛 FileNotFoundError，旧模型保留 | 仅 B 报错，A 正常保存且可加载 |

两种情境均在 Windows 和 Linux 实测；对应原始 API 读数见同名目录下的
`windows-baseline.json`、`windows-fixed.json`、`linux-baseline.json`、`linux-fixed.json`。
修复后的正常情境没有临时残件。

## 修法与兼容边界

每次保存使用固定短前缀加 PID、UUID 的独立临时文件，以 `x` 排他创建；只有本次成功创建的
文件才能进入异常清理。极端命名冲突直接抛 FileExistsError，既不截断、也不删除
另一调用持有的文件。新文件仍按普通 open 的 umask 创建，已有目标的权限继承、
realpath 符号链接处理、同目录原子替换与返回值保持原约定。

临时 basename 不复制目标 basename，长度与模型文件名无关，避免追加 UUID 导致组件名超限。
Linux 上 231 字符模型名已完成真实保存与加载，读数见 `linux-long-name-fixed.json`。
跨平台新增守卫在真实 open/序列化失败路径观察短名与长名，要求临时 basename 等长；
Windows 本机原有长路径限制仍在，未将临时文件守卫当作长目标保存成功的证明。

同一目标取最后一次成功 rename 的完整模型；不承诺调用顺序、自动合并或跨进程锁。
目录项掉电持久性、真实磁盘满、DSH 宿主、真实训练任务和模型效果未验。

## 守卫与具名变异

在当前工作树根目录安装 `.[full,dev]` 后运行：

```text
python -X utf8 -m pytest tests/test_consolidated_save_isolation.py -q
python -X utf8 tools/testing/replay_save_guards.py --spec tests/fixtures/consolidated_save_mutations.json --baseline 40fceb69e79d0f481455f05ea8581b6f6f5e5816 --mutate
```

| 平台 | 同一新守卫在基线的原始汇总 | 修复后的原始汇总 | 具名变异 |
|---|---|---|---|
| Windows / Python 3.12.10 | `4 failed, 1 passed in 0.52s` | `5 passed in 0.36s` | 4/4 命中指定断言 |
| Linux / Python 3.13.12 | `4 failed, 1 passed in 0.86s` | `5 passed in 0.83s` | 4/4 命中指定断言 |

四个变异均固定在规格文件，只运行其指定用例，并要求恰好一个失败、指定断言文本
出现、无错误与 SKIP；超时不能算命中。

| 变异 | 指定断言 |
|---|---|
| share_process_temporary_file | concurrent valid save was rejected |
| truncate_existing_temporary_file | temporary-name collision was not rejected |
| unlink_unowned_temporary_file | another save's temporary file was removed |
| grow_temp_basename_with_model_name | temporary basename grew with model filename |

新守卫在两平台均无 SKIP。replay 工具复用此前事务修复的验证框架，增加模块级函数
定位；工具只是验证配套，不另算独立贡献。每次对子进程固定沙箱源码归属，保留源码、
守卫 SHA256 和真实退出码。原始记录及逐条输出见同名目录的 `*-summary.json` 与日志；
公开日志仅替换临时输出目录根为 `<output>`，并规范化行尾空白。

## 定向回归与尚未全绿的 Windows 旧守卫

```text
python -X utf8 -m pytest tests/test_hex_cnn_atomic_save_guard.py tests/test_issue498_atomic_write_perm_symlink_guard.py tests/test_manifest_integrity.py tests/test_public_docs_privacy_guard.py -q
python -X utf8 tools/verify_manifest.py --strict --json
```

- Linux：`56 passed, 57 warnings in 21.70s`，无 SKIP。
- Windows：`1 failed, 49 passed, 6 skipped, 41 warnings in 6.96s`；不是全绿。
- Windows 唯一失败为旧 `test_p1_符号链接量具能力探测`：本机能创建符号链接，但原守卫
  在成功分支断言必须 POSIX。未修改的基线单跑同一用例也是 `1 failed in 0.67s`，
  断言为「P1 非 posix 上 symlink 竟成功（开发者模式？）——SKIP_LINK 可重核」。
- 6 条 Windows SKIP 是既有 POSIX 权限/符号链接用例，不计为通过；对应 Linux 实测通过。
- 指纹严格检查：64/64 MATCH；组件发现 D 组原有 AST 分类与自证共 6 条断言通过。
- 相关汇总及旧守卫基线断言见同名目录 `related-results.json`。完整 JUnit 与原始
  本机运行记录留在交付目录。首次回归命令误写一个不存在的测试文件名，退出 4、零测试执行，
  随后已纠正；该命令错误没有计入任何通过读数。

未修改旧守卫、未调整断言或添加 deselect/xfail。未跑完整项目门禁，远端 CI 待 PR 发布后验证。

## 查重

对照 lingshu #393（单次序列化失败截断）、#498（权限与符号链接）、#349（核形状与
meta 键覆盖）、#698（写向根闸审计），并检索目标仓全部状态 issue/PR 与 bounty 的
save_consolidated、并发、临时文件关键词。#393 的单次写入原子性在 main 已修，
本件收口其 PID 临时文件在同进程并发时的残余冲突；未见同件交付。
若维护者认为属于既有任务的后续收口，归类与计分由维护者裁定。
