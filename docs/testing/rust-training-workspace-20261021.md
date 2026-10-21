# Rust 训练桥接的独占工作目录

日期：2026-10-21

复核已有 Issue #392，报告者与维护者已确认的固定路径串扰在此基线上仍成立。两个普通并行调用会覆写 data/plan/vec/out 四个文件，成功返回另一任务的曲线或权重。默认调用还会覆盖调用方同名旧文件。

建议优先级 P1；最终优先级与验收由维护者裁定。
修前固定基线 `77c738235846839c33b1932d11e7e61778d427d0`；生产验证提交 `a33215ff5275cdc120efe5854ed33d2f6e4567da`。
生产文件 `lingshu/nn/rust_bridge.py`，LF SHA256 `6df44dd3788c2890019af6327170d0cfe7bbe01a149924b896f8301060f43dc1`。

## 修复及边界

仅修改 run_rust_train：默认目录每次独占并在成功、子进程失败、解包失败时清理；显式 workdir 仍由调用方拥有、保留诊断文件。输出曲线长度及权重长度必须与本次计划和初始向量一致。执行器定位、环境白名单、训练参数与二进制格式保持原约定。

判定：部分验证。真实子进程协议夹具独立读二进制输入，用样本数和初始向量见证任务归属；仅调度两次启动发生在打包完成之后，不替换生产打包/解包实现。不把该夹具称为 Rust 训练器。未验实际 hex_nn 可执行文件、训练算法等价性、真实训练吞吐、DSH 宿主与线上数据。显式工作目录的并发隔离仍由调用方负责，不新增这方面保证。

## 可复跑

检出修复提交，安装 `.[full,dev]`，在仓根执行：

```text
python -X utf8 -m pytest tests/test_rust_training_workspace.py -q
python -X utf8 tools/testing/replay_save_guards.py --spec tests/fixtures/rust_training_workspace_mutations.json --baseline 77c738235846839c33b1932d11e7e61778d427d0 --mutate
```

| 平台 | 同一守卫在基线 | 修复后 | 具名变异 |
|---|---|---|---|
| Windows / Python 3.12.10 | 5 failed, 2 passed in 1.25s | 7 passed in 1.08s | 5/5 指定用例与断言命中 |
| Linux / Python 3.13.12 | 5 failed, 2 passed in 1.37s | 7 passed in 1.51s | 5/5 指定用例与断言命中 |

基线通过的对照项不算新增缺陷。每个变异仅运行指定用例，要求精确用例失败并出现指定断言；error、超时、SKIP 不算命中。源码与守卫指纹、退出码、逐条读数及原始失败正文见同名目录。

| 变异 | 指定用例 | 必须出现的断言 |
|---|---|---|
| reuse_default_shared_directory | test_concurrent_default_calls_keep_their_own_inputs | training result belongs to another call |
| leave_owned_temporary_directory | test_default_call_preserves_caller_files | owned training workspace was not cleaned |
| accept_wrong_curve_length | test_output_must_match_this_call_dimensions[short_curve] | DID NOT RAISE |
| accept_wrong_vector_length | test_output_must_match_this_call_dimensions[short_vector] | DID NOT RAISE |
| discard_explicit_diagnostic_workspace | test_explicit_workdir_retains_diagnostic_files | explicit workdir files were not retained |

## 相关回归与显式跳过

```text
python -X utf8 -m pytest tests/test_issue488_rust_exe_cwd_guard.py tests/test_manifest_integrity.py tests/test_public_docs_privacy_guard.py tests/test_issue253_extras_declared_guard.py -q
python -X utf8 tools/verify_manifest.py --strict --quiet
```

- Windows：`42 passed in 11.51s`。
- Linux：`42 passed in 31.66s`。

既有 #454 二进制格式守卫按其真实脚本入口在常规与 -O 两种模式通过；不运行该旧脚本的原地变异选项。42 条相关回归两平台均无 SKIP。没有新增运行时依赖。

两平台严格登记核验 64/64 MATCH。全部相关回归的条数、跳过原因与临时目录已清理状态见 `related-results.json`。本地未跑全仓门禁，远端 CI 以 PR 最终 head 为准。

公开日志仅替换工作区/临时目录根、规范化换行和行尾空白；完整原始记录保留本地。Linux 驱动的初次 HEAD 探针误读共享 Git 目录，保留原观测并另以源码、守卫、规格和验证提交逐字节复核；见 `source-provenance.json`，不拿错误探针作提交证据。

## 查重与交付

根因来自 [Issue #392](https://github.com/FuRongJun-1999/lingshu/issues/392)，不是本轮新发现；本轮提供独立实现、复现和守卫。对照 #350、#454、#488 与在飞 PR #838，分别保留环境、格式、执行器定位和计划规模的既有边界。

两项独立修复各一份 PR；本文的工具、规格、日志只属于该修复的验证，不另算贡献。维护者独立人审待完成。
