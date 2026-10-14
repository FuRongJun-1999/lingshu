# PR #460 · bounty #34 修复复验（2026-10-12）

Unified 关系准入收紧后，旧关系撤销仍只看实测间隔数，导致一两段方向支持夹杂静止与缺帧时删除旧 seek。本次将撤销条件对齐学习器：至少三段有效方向样本，或至少三段共同实测、零方向样本且源轨迹没有运动，才构成更新/撤销依据。支持不足保留旧关系，实测停止仍撤销；共同移动但同位置不视为停止。

本次生产差异只在 `UnifiedWorldModel.infer_patterns` 的 sufficient 判据，复用当轮 ObservationWindow.motion 缓存。没有改准入阈值、预测公式、历史结构、依赖、CI 收集面、异常处理或 SimulationLoop 假设生命周期。

基线 `cb3a6ea2dbd9771d685b2bf5c1fe398abe69e419`；待修原 head `6921932b15db31d4be2c150e90f8924f7fab132b`。[原始评估](https://github.com/FuRongJun-1999/dsh-bounty/issues/34)保留旧版本失败事实，不用新结果覆盖历史。

## 行为与反证

同一份 `test_pr460_review_relation_retention.py`：基线 4 passed；原 head 2 failed / 2 passed，失败精确为 one-direction 与 two-directions；修复版 4 passed。该文件还显式检查保留时置信度及 relation_history 不变、实测停止时记录撤销。

原有 M8/M10 守卫先断言关系数，再取元素。共同移动守卫同时覆盖学习器与 Unified。两个文件合计 **18 passed / 0 failed / 0 errors / 0 skipped**。

定点变异 **12/12 检出**，每个指定单用例恰为 1 failed / 0 errors / 0 skipped。新增 M11 回退 Unified 的 sufficient 判据，M12 去掉 Unified 的零运动要求；M8/M10/M11/M12 均在关系数业务 AssertionError 处失败，变异工具额外核验其异常信息，不再把 IndexError 当这四项的成功反证。其余变异沿用原机制，M9 仍以掩码被改坏后的 KeyError 识别，不将其描述成显式业务断言。

112,000 次原始统计差分、0 差异；严格导出清单 61/61。导入方向、覆盖目录、静默宽异常 lint 及 git diff --check 均通过，未改 lint 基线。

## 本地门禁与预测

Windows / Python 3.13.13：完整 pytest **1290 passed、3285 warnings、73 subtests passed**，356.40 秒，0 failed / 0 errors / 0 skipped；按原 gate 将脚本式件单独运行，未排除两个 NN 重件。18 个脚本式件全部退出 0；真实脑包固定 `98d6b899533a82fb1047f952987229c62a9929fb`，隔离库 Brain 对接 **23/23 PASS**。本地安装了可选 SciPy，故不能把本地无 skip 与远程默认环境的 skip 数直接比较。

原 head 和修复版运行相同 bench_world.py（60 种子 × 9 场景）：**540/540 case 全字段相同，summary 与 contracts 相同**。仅说明既有受控场景无新增预测回归，不证明真实环境泛化。

## 推断性能复跑

完整测试结束后重跑两版本同一基准，每组五次。输入指纹逐组相同，13/14 输出指纹相同；随机四实体的 Unified 26→0 仍是原 PR 的三段方向准入变化。本轮数据不与原评估计时拼接。

| 模型 | 实体 / 帧 / 输入 | 基线 ms | 修复 ms | 输出相同 |
|---|---|---:|---:|---|
| learner | 16 / 512 / dense | 1.068 | 0.781 | 是 |
| unified | 16 / 512 / dense | 1.288 | 1.280 | 是 |
| learner | 64 / 512 / dense | 16.806 | 12.960 | 是 |
| unified | 64 / 512 / dense | 20.325 | 16.557 | 是 |
| learner | 128 / 512 / dense | 67.503 | 50.462 | 是 |
| unified | 128 / 512 / dense | 79.947 | 66.088 | 是 |
| learner | 64 / 2000 / single | 324.406 | 0.767 | 是 |
| unified | 64 / 2000 / single | 326.018 | 0.962 | 是 |
| learner | 64 / 8000 / single | 1344.441 | 0.793 | 是 |
| unified | 64 / 8000 / single | 1328.051 | 0.958 | 是 |
| learner | 64 / 2000 / random_four | 79.403 | 1.637 | 是 |
| unified | 64 / 2000 / random_four | 79.374 | 2.194 | 否（26→0） |
| learner | 64 / 2000 / empty_tail | 6.960 | 0.721 | 是 |
| unified | 64 / 2000 / empty_tail | 7.450 | 0.993 | 是 |

原始逐次计时、输入/输出/生产源码指纹、12 项变异的 nodeid 与失败信息、JUnit 摘要、脚本退出码均公开于 [复验证据 JSON](testing/pr460_review_fix_evidence.json)。证据文件的 tested_source_lf_sha256 标识实际测试源码；测试在文档提交前完成，未把当时 checkout parent 的 SHA 冒充最终远程 head。

## 复跑入口

```sh
python -X utf8 -B -m pytest tests/test_world_inference_window.py tests/test_pr460_review_relation_retention.py -q -p no:cacheprovider
python -X utf8 -B tools/testing/check_world_inference_mutations.py
python -X utf8 -B tools/testing/check_world_statistics_equivalence.py --baseline PATH_TO_CB3A6EA
python -X utf8 -B tools/verify_manifest.py --strict --quiet
python -X utf8 -B bench/bench_world.py --seeds 60 --output artifacts/predictions.json
python -X utf8 -B bench/bench_world_inference.py --repeats 5 --output artifacts/inference.json
```

完整回归沿用 .github/workflows/gate.yml 的 SCRIPT_TESTS 分离协议。本地 pytest 使用仓库 dev extra 已声明的 xdist；公开性能不设 CI 时间阈值。

## 验收边界

本轮补齐原评估 C3 的关系生命周期守卫与 C6 的指定反证；C7 的远程最终 head 门禁由提交后 Actions 核验，链接和实际读数回帖到 bounty #34。维护者人审 C8、理论核、难度/影响等级与结算仍由维护者裁定；作者账户复验不作为独立维护者签字。没有执行 /verify、/settle 或修改账本。
