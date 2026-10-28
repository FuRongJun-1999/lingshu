# 世界模型观测窗口运动统计复用 · 验收记录

日期：2026-10-28

## 结果与范围

同一观测视图内，学习/推断先读 motion 再读 motion_detail，旧实现会为每个实体重复回填历史并估计速度。
本件仅改原生 `lingshu/world/motion.py`：按视图、按实体复用原始估计和上下文；诊断仍按需构造，公开可变字典与四元组缓存分开。
数值公式、非有限与溢出闸、真实时间差、稀疏证据计数、关系和预测判据均保持。

固定基线 `cebbe423779bd25cb106998a6e1f4ca2a4e262ad`；最终验证/计时源码 `2ea4835ec987d45d87d27cac37525b40e9bda5ff`。
代码、守卫、规格、基准和依赖的 LF 归一指纹在 [原始证据](world-motion-stat-reuse-20261028/) 中。
motion.py 是原生模块，严格导出清单不含它，本件未修改清单或模块数量。

## 集中验证

| 检查 | 实测 |
| --- | --- |
| 当前 main，同一新守卫 | 8 failed、9 passed；0 errors / 0 skipped |
| 优化后 | 17 passed；0 errors / 0 skipped |
| 具名变异 | 4/4，各恰好 1 个指定断言失败；0 errors / 0 skipped |
| Windows 相关回归 | 205 passed、2 skipped |
| 严格清单 | 64/64 MATCH |
| 时间核 | 0 新违规；23 条既有登记 |

具名变异：recompute_between_accessors（重新计算）、read_mutable_public_estimate（读公开可变字典）、
share_cache_between_views（跨视图共享旧值）、force_diagnostics_for_motion_only（强制生成未请求诊断）。
它们分别验证性能工作量、调用方改写隔离、新观测刷新和按需计算。
公开 core 的 world_learner.learn / world_model.generate 守卫核对每实体仅估计一次，并断言数据库无额外写入。

本地两条 SKIP：`test_scene_relations_round_trip_real_brain`、`test_experience_real_brain_restart_preserves_predictive_relations`，原因是未配置脑端依赖，不计 passed。
没有改既有测试、CI、阈值或依赖。仅在开发集中跑相关面，全套由远端 CI 执行。

## 配对性能实验

Windows / Python 3.12.10，16 组固定合成输入；每组一对预热，随后 11 对随机先后顺序的改前/改后调用。
实际计时入口是 core.world_learner("learn") 与 core.world_model("generate")。
两种实现使用同一 core/模型代码，只替换 ObservationWindow 为固定 main 源码类或最终优化类；模型构造、结果编码在计时外，双方计时段均关闭 GC。
每对使用全新模型。返回、节点、历史、模型参数/模式、诊断、预测缓存及关系历史联合指纹在改前改后完全一致：16/16，且没有新增 SQLite 写入。

| 入口 | 输入 | 实体 / 历史帧 | 改前中位数 ms | 改后中位数 ms | 加速比 |
| --- | --- | --- | ---: | ---: | ---: |
| world_learner.learn | dense | 16 / 512 | 1.4666 | 1.2508 | 1.1725× |
| world_model.generate | dense | 16 / 512 | 12.6247 | 11.8062 | 1.0693× |
| world_learner.learn | dense | 64 / 512 | 15.6144 | 14.9826 | 1.0422× |
| world_model.generate | dense | 64 / 512 | 78.0224 | 88.5314 | 0.8813× |
| world_learner.learn | dense | 128 / 512 | 64.7035 | 59.6195 | 1.0853× |
| world_model.generate | dense | 128 / 512 | 191.8044 | 197.2278 | 0.9725× |
| world_learner.learn | single | 64 / 2000 | 2.9897 | 2.0925 | 1.4288× |
| world_model.generate | single | 64 / 2000 | 9.9602 | 8.7460 | 1.1388× |
| world_learner.learn | single | 64 / 8000 | 3.3730 | 2.3118 | 1.4590× |
| world_model.generate | single | 64 / 8000 | 42.3288 | 41.4103 | 1.0222× |
| world_learner.learn | random_four | 64 / 2000 | 3.7711 | 3.2383 | 1.1645× |
| world_model.generate | random_four | 64 / 2000 | 50.8243 | 53.7799 | 0.9450× |
| world_learner.learn | empty_tail | 64 / 2000 | 2.5515 | 1.8208 | 1.4013× |
| world_model.generate | empty_tail | 64 / 2000 | 95.3268 | 93.3794 | 1.0209× |
| world_learner.learn | long_occlusion | 64 / 4000 | 42.5344 | 22.3376 | 1.9042× |
| world_model.generate | long_occlusion | 64 / 4000 | 64.1497 | 31.4166 | 2.0419× |


长遮挡输入为 64 实体，前 2 帧真实观测、随后缺失至 4000 帧；它测量长期缺失后的上下文回填，不是生产轨迹。
该组学习入口 42.5344→22.3376 ms（1.9042×），完整生成入口 64.1497→31.4166 ms（2.0419×）。
单实体稀疏采样的学习入口为 1.4288× / 1.4590×。

**没有全场景加速结论**：本批 dense/64 的 Unified 78.0224→88.5314 ms，耗时增加 13.47%；dense/128、random_four Unified 也分别为 0.9725×、0.9450×。
这些负向读数原样保留；本轮不能把它们确定归因为调度噪声或代码回归。收益只按已测稀疏/遮挡场景申报，不用于全项目性能承诺。
单独 motion() 不强制生成诊断由守卫验证，未另做该 accessor 的微基准；内存峰值、并发与吞吐未测。缓存复用不降低既有算法的渐进复杂度。

首轮曾让 motion() 直接触发完整诊断；收紧为原始估计复用、诊断按需后，新增相应守卫，冻结最终源码再集中验证和计时。
首轮 16 条守卫/3 个变异及计时保留在本地，不混入本表最终读数。本轮两批测试临时副本都已清理。

## 复跑

在仓根安装 .[full,dev] 后：

```text
python -X utf8 -m pytest tests/test_world_motion_stat_reuse.py tests/test_world_inference_window.py tests/test_issue418_sparse_motion_guard.py tests/test_world_motion_finite_arithmetic.py -q
python -X utf8 tools/testing/replay_save_guards.py --spec tests/fixtures/world_motion_stat_reuse_mutations.json --baseline cebbe423779bd25cb106998a6e1f4ca2a4e262ad --mutate --output <新证据目录>
python -X utf8 bench/bench_world_motion_reuse.py --baseline cebbe423779bd25cb106998a6e1f4ca2a4e262ad --repeats 11 --output <新计时文件.json>
python -X utf8 tools/verify_manifest.py --strict --quiet
```

重放只替换规格列出的 motion.py；严格重现应检出最终固定验证提交。
公开日志只归一工作区/解释器前缀、换行和行尾空白，原日志保留本地。

## 边界与查重

本地 Linux 重放、变异与性能未运行，远端 Linux CI 单独列出。真实设备、在线脑/MCP、DSH 与生产数据未测。
保持单视图历史稳定的既有使用方式；不提供共享视图并发事务或跨 tick 缓存。
基准证明的是这些输入的调用耗时与结果等价，不证明通用预测精度、实机速度或整体系统加速。

已核对 353 个全状态 PR、682 个全状态 Issue。既有 PR #460 引入当轮观测/实体对共享视图；
PR #615 补稀疏样本与诊断，PR #1080 修有限运动算术。本件处理当前 main 仍存在的两种 accessor 重复估计，
沿用三者规则，不重复申报既有算法与数值修复，也不按 accessor、模型或平台拆成多项交付。
维护者人审与架构核验待完成；human review pending。
