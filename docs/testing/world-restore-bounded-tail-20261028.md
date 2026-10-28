# 世界模型显式历史恢复 · 有界工作集验收

日期：2026-10-28

## 结果与改动

`restore_trajectory_from_history()` 最终每实体只保留 256 点，旧实现却先构造全部历史的第二份归一化轨迹，再裁尾。
本件边转换边维护每实体 256 点的临时 deque，成功后发布原有 list 缓冲；完整 history 不裁剪。
每帧 int tick、每个坐标 float 转换、实体首次出现顺序、尾部顺序、返回值及误差队列重置保持。
转换失败时活动模型不半替换；即使坏记录最终会被裁掉，也仍转换并按原异常处理。

基线 `68f2c3ab22ed667832311bb3207f88e9294c1efd`；固定功能验证/计时源码 `548ee0cffc3e90a514df13cba71233fc03d4ef86`。
原生改动只在 world_model.py 的显式恢复方法及 deque 标准库导入；导出清单仅同步该文件指纹。
周期判据、生成算法、懒加载轨迹回填、接口签名、依赖与旧测试均未修改。
[全部原始证据](world-restore-bounded-tail-20261028/) 含输入/输出/源码指纹、每次计时、独立峰值、重放和初轮测量问题。

## 集中验证

| 检查 | 读数 |
| --- | --- |
| main，同一新守卫 | 2 failed、12 passed；0 errors / 0 skipped |
| 优化后 | 14 passed；0 errors / 0 skipped |
| 具名变异 | 4/4，各恰好一个指定断言失败，0 errors / 0 skipped |
| Windows 相关回归 | 226 passed、2 skipped |
| 严格清单 | 64/64 MATCH |
| 时间核 | 0 新违规；23 条既有登记 |

资源守卫在恢复入口给归一化 float 注入可观察的存活探针：4096 帧全部转换，
但同时存活的坐标最多为 `3 × (实体数 × 256 + 1)`；探针用于检查工作集，不把其字节数当真实分配。
最终数据类型、顺序、稀疏保留、原历史和无关状态、失败原子性、后续 perceive 继续运行均有控制。
使用合成 read_content 收据调用实际 `SimulationLoop.load_priors`，300 帧历史完整保留且尾缓冲与后续预测可用；该项不是在线脑/MCP 验收。

具名变异：unbounded_intermediate_tail（取消容量）、reverse_retained_observations（反转入队）、
trim_source_history（裁原历史）、publish_before_validation（校验前发布）。
没有放宽旧断言、改业务阈值或加 CI 豁免。

## 配对耗时与独立分配峰值

Windows / Python 3.12.10，7 组固定合成历史，每组一对预热、随后 9 对随机顺序计时。
直接调用实际恢复方法；只切换该方法为 main 的固定源码或最终源码，其他模型代码和输入相同。
计时不启用 tracemalloc；Python 分配峰值另测。历史构造、模型初始化、结果编码在测量外，双方 GC 策略相同。
返回点数、完整尾缓冲、两条误差队列的输出指纹 7/7 一致；原始 history 指纹与引用保持。

| 实体 / 帧 | 采样 | 改前 / 改后 ms | 加速比 | 改前 / 改后峰值 MiB | 峰值变化 |
| --- | --- | ---: | ---: | ---: | ---: |
| 4 / 128 | dense | 0.2360 / 0.2627 | 0.8984× | 0.0636 / 0.0708 | 增加 11.256% |
| 4 / 256 | dense | 0.4683 / 0.5030 | 0.9310× | 0.1265 / 0.1373 | 增加 8.557% |
| 16 / 512 | dense | 4.9321 / 4.2377 | 1.1639× | 1.0046 / 0.6687 | 降低 33.436% |
| 16 / 4000 | dense | 41.9318 / 31.8194 | 1.3178× | 7.8313 / 0.6686 | 降低 91.463% |
| 32 / 4000 | dense | 109.3572 / 97.4738 | 1.1219× | 15.6601 / 1.2140 | 降低 92.248% |
| 64 / 8000 | dense | 396.9983 / 349.0053 | 1.1375× | 62.7006 / 2.3049 | 降低 96.324% |
| 16 / 16000 | single | 17.5530 / 15.3756 | 1.1416× | 1.9689 / 0.6687 | 降低 66.038% |


64 实体 / 8000 帧：峰值 62.7006→2.3049 MiB，降低 96.324%；耗时 396.9983→349.0053 ms，1.1375×。
16 实体 / 4000 帧：峰值降低 91.463%，耗时 41.9318→31.8194 ms，1.3178×。
**短历史有代价**：4 实体 / 128、256 帧分别为 0.8984×、0.9310×，约增加 0.027/0.035 ms；峰值分别增加 11.256%/8.557%，约 7/11 KiB。
所有负向读数原样保留，不声称所有输入都会提速。

时间仍需遍历全部真实观测；只降低恢复阶段第二份轨迹的工作集，从随历史总观测数增长，改为受实体数×256 限制。
原始 history、JSON 解析、在线脑传输和进程 RSS 不在本峰值范围内；不把 96.324% 说成整个项目内存降幅。
源代码与方法层面的结果等价不证明真实设备、DSH 或生产脑的性能，本轮均未测试。

## 初轮量具修订与清理

初版新字节比例守卫对照 512/4096 帧，单实体实测 53264→164880 字节，超过初拟三倍比例；初轮为 1 failed、13 passed。
字节比例不能直接代表所保留归一化数据的上界，最终用对象存活计数验证有界工作集，真实字节数仍由独立 tracemalloc 实验完整给出。
没有把初轮失败计作通过；其日志在 initial-instrumentation-failure.log 保留。
生产恢复方法从首次实现起未因这次量具修订改变；最终守卫/规格提交后，集中重放与配对实验均在固定源码上执行。
两批临时副本已清理。首个 32 实体/4000 帧探针的耗时包含追踪开销，baseline-probe.json 明示其性质，不当作最终延迟。

## 复跑与边界

安装 .[full,dev] 后在仓根执行：

```text
python -X utf8 -m pytest tests/test_world_restore_bounded_tail.py tests/test_world_prior_atomic_restore.py tests/test_trajectory_resolved_observations.py tests/test_issue455_subprocess_encoding.py -q
python -X utf8 tools/testing/replay_save_guards.py --spec tests/fixtures/world_restore_bounded_tail_mutations.json --baseline 68f2c3ab22ed667832311bb3207f88e9294c1efd --mutate --output <新证据目录>
python -X utf8 bench/bench_world_restore_tail.py --baseline 68f2c3ab22ed667832311bb3207f88e9294c1efd --repeats 9 --output <新计时文件.json>
python -X utf8 tools/verify_manifest.py --strict --quiet
```

重放只替换规格内 world_model.py，其余依赖取运行检出；严格重现应检出固定验证提交。
公开日志只归一工作区/解释器前缀、换行和行尾空白，原日志留本地。
本地 SKIP：test_scene_relations_round_trip_real_brain、test_experience_real_brain_restart_preserves_predictive_relations，未配置脑端，不计 passed。
本地 Linux 重放/变异/性能、在线生产脑/MCP、DSH、真实设备、生产数据、并发与进程 RSS 未测；远端 CI 单独记录。
范围是显式历史恢复；懒加载 _wm_traj_buf 和周期计算的开销仍在。

## 查重与交付范围

本轮核对 358 个全状态 PR、688 个全状态 Issue，未见同一显式轨迹恢复峰值优化。
上一项 [PR #1088](https://github.com/FuRongJun-1999/lingshu/pull/1088) 已合入本基线，优化的是单窗运动估计重复计算；本件是历史恢复的临时空间，与其分别可复现、可独立验收。
旧先验恢复、匿名观测与原子性修复的行为契约继续保留，本件不重复申报这些功能。
一个独立 PR/intake，不按输入规模或入口字段拆分；维护者人审、定级和结算待完成，human review pending。
