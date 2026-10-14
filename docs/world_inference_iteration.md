# 世界模型推断窗口迭代

> 2026-10-12 评审修复：原 head 的 Unified 混合证据撤销回归已修正，
> 本轮守卫 18 项、定点变异 12 项及完整门禁见 [bounty #34 修复复验](pr460_review_fix.md)。
> 以下原始实验记录对应 `6921932`，保留历史读数，不混作修复版读数。

本次基于 `cb3a6ea2dbd9771d685b2bf5c1fe398abe69e419`，收束 WorldLearner 与
UnifiedWorldModel 的观测统计，并消除无共同观测时的历史回溯。沿用世界图、关系边、
shadow 预测、目标时刻验证和七层闭环；运行时仍仅使用标准库。

## 源码读取与问题定位

当前物理世界通过 `observe/perceive` 提供位置、类别；模型从真实观测历史估计速度和
方向持续性，通过位移与实体间方向的余弦推断 seek/flee。预测在临时 shadow 上按实体
顺序推进，验证对照真实观测快照；好奇调度与 SimulationLoop 消费这些结果。
记忆适配、视觉几何与语义图是关联的其他路径，不在本次补丁范围。

两条运动模型的关系统计重复实现，存在两个证据语义差异：

1. 学习器原有“至少三段方向样本”门槛，但 UnifiedWorldModel 的关系选择没有使用它。
   三帧、两段位移就能在后者建立新 seek 关系。
2. 学习器把“没有方向样本”都当作证据不足。实体持续被观测且已经静止时，旧关系仍被
   标成 stale 并保留；应区分静止的实测间隔与缺失观测。

性能问题也在同一数据流：每个有向实体对重新扫描窗口，并在缺帧时查旧历史，连窗口内
从未共同出现的实体对也如此。若每帧只观察一个实体，实体对不可能产生关系样本，
仍会发生约 `E×(E−1)×H` 次无用历史检查。

## 实现与判据

`motion.ObservationWindow` 是单次推断的局部视图。它归一两种内部观测格式，记录
每个实体在当前窗口的出现位图，复用单实体运动统计。实体对的位图交集为空时，
关系样本数为零，不再读取旧帧；交集非空时仍保留原有的窗口前共同观测上下文。
两个方向的余弦和样本计数在同一次遍历中计算。

关系统计分别返回：平均余弦、有效方向样本数、实测间隔数。零位移或同位置的实测
间隔计入第三项，不作为方向支持；缺失观测两项都不贡献。运动估速继续按真实 dt
计算，均值、方向持续性、舍入和最近方向规则不变。

- 新关系仍需至少三段有效方向支持；这是学习器已有门槛，现由统一模型一起使用。
- 学习器保留不足三段的方向候选；至少三段共同实测间隔且源轨迹没有运动时，允许移除旧关系。
  “有一段方向支持、其余静止”的混合稀疏证据不被误当成全窗口静止。
  两实体同位置而共同移动时，目标方向不可求，不被当作静止反证。
- SimulationLoop 假设关系的保留规则不变；关系强度阈值、点命中阈值与半径公式不变。

视图只活在一次 `learn/infer_patterns` 调用内，没有跨 tick 缓存、失效协议、新依赖、
调参开关或兜底预测。合法的观测不足直接体现为样本不足；错误输入不被捕获后伪装成结果。

## 验证

新回归在原版本上复现两个行为失败，四个健康对照通过；候选版新增 13 用例通过。
另外将 200 个随机历史、5 种窗口、两种观测格式、两种方向与缺失实体的统计逐项
对照未修改基线：112,000 次比较，0 差异。这验证的是统计计算等价，不把新增的
关系准入修正算作旧行为等价。

本地整仓快档（仅排除两个已登记重件）为 1280 passed、73 subtests passed、
3283 warnings，452.36 秒。随后补充“同位置运动不是静止”的最后一条判据，最终受影响
测试集合为 266 passed、294 warnings，17.13 秒；含全部 13 个新增用例。
严格清单 61/61 匹配，导入方向、覆盖目录与静默异常 lint 均通过，没有改其基线。
本地未运行 Python 3.12；远程门禁另按最终提交核验。测试耗时不作为模型加速证据。

既有 `bench_world.py` 原样运行，9 个场景、种子 0–59：540/540 物理轨迹哈希相同，
全部逐种子 case、summary 和契约探针相同。它是受控模拟器在线更新协议，不证明
真实环境泛化。既有翻转误差约 0.1 的限制仍存在，本次不宣称修复该场景的预测精度。

`bench_world_inference.py` 使用固定历史、两种模型、每场景 5 次推断；时钟仅用于
性能测量，没有 CI 时间门槛。最终版本 Windows / Python 3.13.13 实测中位数：

| 输入 | 学习器前→后 | 统一模型前→后 |
|---|---:|---:|
| 128 实体，512 帧，全观测 | 66.050→51.483 ms | 79.600→67.558 ms |
| 64 实体，2,000 帧，每帧一个实体 | 327.145→0.793 ms | 325.415→0.912 ms |
| 64 实体，8,000 帧，每帧一个实体 | 1,365.010→0.793 ms | 1,359.379→0.914 ms |
| 64 实体，2,000 帧，随机四实体 | 80.878→1.823 ms | 83.873→2.165 ms |

倍率只代表这些合成输入，不能推广成整个世界模型的加速。14 个模型/输入组合中，
13 个输出摘要哈希相同；随机四实体的统一模型候选关系由 26 条变为 0，原因是没有
达到三段方向支持，而非预测精度标签或新的真值评测。

十项定点变异由 `tools/testing/check_world_inference_mutations.py` 复跑。每项仅运行
指定一个断言，预期恰为 1 failed、0 errors、0 skipped；正常脚本汇总 `10/10`，退出 0。
覆盖 dt、共同观测掩码、窗口前上下文、反向几何、静止计数、三段门槛、混合证据与同位置运动。

## 复跑

```bash
python -X utf8 -B -m pytest tests/test_world_inference_window.py tests/test_world_observation_contracts.py tests/test_pursuit_prediction.py -q -p no:cacheprovider
python -X utf8 -B tools/testing/check_world_inference_mutations.py
python -X utf8 -B tools/testing/check_world_statistics_equivalence.py --baseline PATH_TO_CB3A6EA
python -X utf8 -B tools/verify_manifest.py --strict --quiet
python -X utf8 -B bench/bench_world.py --seeds 60 --output artifacts/world_after.json
python -X utf8 -B bench/bench_world_inference.py --output artifacts/inference_after.json
python -X utf8 -B bench/bench_world_inference.py --compare artifacts/inference_before.json artifacts/inference_after.json
```

改前数据需在固定基线 checkout 中运行同一份 benchmark；新性能脚本复制到该 checkout
的 `bench/` 即可。预测比较使用 `bench_world.py --compare BEFORE AFTER`。
原始逐次计时、完整结果和随机差分日志保留在未跟踪的 `artifacts/world_iteration/`。

## 实际边界

历史与审计记录仍完整保留，不改变公开 history 的列表形态。本次消除了无共同观测的
回溯；存在共同观测时，窗口前上下文和单实体缺帧上下文仍可能读取旧历史，因此不声称
所有推断都与历史长度无关。局部视图空间为 `O(EW+P)`，P 是有共同观测的有向实体对数，
密集场景最坏为 `O(E²)`；视图在该轮结束后释放。没有把该成本隐藏成永久缓存。

仅共享运动与关系统计，没有合并两种世界模型的公开接口、假设生命周期或观测事实与
预测状态，也没有修改脑端存储、因果图或视觉几何。清单只更新被修改导出源码的
指纹，保留此前来源并登记本次基线，不作为重新导出。
