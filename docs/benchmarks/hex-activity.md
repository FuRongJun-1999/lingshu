# Hex 活动稀疏度测量（候选工具）

依据本仓[存算一体架构总纲 §7/§9](../theory/存算一体神经网络架构_总纲与交接_20260916.md)
与[外部理论对照 §4](../theory/存算一体架构学习_外部理论对照与路线交接_20260916.md)
列出的「活动稀疏度量化」候选，提供一个可复算的测量入口。是否采纳仍由设计者裁定。

外部口径采用 [NeuroBench 的 ActivationSparsity 定义](https://neurobench.readthedocs.io/en/latest/neurobench.benchmarks.html#neurobench.metrics.workload.ActivationSparsity)：
所有被测激活中**精确为零的个数 / 总个数**。
按层、批次累加整数计数，再除总分母；不平均各层比例，不把近零当零。

## 运行与条件声明

在本仓已有 `.[full]` 环境运行：

```bash
python -X utf8 scripts/measure_hex_activity.py --samples 8 --cells-across 16 --seed 7
python -X utf8 scripts/measure_hex_activity.py --samples 8 --cells-across 16 --seed 7 --unstacked
```

测量当前 `HexHierNet.l2_features` 前向路径中 L1 与可选 L1.5 的
**LeakyReLU 输出**。记录真实调用返回值，原运算交给父类；不重写卷积或激活算式。
输入使用现有合成场景生成器、晶格映射与归一化，模型是指定 seed 的**未训练基线**。
报告含场景数、晶格尺寸、堆叠状态、输入及参数 SHA-256、运行版本、逐层及总计数。
测量后逐位检查参数未改变。

## 本次读数

代码基线 `e18562f`，Windows / Python 3.13.13 / numpy 2.4.4 / Pillow 12.2.0，
8 个 48×48 合成场景，seed=7，16 列晶格，晶格形状 `[8,10,16,3]`：

| 配置 / 范围 | 零激活 | 总激活 | 稀疏率 |
|---|---:|---:|---:|
| stacked / L1 | 922 | 7680 | 0.1200520833 |
| stacked / L1.5 | 0 | 7680 | 0.0 |
| stacked / 合计 | 922 | 15360 | 0.0600260417 |
| unstacked / L1（合计） | 922 | 7680 | 0.1200520833 |

这是该输入、未训练模型及被测范围下的观察值。Python 3.13 属本仓门禁的
advisory 档；本记录未声称覆盖 README 的 Python 3.12 版本矩阵。

## 范围与判据

- 仅覆盖上述激活输出；输入、池化/RMS、颜色均值、L3 线性 logits、L4 位置输出未计入。
  `scope` 随 stacked 条件明确变化；无 L1.5 时不补造该层读数。
- 这是采用同一**指标定义**的局部测量，不是全套 NeuroBench 评测或认证。
- 不测训练信息差死区触发率、连接稀疏率、MAC 数、耗时或功耗。
  激活稀疏率不能代替这些量，不从本次读数推出节能或精度结论。
- NaN/Inf 仍计入总元素并单列污染个数，但该层及合计比例为 `null`、状态 `invalid`；
  空测量比例为 `null`、状态 `not_measured`，避免把无观测写成满分。
- 不引入 NeuroBench/PyTorch 依赖，不修改蜂窝网络、信息差训练、四态判据或既有阈值。

守卫测试：

```bash
python -X utf8 -m pytest tests/test_hex_activity_measurement.py -q
```

测试用手工可算的反例区分元素加权与层平均、精确零与近零；同时检查空/污染输入、
切批不变性、真实调用覆盖、与原模型前向逐位相等、参数未变及 seed 复现。
