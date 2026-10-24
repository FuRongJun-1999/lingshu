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

层标签由委托给父类的 `l2_features` / `_h1` 调用上下文确定：`_h1` 内为 L1，
堆叠前向的其余激活为 L1.5。进入时保存上下文，退出时用 `try/finally` 恢复，
包括父类或计数器抛异常的情况；该入口以外的 `l1` / `_h1` / `_lrelu` 不计入。
不按激活调用次数或张量形状猜层，因此相同核数（默认 `K=K2=6`）也可区分。
当前融合前向每层调用一次 `_lrelu`；`calls` 报告真实调用数，`elements` 报告全部核的元素数。

## 本次读数

2026-10-11 重新测量：整合 main `9a8fdf267e4be6a53937616e1d87856d385b52c4` 后的量具修复，
Windows / Python 3.13.13 / numpy 2.5.4 / Pillow 12.3.0，
8 个 48×48 合成场景，seed=7，16 列晶格，晶格形状 `[8,18,16,3]`：

| 配置 / 范围 | 零激活 | 总激活 | 稀疏率 |
|---|---:|---:|---:|
| stacked / L1 | 9 | 13824 | 0.0006510417 |
| stacked / L1.5 | 0 | 13824 | 0.0 |
| stacked / 合计 | 9 | 27648 | 0.0003255208 |
| unstacked / L1（合计） | 9 | 13824 | 0.0006510417 |

输入 SHA-256：`6ad6fa6520b85b09f7980f8fa14cbfe46190ee2a52cf2435146a6a8fd50474e8`。
参数 SHA-256：stacked `39e4c94d31492c8a8802fcd70412be4729dcae277579582f3e24b7d30da3eb49`；
unstacked `57ee16e10f7de8a0b98aec3fc5fbbf20de24ab109e0735a7c73bad754383c9c9`。

本表替换旧基线 `e18562f` 的记录；main 已融合卷积并修正晶格几何，
旧晶格 `[8,10,16,3]` 的分母及精确零计数不能沿用。

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
保留评审指出的前向等价与重复前向回归，扩展相同/不同核数、逐层精确零及元素计数、
异常及嵌套调用的上下文恢复、测量范围外调用，以及 stacked/unstacked 仓外 CLI JSON 检查。

本次相关检查：

```bash
python -X utf8 -m pytest tests/test_hex_activity_measurement.py tests/test_hex_conv_fusion.py tests/test_hex_conv_fusion_matrix.py -q
```

结果为 `17 passed, 14 subtests passed`。内存中换回按调用次数分层的旧量具，
前向等价计数、重复前向和手工计数守卫均失败；移除 `_h1` 上下文恢复，
捕获 L1 异常后继续的守卫失败。变异核验未修改生产文件。
