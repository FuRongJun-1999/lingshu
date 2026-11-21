# HexGrid odd-r 六邻居修复与验证（2026-11-21）

## 范围与触发

`HexGrid.neighbors` 在 `mode="odd_r_offset"` 时仍直接加 axial 偏移。
以 `size=1`、`origin=(0,0)` 为例，偶数行 `(3,2)` 错取 `(4,1)`、漏取
`(2,1)`；奇数行 `(3,3)` 错取 `(2,4)`、漏取 `(4,4)`。错取的格心距离
为 `3*size`，真实六邻距离均为 `sqrt(3)*size`。公开 `image_to_grid` 返回的
正是该 offset 模式，故不需要调用未支持的坐标模式即可触发。

修复只改 `lingshu.nn.hex_cnn.HexGrid.neighbors`：将输入列号按
`axial_q = offset_col - floor(r/2)` 转为轴向，在原有方向顺序上移动，再以
目标行的 `floor((r+dr)/2)` 转回列号。Python `//` 对负行向下取整。
默认及显式 axial 的列表/顺序保持一致，六邻居仍完整返回，边界由独立的
`in_bounds` 判断。没有修改采样、卷积 tap、最近像素定位或 axial 原点契约。

该问题与 #6 的采样/返回网格往返、#18 的卷积 tap 顺序、#1473 的任意像素
最近格心定位分别落在不同方法。这里的结论限于公开邻居 API 的几何错误；
没有量得训练、识别或实际调用方的质量损失，也不宣称其生产性能受到影响。

## 固定身份

- 初始只读审计：`04e0ecb6a3669c90efbb4a1583816a8f7ca26bc7`。
- 更新后原样基线：`6b73082406d40a630a44cd68559c1f66b3652390`。
- 干净工作树测量提交：`df0838c2181abf5fb1dc8e1e4b0dfcb66c19a232`。
- 基线 `hex_cnn.py` LF SHA256：
  `e1b21e3b7e74cbdd34627033a217520a3e272a30bcb3799f93e8d0e6d3c81081`。
- 修复 `hex_cnn.py` LF SHA256：
  `5671ae505f6dd026fc4fac7cb50ca02ca76ca972a51292f2dc50b5c7ae37efee`。
- 导出清单只更新 `hex_cnn.py` 的 SHA 前 16 位；该条目没有字节数字段。

后续文档提交不改变已测生产源码、守卫、变异规格或基准脚本。

## 行为守卫、原样基线与变异

`tests/test_hexgrid_odd_r_neighbors.py` 共 30 项，涵盖：

- E、NE、NW、W、SW、SE 有序输出；奇偶行、负列、负行及双负坐标。
- 4 组正尺寸/平移原点、正负格点；axial 只测文档承诺的零原点。
- 1×1、单行、单列及矩形的边界/界外查询；仍返回六个不同格点。
- 邻接互反性，以及 `image_to_grid` 的方形、横幅、竖幅三种公开路径。
- 默认及显式 axial 的原有有序列表保持逐项相同。

守卫的 oracle 在 7×7 文档笛卡尔格心点云中找最近六点，再按物理方向排序，
不导入 `HEX_DIRS`，不调用 `hex_conv` 或 `pixel_to_cell`。另一个只读点云探针
交叉覆盖 4 种尺寸 × 3 种 offset 原点 × 81 个正负格点。

| 验证 | 原样基线 | 修复版 |
|---|---:|---:|
| 30 项守卫 | 11 passed / 19 failed | 30 passed |
| 独立 odd-r 点云格点 | 972/972 错误 | 0/972 错误 |
| 独立 axial 点云对照 | 0/324 错误 | 0/324 错误 |
| 公开图像入口内部格点 | 400/400 错误 | 0/400 错误 |

三种图像入口逐组为 42、98、260 个严格内部格点，没有把边界裁剪纳入争议。
完整守卫的原样基线失败清单固定在 fixture 中。以下 6 个命名变异分别由指定
测试的行为断言杀死，全部为 1 failed，均为 0 error / 0 skipped：

| 变异名 | 故意引入的错误 |
|---|---|
| `ignore_offset_mode` | 忽略模式，回到 axial 偏移 |
| `reverse_offset_correction` | 把行偏移修正符号写反 |
| `truncate_negative_rows` | 负行用向零截断代替 floor |
| `reverse_direction_order` | 最近六点正确但方向顺序反转 |
| `clip_offset_neighbors_to_shape` | 静默裁去界外邻居 |
| `apply_offset_correction_to_axial` | 错把转换应用到 axial |

使用仓内原有 runner，复制生产代码和原样守卫后测量，不在真实工作树上打变异：

```sh
python -B -X utf8 tools/testing/replay_save_guards.py \
  --spec tests/fixtures/hexgrid_odd_r_neighbors_mutations.json \
  --baseline 6b73082406d40a630a44cd68559c1f66b3652390 --mutate \
  --output artifacts/hexgrid-neighbors-replay
```

原始输出为 `summary.json` 及各 case 的 `.log` / `.xml`，fixture 把完整基线
失败名、每个变异的目标方法、唯一文本锚和预期断言一并固定。

另一次只读独立审阅用显式奇偶偏移表核对 20,050 个整数用例，包含
`q/r = ±2**60` 附近，全部通过；同时逐项核对 8 份 replay XML、相邻回归
XML 及基准的样本数、源码/脚本哈希。该检查不重用生产转换公式，没有改动
生产树或重跑长基准。原始结果为 `independent-review/summary.json`，探针为
`independent-review/parity_probe.py`。

## 相邻回归

下列 pytest 合计 **47 passed**，JUnit 为 `neighbors-adjacent.xml`：

```sh
python -B -X utf8 -m pytest -q \
  tests/test_hexgrid_odd_r_neighbors.py \
  tests/test_hexgrid_odd_r_nearest_cell.py \
  tests/test_issue6_grid_roundtrip_guard.py \
  tests/test_issue455_subprocess_encoding.py \
  tests/test_gate_script_leg_reachability_guard.py
```

另按门禁要求单跑两个原生脚本：

- `python -B -X utf8 tests/test_hex_grid_extent.py`：23/23 通过。
- `python -B -X utf8 tests/test_hex_cnn.py`：13 passed / 0 failed；脚本原有
  4–6 素材段因 `data/img/0.png` 未随导出而 SKIP，没有冒充完成该图片验证。

导入方向、静默异常、docstring 编译、intake 登记、门禁覆盖及覆盖图 lint 的
现有测试另得 **63 passed**（`neighbors-lint.xml`），没有修改任何 lint 基线：

```sh
python -B -X utf8 -m pytest -q \
  tests/test_import_direction_lint.py \
  tests/test_silent_except_lint.py \
  tests/test_docstring_surrogate_compile_guard.py \
  tests/test_issue10_intake_registration_guard.py \
  tests/test_gate_coverage.py \
  tests/test_coverage_map_lint.py
```

本件没有重跑整仓 gate，不能用这些相邻读数代替整仓或 Linux CI 结论。

## 公平 timing / memory / scaling

```sh
python -B -X utf8 bench/bench_hexgrid_neighbors.py \
  --baseline 6b73082406d40a630a44cd68559c1f66b3652390 \
  --output artifacts/hexgrid-neighbors-benchmark.json
```

默认 12 组：2 种模式 × 正/负坐标 × 1,000/10,000/50,000 个查询。每组每版本
3 次预热、9 个保留 timing 样本、3 次单独 memory probe。输入提前生成且两个
版本逐项相同，两种行奇偶均有覆盖。计时包括完整返回列表/元组的分配，并将
整批 `6*N` 个格点保留到计时结束；导入、输入分配、断言、哈希、GC 前置回收
及结果释放均在计时外。每对样本交替 AB/BA 顺序，GC 保持启用。

每批验证输入/输出基数和六点唯一性。正确性用独立笛卡尔向量（长度
`sqrt(3)*size`、每隔 60°、固定方向顺序）逐项检查；odd-r 基线每组正确查询
均为 0，修复版均为 N。axial 两版不仅正确数均为 N，输出 SHA256 也逐组相同。

以下单位为 **微秒/查询**，显示 9 个原始 timing 样本的中位数；比值为修复/基线。

| 模式 | 坐标 | N | 基线 | 修复 | 比值 |
|---|---|---:|---:|---:|---:|
| axial | 正 | 1,000 | 0.598 | 0.616 | 1.030 |
| axial | 正 | 10,000 | 0.605 | 0.634 | 1.048 |
| axial | 正 | 50,000 | 1.306 | 1.263 | 0.968 |
| axial | 负 | 1,000 | 0.667 | 0.699 | 1.048 |
| axial | 负 | 10,000 | 0.790 | 0.827 | 1.048 |
| axial | 负 | 50,000 | 1.759 | 1.843 | 1.048 |
| odd-r | 正 | 1,000 | 0.956 | 1.466 | 1.533 |
| odd-r | 正 | 10,000 | 0.995 | 1.500 | 1.508 |
| odd-r | 正 | 50,000 | 1.598 | 2.080 | 1.302 |
| odd-r | 负 | 1,000 | 1.045 | 1.638 | 1.567 |
| odd-r | 负 | 10,000 | 1.221 | 1.999 | 1.637 |
| odd-r | 负 | 50,000 | 1.693 | 2.574 | 1.520 |

修正有实际的算术成本，odd-r 中位耗时为旧错误实现的 **1.302–1.637 倍**。
axial 对照为 0.968–1.048 倍；共享 Windows 主机上的这些样本仅作描述性成本
记录，不能据此宣称加速或统计显著性。整批保留结果带来的分配/GC、小整数
缓存以及主机负载都会影响每项耗时，不能把它理解成纯算术微指令成本。

memory 在独立的 `tracemalloc` 区间测量，报告 Python traced retained/peak bytes，
不包含预建输入，也不是进程 RSS 或 NumPy native memory。修复版正坐标峰值
随 N 为 559,462 / 5,619,270 / 28,162,790 bytes，负坐标为
655,846 / 7,973,670 / 41,889,094 bytes；两版对应组峰值差的绝对值最大 64 bytes。
负坐标的小整数缓存覆盖率随行范围变化，故不能假定各规模每查询字节数完全恒定。

算法对 N 个查询为 O(N)，每个查询六点，保留结果空间为 O(N)。原始 JSON 保存
全部 timing 样本、每次 retained/peak bytes、各组正确计数/输出哈希、两版源码
SHA256、基准脚本哈希、精确测量提交及干净工作树状态。实测环境为 Windows、
Python 3.12.10、NumPy 2.5.5；此轮基准用时约 46.43 秒。
