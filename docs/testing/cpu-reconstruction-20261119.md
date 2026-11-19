# CPU 重建接入验证 · 2026-11-19

**本页是第一轮历史记录。最新修复、20场独立真值、干净wheel与实拍验收见
[第二轮验收](cpu-reconstruction-round2-20261119.md)，本页旧失败/未测项不冒充最终状态。**

上游基线：`d78185c3796e5782df05b6304a4b20fe8a1cb048`。
本次检查范围是可选 CPU 几何生产、静态附件、相机投影与既有三角化接口。
三轮整线和消费端读数见 [integration.json](cpu-reconstruction-20261119/integration.json)。
安装与接口约定见 [CPU 重建说明](../cpu_reconstruction.md)。

## 架构与接入范围

生产链为：有序重叠照片 → 完整 SIFT/匹配 → GLOMAP → 去畸变相机与稀疏轨迹
→ DA3 Base q8_0 深度/置信度 → 全局尺度和平移拟合 → CPU 哈希体素 TSDF
→ 网格、点云、来源记录和校验报告。
所有规模均调用 `global_mapper`；COLMAP 工具仍承担特征、匹配和去畸变。

`WorldModel.load_reconstruction()` 先验证整包元数据和来源，再替换静态附件。
原始相机 K、`[R|t]` 和原图映射保留；稀疏点和相机平移按同一比例转换。
现有单焦距 `Camera3D` 通过显式虚拟像素变换连接完整 K，相机原生像素不被替换。
`triangulate_track()` 使用该变换调用原有 `multiview._triangulate`。

附件不自动生成语义实体、不推进时间、不写预测观测历史，也不自动加入语义场景渲染。
相对坐标不冒充米制或已配准的仿真地图；多个独立分量不被强行拼成同一坐标系。
PLY、SQLite、图像和 NPZ 在读取层只检查存在性；原生内容由生产端生成。
稀疏轨迹和两类来源 JSONL 按行读取并检查，坏包不会替换旧附件。

## 环境与运行配置

- Windows，Python 3.12.8，Intel Core i5-8250U，4 物理核 / 8 逻辑线程。
- NumPy 2.5.5、Pillow 12.3.0、Open3D 0.20.0、pycolmap 4.2.1、
  SciPy 1.16.1、psutil 7.1.3；没有 GPU/torch 推理路径。
- 原生 CPU COLMAP 4.2.1，DA3 Base q8_0，`depth-anything.cpp` 共享库。
  源码及资源版本见 [资源锁](../../tools/reconstruct/resources.lock.json)。
- 三轮均使用完整 SIFT、输入长边上限 640 px、DA3 `direct` 卷积。
- 两轮全稠密运行使用 SfM 2 线程 / 深度 4 线程；第三轮使用 SfM 4 线程 /
  深度 4 线程，显式批量匹配和自动稠密选帧。
- 以下耗时取生产报告的墙钟时间，不含解释器启动和后续消费端检查；
  内存为采样到的进程树峰值 RSS。未测目标 20 线程部署环境。

## 三轮整线实测

| 运行 | 输入 / 主模型 / 稠密帧 | SfM / 深度线程 | 总耗时 | 峰值 RSS | 网格三角面 |
|---|---:|---:|---:|---:|---:|
| `pr-integration-synth7` | 7 / 7 / 7 | 2 / 4 | 31.511 s | 484.8 MiB | 38,953 |
| `pr-integration-temple12` | 12 / 7 / 7 | 2 / 4 | 32.461 s | 506.1 MiB | 5,255 |
| `pr-integration-metric-batched7` | 7 / 7 / 4 | 4 / 4 | 17.686 s | 484.7 MiB | 24,012 |

| 运行 | SfM | DA3 深度 | TSDF/导出 | 对齐 + 校验 | 稀疏重投影 RMSE |
|---|---:|---:|---:|---:|---:|
| synth7 | 4.925 s | 22.887 s | 3.061 s | 0.233 s | 0.3514 px |
| temple12 | 7.579 s | 23.429 s | 0.798 s | 0.181 s | 0.3741 px |
| metric-batched7 | 3.913 s | 11.388 s | 1.881 s | 0.142 s | 0.3553 px |

前两轮输出相对 SfM 单位。第三轮的 `0.01 meters_per_sfm_unit` 是人为输入的
格式/尺度传递测试因子，不是测量基线得到的物理标定；报告虽以 `meters` 标注输出，
本轮不据此宣称真实米制精度。第三轮同时改变线程、匹配调度、稠密覆盖和单位，
不能把耗时差全部归因于某一项，也不与 7 帧全稠密混报同质量提速。

TempleRing 12 张实拍图被解成 7 张和 5 张两个独立分量，仅主模型的 7 张参与稠密。
其余 5 张在 `unregistered_images` 中表示“未进入所选主模型”，不表示没有任何位姿。

## 消费端与质量读数

检查器重新读取全部稀疏轨迹，通过世界模型附件的原生相机投影计算重投影误差，
并与生产报告一致核对。三轮最大归一化投影往返误差为 `4.81e-16`，
最大虚拟相机像素差为 `4.02e-13 px`。
每轮抽取 32 条现有轨迹，经原有多视图三角化得到 32 个有限结果；
与 BA 稀疏点的差值只是两种估计的一致性读数，不是对真值的位置误差。
三轮均未创建语义实体，原有语义场景实例保持不变。

| 运行 | 稀疏点 / 观测 | 跨视角深度差中位数 / p95 | 有效对应数 |
|---|---:|---:|---:|
| synth7 | 1,076 / 4,670 | 9.23% / 24.03% | 7,108 |
| temple12 | 829 / 3,386 | 0.85% / 1.80% | 12,260 |
| metric-batched7 | 1,076 / 4,672 | 22.20% / 39.49% | 2,699 |

跨视角读数取相邻保留深度视角间的投影一致性；自动选帧改变视角间距和有效样本。
它既不是独立深度 GT abs-rel，也没有检验缺失的表面覆盖。
三轮提取网格顶点均有邻近观测体素来源映射；这表示来源可追溯，
不表示每个顶点/三角面均有精确、独立的真实表面观测。

## 证据等级与未完成边界

- synth7 来源于 DA3 深度重投影生成的合成视图：属于 E1 整线接入证据，
  不作为独立重建精度基准；第三轮也是该合成输入的配置检查。
- TempleRing 完成真实输入、部分主模型覆盖的接入与一致性检查，本轮也按 E1
  工程接入证据记录；W3 所要求的 20 场景独立真值验收仍未完成。
- 本轮没有独立 GT 位姿/深度、轮廓 IoU 或完整表面真值，
  不宣称位姿 ≤0.5°、平移 ≤5 cm、深度误差 ≤5% 已通过。
- 未测在线 SLAM、动态物体、语义检测、自动分量配准及 20 线程性能。
- wheel 隔离检查已通过：标准库导入、`--help`、缺少重载依赖时的明确提示；
  wheel 包含 12 个生产模块，不携带 GGUF 或原生二进制。

## 复跑入口

安装可选依赖并按 CPU 重建说明准备原生资源。以下使用现成的同一组输入照片，
每次指定新输出目录；原始照片、模型和本机运行产物不纳入 PR。

```sh
python -m lingshu.world.reconstruct /path/to/photos --output runs/full \
  --resource-root /path/to/resources --threads 2 --depth-threads 4 \
  --max-image-size 640 --depth-views all --depth-conv direct
python -m lingshu.world.reconstruct /path/to/photos --output runs/scaled-auto \
  --resource-root /path/to/resources --threads 4 --depth-threads 4 \
  --max-image-size 640 --sfm-matching batched --depth-views auto \
  --depth-conv direct --meters-per-sfm-unit 0.01
python tools/reconstruct/check_integration.py --run runs/full \
  --run runs/scaled-auto --output integration-local.json
python -m pytest tests/test_reconstruction_camera.py tests/test_reconstruction_import.py \
  tests/test_reconstruct_geometry.py tests/test_prepare_depth_source.py \
  tests/test_reconstruct_fusion_failures.py tests/test_reconstruct_pipeline.py -q
```

本机三份生产报告在 `test-output/pr-integration-{synth7,temple12,metric-batched7}/report.json`；
本文件表格取自对应产物和已归档的 `integration.json`。

## 全仓 gate 与修复后复测

机器回执见 [validation.json](cpu-reconstruction-20261119/validation.json)。

| 检查 | 实际结果 |
|---|---|
| 冻结批次：全仓 pytest，按 CI 排除脚本件，4 worker | 5,548 passed / 20 failed / 39 skipped，另 119 子测试通过，1,032.91 s |
| CI 列出的脚本式测试，逐件独立进程 | 102/102 退出码 0 |
| 同一环境仅把 20 个失败项放回原始官方基线 | 16 passed / 4 failed |
| 修复后定点范围 | 246 项通过，含全部 100 项新增重建测试，另 21 子测试通过 |
| 逐模块导入 | 103/103，已安装可选重建依赖 |
| 最终 wheel | 构建成功；标准库隔离导入/帮助/缺依赖提示通过；12 个生产模块齐全，无模型和原生二进制 |

16 项接入引入失败的共用根因及修复：几何缩放被时间衰减规则误报，改用显式
图像缩放矩阵；管线缺少直接守卫，补资源预检和尺寸/截断边界测试；模块递归计数、
活源码指纹、第三方/标准库分类随新代码登记；测试子进程明确 UTF-8。
没有放宽 lint、增加违规白名单、删除用例或改验收阈值。

独立审阅另复现“已推理帧被融合静默丢弃，成功包却不能加载”的契约错误：
修复前专用回归 4 failed / 1 passed；修复后 5/5 通过。
缺图、坏图、过滤无点、无 TSDF 块均报出失败帧，未推理视角仍允许延期。

定点验证按 178 + 15 + 52 + 1 项统计，避免重跑整个仓库。第一组另遇到现有
`_pathguard` 与 `tools` 命名空间的收集顺序冲突，导致覆盖扫描文件未收集；
在新进程单独运行该文件后 15/15 通过。其余 52 项是时间规则文件的完整检查；
并把首轮因并发跳过的串行耗时守卫单独补跑，1/1 通过。
未再次运行整仓，所以不把各批数字合写成一次“最终全仓全绿”。

剩余 4 项均在未改动的官方基线上同样失败：既有查看器测试访问固定版本
Cytoscape CDN 时发生 `URLError / WinError 10054`，未改相关产品代码。
首轮其余 38 项跳过按依赖缺失/平台边界归档，不算已验证；Linux 原生整线、独立真值与
20 线程部署也未在本轮补测。此次仅准备 PR 材料，尚未推送或创建线上 PR。
