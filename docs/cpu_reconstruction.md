# CPU 多视角重建与世界模型接入

本入口从按拍摄顺序命名、有重叠的静态照片生成相机、稀疏轨迹、相对深度、
TSDF 网格和来源记录，再把产物加载为 `scene_model.WorldModel` 的静态几何附件。
相机和稀疏几何可直接供原有多视图三角化使用。照片不被当作多个时刻，
几何点不被自动当作有类别的语义实体。

## 生产与依赖

`pip install -e ".[world-reconstruct]"` 安装可选 Python 重建依赖；
`core`、普通 `world` 和 `full` 不因此引入 Open3D、COLMAP 或 torch。
导入读取层、导入重建包和 `--help` 只需要标准库。
原生执行文件、GGUF 和图像不进入 Python wheel，也不在运行时自动下载。

```sh
python -m lingshu.world.reconstruct ./photos --output ./runs/scene \
  --resource-root /path/to/resources --depth-views all
```

`--resource-root` 也可通过 `LINGSHU_RECONSTRUCTION_HOME` 指定，缺省为
`~/.cache/lingshu/reconstruction`。资源布局沿用：

```text
.tools/colmap/bin/colmap.exe
.tools/depth-anything.cpp/build/libdepthanything.dll
.tools/llvm-mingw/bin/                  # Windows DLL 依赖目录
models/depth-anything-base-q8_0.gguf
```

其他系统可分别通过 `--colmap`、`--depth-library`、`--model` 和 `--runtime-bin`
指定本机资源；当前原生整线实测是 Windows，不承诺未经测试的平台。

生产顺序：

1. 规范图像尺寸、记录原图到标准帧名的清单；保留输入 alpha 声明的有效像素域。
2. CPU 完整 SIFT、有序候选匹配、GLOMAP 全局位姿与三角化/BA。
   所有规模均用 `global_mapper`；没有增量建图后端或自动回退。
3. 去畸变，导出原始 SfM 坐标中的相机与稀疏轨迹。
4. DA3 Base q8_0 常驻 CPU 推理，估计深度和置信度。
5. 用稀疏相机深度拟合共享尺度和平移，过滤低置信深度，再做 CPU 哈希体素 TSDF。
6. 导出网格、点云、逐体素来源、网格顶点的邻近观测体素来源和校验报告。

COLMAP 4.2.1 的 CPU 工具仍用于 SIFT、匹配、去畸变及其集成的 GLOMAP。
若解出多个独立分量，只融合注册相机最多的主模型，剩余图像列入
`unregistered_images`，其含义是“未进入所选主模型”。原生模型全部保留，
`01_sfm/sparse/components.json` 列出每个分量的图像、点数和独立坐标系；
`report.json.coverage` 分别记录主坐标系覆盖与全部独立分量覆盖，不混为同一地图。

`--depth-views all` 是默认值，为每个主模型注册视角计算稠密深度；`auto` 根据轨迹和画面
分块覆盖减少深度视角，并在尺度支持不足时补算。减少稠密帧会改变表面覆盖，
不作为与全帧质量相等的默认承诺。`--sfm-features native` 同样是有损选项；
默认保留完整 SIFT。有序匹配要求文件名顺序反映拍摄邻近关系。
首次提取后，每图少于 512 个特征时，最多以 `peak_threshold=0.002` 重提一次该图，
恢复低对比输入的观测支持；正常图不重提。该工程下限不是重建精度保证。
恢复只作用于本次新建且尚无匹配的数据库：临时库复制元数据而不复制正常图的描述子，
验证相机、图像 ID 及原特征身份后，事务回填弱图特征。失败保留原数据并报错；
已有数据库不执行恢复。初次/恢复特征数、触发图像和耗时见 `logs/sift_support.json`
与 `stages.sfm.feature_support`。
乱序小图库可显式选择 `--sfm-matching exhaustive`，逐对匹配成本按图像数平方增长；
`auto` 不会把大图库静默切成穷举匹配。
默认 `--confidence-percentile 0` 保留有效域，不强行删除固定比例深度；
提高该分位数是显式有损筛选，需要同时检查精度、表面覆盖和轮廓。
已推理并生成对齐深度的视角必须成功融合；缺图、坏图、无有效点或无 TSDF 块时，
整次运行报出失败帧名，不生成成功报告。只有未推理、无深度文件的延期视角会跳过。

uint8/uint16 RGBA 输入的 `alpha>0` 是有效域，包括部分透明像素；完全透明输入会拒绝。
该域也约束原生 SIFT，声明无效区域不再提供特征。原生 mask 使用包含图像扩展名的
`00_input/feature_masks/<image.name>.png`；混合 RGB/RGBA 输入中，未声明 mask 的图像
补全白有效域，以满足原生工具逐图 mask 契约。纯 RGB 输入不额外创建 mask。
有效域按原生相机模型去畸变后写入原始与对齐 NPZ 的 `input_valid_mask`，
域外深度不参与尺度拟合或融合。RGB/JPEG 不按黑色猜背景，也不读取评测 GT mask。
提供 alpha 的重建证据必须标注输入含有效域信息，不能冒充未分割 RGB 输入的轮廓能力。

TSDF 网格保留每个 Marching Cubes 角点至少一次真实积分支持的表面，排除零权重未知空间。
这能保留仅被一个视角看见的有效边缘，也会保留更多单视图深度噪声；来源可追溯不等于
几何精度保证。`stages.fusion.mesh_support` 记录该支持策略，精度、覆盖与轮廓应并列检查。

## 原生深度库的可复现准备

实测外部资产摘要见 [`resources.lock.json`](../tools/reconstruct/resources.lock.json)。
锁定的 depth-anything.cpp 提交是
`14f7461d1f704761a038ac9f50dbde8fdb7275e2`，ggml 是
`eced84c86f8b012c752c016f7fe789adea168e1e`。
代码、模型各自遵循其上游许可；本 PR 不分发它们。

在已取得的 depth-anything.cpp 源码树及 `third_party/ggml` 内容就位后执行：

```sh
python tools/reconstruct/prepare_depth_source.py --source /path/to/depth-anything.cpp
cmake -S /path/to/depth-anything.cpp -B /path/to/depth-anything.cpp/build \
  -DDA_SHARED=ON -DDA_BUILD_CLI=ON -DDA_GGML_CUDA=OFF \
  -DDA_GGML_METAL=OFF -DDA_GGML_VULKAN=OFF \
  -DDA_HAS_AVX512F=OFF -DGGML_NATIVE=OFF -DGGML_AVX512=OFF
cmake --build /path/to/depth-anything.cpp/build --config Release --parallel 4
```

准备工具核对唯一锚点后，给 C API 增加复用现有 fused-depth 路径的分支；
重复执行不修改文件，无法识别的源码会在写入前拒绝。
Windows 实测工具链是 llvm-mingw，显式关闭 `GGML_NATIVE`、AVX-512 和 OpenMP，
保留 AVX2/FMA；应按部署 CPU 选择兼容构建参数，不把一台机器的 native 编译物
直接当作所有 CPU 的通用二进制。上游 CMake 的 AVX-512 检测只证明编译器能生成指令，
不是 CPU/操作系统运行能力检测；上面的保守构建显式关闭该指令集。
Winograd 的标量回退和 AVX-512 路径是不同部署条件，取舍需在对应硬件实测。

## 世界模型读取接口

```python
from lingshu.world.scene_model import WorldModel

world = WorldModel().load_reconstruction("runs/scene")
geometry = world.reconstruction
camera = geometry.cameras["000000.jpg"]
track = next(geometry.iter_tracks())
pixel = camera.project(track["xyz"])
estimated_point = geometry.triangulate_track(track["point3D_id"])
mesh_path = geometry.artifacts["mesh"]
artifact_receipt = geometry.validate_artifacts()  # 显式解码原生文件和图像
preview = world.render_reconstruction("000000.jpg")  # 实际 CPU z-buffer 渲染
rgb, camera_z, mask = preview.rgb, preview.depth, preview.mask
```

`Reconstruction.load()` 也可独立调用。完整验证结束后才替换 `world.reconstruction`；
坏包保留旧附件和原场景。附件独立于 `entities`，`build()` 保留它。
`render_reconstruction()` 通过既有 CPU 光栅器输出 RGB、相机 Z 深度、可见性 mask；
支持全 K、近平面/远平面裁剪和遮挡。顶点色平均预览不冒充纹理或光度重建。
如需 Pillow 图片，可调用 `preview.to_image()`，该转换需要现有的 `world` extra。
原有语义场景的 `render()` 和时序模型的 `perceive()` 继续独立：
没有类别、对象身份、拍摄时间或跨地图配准信息时，不写实体记忆或动态预测历史。

读取层核对清单、相机、稀疏轨迹、深度帧覆盖和来源 JSONL，检查 PLY、SQLite、
图像与 NPZ 的存在性。调用 `validate_artifacts()` 才显式解码 PLY 网格/点云、
SQLite 完整性/表结构/来源计数、深度/置信度/有效域 NPZ 和归一化/去畸变图像，
拒绝损坏内容、非法形状、非有限值和无效深度。稀疏点和 provenance 按行读取，
使用期间产物文件应保持不变。包内引用限定在运行目录中，支持搬迁后的原始绝对路径重定位。
原图来源是证据引用，原图可在目录外；搬迁后不要求原始本机路径仍然存在。

## 相机、尺度与像素

- `ReconstructionCamera` 保留去畸变后的完整 K 与世界到相机 `[R|t]`，
  原生像素遵循 x 向右、y 向下；只接收 `PINHOLE`/`SIMPLE_PINHOLE`。
- 现有 `Camera3D` 只支持居中单焦距，因此 `as_camera3d()` 必须与
  `to_virtual_pixel()` 配套使用。该显式变换保留不等焦距、偏心主点和 skew；
  虚拟像素不冒充原图像素。`triangulate_track()` 内部完成该转换，再调用现有三角化器。
- 默认坐标是任意尺度、任意原点与方向的 SfM 局部坐标，不声称已对齐重力或地面。
- `--meters-per-sfm-unit` 是调用方提供的可信尺度因子，不会从普通单目照片自动
  恢复绝对尺度。导出相机/稀疏轨迹保留原 SfM 单位，读取层恰好转换一次；
  稠密产物已经是输出单位，不再次缩放。仅提供尺度仍不代表完成仿真地图的轴向配准。
- 来源像素属于深度图分辨率；它们与原图、去畸变图、虚拟像素是不同的坐标域。
  网格来源是邻近已观测体素映射，不等同于每个面均有精确表面观测。
- 导出的连续相机像素遵循 COLMAP 的 `(i+0.5,j+0.5)` 中心；数组用整数索引。
  Open3D 内部栅格的 K 主点仅在边界转换一次，导出 K 不改写。
  mask 重映射、稀疏取样、回投和消费端光栅化使用同一约定。

## 校验与边界

```sh
python -m pytest tests/test_reconstruction_camera.py tests/test_reconstruction_import.py \
  tests/test_reconstruct_geometry.py tests/test_prepare_depth_source.py \
  tests/test_reconstruct_fusion_failures.py tests/test_reconstruct_pipeline.py -q
python tools/reconstruct/check_integration.py --run runs/scene --output integration.json
```

稀疏重投影、尺度锚点残差和跨视角深度差各有独立口径。
后者检查相邻保留深度视角间的一致性，不是独立深度真值误差，也未替代融合网格的
独立表面/轮廓评测。报告中轮廓 IoU 缺失会明确标记。当前不提供在线 SLAM、
动态物体跟踪、语义检测或跨独立分量的自动配准。

独立相机标定可通过以下入口评价既有运行，不向重建器输入真值：

```sh
python bench/reconstruction_pose.py --run RUN --calibration CAMERA_par.txt --output result.json
```

校准文件采用相机数及逐行`name + 9K + 9R + 3t`，投影为`K[R|t]`。
评测优先原始`01_sfm`，使用单一相机中心Sim3并报告相对旋转；格式未声明长度单位，
结果保留dataset单位，不按米输出。该项不代替稠密深度、网格表面或固定20场W3验收。

最新整线和回归证据见 [第三轮验收](testing/cpu-reconstruction-round3-20261119.md)，
[第二轮](testing/cpu-reconstruction-round2-20261119.md)及第一轮历史记录保留。
本 PR 的范围是可选几何生产与静态接入；原20场、100视角均完成注册与全帧稠密，
R4轮廓20/20通过，W3全部质量门槛同时通过13/20。单观测表面支持提升覆盖但降低部分
表面精确率，完整权衡见报告；不作为已验收的完整重建能力提交。
