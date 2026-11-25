# 身体批量深度样本反投影 v0.1 · 新件登记

来源：本仓新实现，复用现有 `Camera3D.unproject()`。

- 接受显式 `(pixel_x,pixel_y,forward_depth)` 样本和 `Camera3D`，返回保持顺序的世界坐标。
- 先校验相机有限位姿、合法视场角、显式右手正交基、画布、样本数量、有限性、越界像素和前向深度；最多 4096 条，拒绝发生在调用 `unproject()` 前。
- 不估计深度、不读文件、不生成点云、不写文件/数据库/记忆，不引入 NumPy/Pillow。
- 深度是相机前向 z，不是欧氏距离；调用方负责样本来源与标定可信性。

验证覆盖相机中心/边角、平移、yaw/pitch/roll、explicit basis、投影回代、坏输入、预算前拒绝、JSON 安全、输入不变和无 NumPy/Pillow 导入。

## 复跑

```sh
python -m pytest tests/test_depth_samples.py tests/test_world3d_camera_roundtrip.py tests/test_issue1194_camera_bridge_guard.py tests/test_doc_module_counts_guard.py tests/test_issue79_architecture_layers_guard.py tests/test_issue455_subprocess_encoding.py tests/test_issue80_intake_rerun_executable_guard.py tests/test_issue156_component_discovery.py tests/test_gate_python_floor_and_matrix.py tests/test_manifest_integrity.py -q --no-header
# 361 passed
python -X utf8 tools/intake_smoke_min.py --quiet
# import ok=90 fail=0 constructive=ok
python -X utf8 tools/verify_manifest.py --json --strict
# exit_code=0
```

分支包含相机前置修复PR #1465（显式basis与roll共存时的逆变换一致性），合并时应先合并该前置或保留其相同改动。
