# 批量深度样本反投影 v0.1

`unproject_depth_samples(samples, camera, screen_size)` 把调用方已经明确提供的屏幕像素和相机前向深度，批量委托现有 `Camera3D.unproject()` 转成世界坐标。它不估计深度、不读深度图文件、不创建点云、不写记忆。

## 输入契约

- `camera` 必须是 `Camera3D`，位姿及 fov 为有限实数，视场角在 `(0,180)`；显式 basis 必须是有限、单位、正交且右手的 3×3 基（容差 1e-9）。`screen_size` 是两个正整数 `[width,height]`，且维度可表示为有限浮点数。
- 每条样本为 `[pixel_x,pixel_y,forward_depth]`；像素坐标是屏幕左上角原点，范围是半开矩形 `0 <= x < width`、`0 <= y < height`；depth 是相机坐标前向深度，必须大于 0.1，与 `Camera3D.project()` 的可见性阈值一致。
- 样本最多 4096 条；先完整校验容器、元素、有限性、范围和预算，再调用任何反投影。拒绝非有限值、bool 冒充数值、错误结构、越界像素和非正/过近深度。
- 输出保持输入顺序，只含原样像素、depth 和 `[x,y,z]` world 坐标；不返回相机对象、文件路径或中间数据。

## 数学与边界

函数只调用 `Camera3D.unproject()`，不复制 yaw/pitch/roll 或 explicit basis 数学。分支包含 [相机前置修复 PR #1465](https://github.com/FuRongJun-1999/lingshu/pull/1465) 的原始实现，使显式 basis 与非零 roll 同时存在时也保持 `project()`、`depth_of()` 的回代契约。前向深度与欧氏距离不同；输出点不会自动升格为实体、证据或记忆。

## 验证

合成相机覆盖中心/角落、平移、yaw/pitch/roll、explicit basis、投影回代、错误输入和预校验不调用 unproject；结果可 JSON 序列化且输入不变。
