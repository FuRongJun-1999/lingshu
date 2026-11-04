# 相机容差内焦距损失显式报告的调用约束

对应 README 的真实相机接入及 camera_bridge 的显式报告损失契约；容差决定是否接受近似，不能证明数学上无损。

任何非零 fx/fy 差别都写入已有 lossy 报告并将 representable_exactly 置假。strict 的拒绝门槛仍是原有 tol_frac；容差内继续允许近似，投影、fov 公式与相机表达能力不变。

实际调用 Camera3D.project，验证 0.5 px 残差与逆转换 fy；使用合成内参和世界点，未接入物理相机、DSH 或生产数据。不是远程安全边界或 P0。
