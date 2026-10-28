# 显式历史恢复的有界轨迹工作集

入口：`UnifiedWorldModel.restore_trajectory_from_history()`。
按原顺序遍历全部 history，仍对每帧 tick 执行 int()、对每个位置分量执行 float()。
每实体以 maxlen=256 的 deque 暂存尾部，成功后转为原有 list[(tick, tuple)] 再发布。

- 每实体的 256 点上限、插入顺序、返回保留点数不变。
- 完整 history 不裁剪；稀疏实体按其自己的观测点计数，不按总帧截断。
- 未完成转换时不替换活动轨迹、误差队列；成功后仍清空两条误差队列。
- 原始 NaN/Infinity 的归一化行为保留；本件不新增数值过滤或验证资格。
- 新增峰值工作集是每实体最多 256 个已归一化点，加一个待入队点；原始 history 仍按原体积保留。

范围为显式恢复方法及其既有先验恢复消费路径。懒加载 _wm_traj_buf、周期检测、预测公式与数据库通路保持原实现。
Python 分配峰值与不带追踪的耗时分开测；该峰值不代表进程 RSS、整个项目内存或在线脑的总资源消耗。
复跑和完整读数见 [验收记录](testing/world-restore-bounded-tail-20261028.md)。
