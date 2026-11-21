# 动力学受控动作的初值重锚（2026-11-21）

## 问题与修复

`SceneSimulator.step` 的既有优先级是 command > dynamics > autonomous；F08 执行的是实际动作。此前 `_apply_command` 改了实体位置，却保留旧 `physics_state.p0/v0/t_ref`。下一次动力学求值从旧闭式轨迹重播，使已完成的位移被撤销。真实 `SevenLayerLoop` 的 execute → clear → resume 路径同样受影响。

最小例：`dt=0.1`，动力学实体位于 x=64；执行 east、step_size=2 后 x=66，清除行动后旧版恢复到 x=64。修复后从 x=66 与本次执行的速度恢复，下一 tick 为 x=68。

本补丁只在动力学实体执行受控动作时复制并重锚初值：

- `p0` 为边界钳制、两位小数取整后的真实端点，`t_ref=tick_count*effective_dt`。
- east/west/north/south 的物理 `v0=(new_pos-old_pos)/effective_dt`，端点差不再次取整。命令 tick 对外的 `velocity` 仍按既有两位小数、位移/tick 口径。
- 连续旧坐标 x=64.003 执行 east 2 后端点为 66，实际差为 1.997；dt=0.1 时 `v0.x=19.97`，公开 `velocity.x=2.00`。公开读数不充当独立物理真值。
- 小的正方向步长经过既有取整后可能产生微小反向位移，例如 x=64.003、east 0.001 → x=64.00。物理初值使用真实端点差，不能仅因公开速度取整为 0 而当作静止。
- `stay` 沿已有“原地/停止”动作语义给零物理初速；既有坐标取整可微调位置，但不由此生成动量。真正 `new_pos==old_pos` 的方向动作自然得到零初速。
- 第一次动作也建立 state，避免恢复动力学的首 tick 仅登记初值。旧 state 的 beta/radius/mass 与自定义元数据保留，旧 dict 不原地变异。
- 写回实体之前检查新物理速度与参考时刻的有限性；本地拒绝范围仅服务这次重锚，不改其他入口或整个 tick 的事务模型。
- 真实碰撞后的最终端点/速度继续由既有碰撞流程重播种。

动作表仅有五个平移动作；此补丁不定义旋转或角动量。环境时间轴与运行时环境切换保持原实现。`Environment` 文档声明唯一 dt；实体环境覆盖守卫使用同一 dt，本补丁不新增每实体异步时钟承诺。

## 可复现验证

```powershell
python -B -X utf8 -m pytest tests/test_dynamics_command_reanchor.py -q
python -B -X utf8 tools/testing/replay_save_guards.py --spec tests/fixtures/dynamics_command_reanchor_mutations.json --baseline 04e0ecb6a3669c90efbb4a1583816a8f7ca26bc7 --mutate --output artifacts/command-guards
python -B -X utf8 bench/bench_dynamics_command_reanchor.py --baseline 04e0ecb6a3669c90efbb4a1583816a8f7ca26bc7 --mode all --repeats 11 --ticks 400 --oracle-cases 2000 --output artifacts/command-benchmark
```

| 验证 | 结果 |
|---|---|
| 固定版持久守卫 | 61 passed，0 failed/error/skip |
| 同一守卫运行原始版 | 58 failed，3 passed，0 error/skip |
| 命名变异 | 16/16 按指定断言失败，0 error/skip |
| 相邻动作、环境、碰撞、核心接线及广物理集 | 340 passed，0 failed/error/skip |
| 独立 Decimal(60) 数值对照 | 修复版 2,000/2,000 通过；原始版 2,000/2,000 偏离 |

数值样本使用连续初始坐标、5 动作、11 个 dt、7 个步长、0–7 个前置 tick、1–7 个恢复 tick，并含边界钳制与坐标量化。真值由端点坐标差与独立弹道方程计算，不读取公开 velocity 或生产 physics 的结果。修复版位置、速度最大绝对误差均为 1.1641532182693481e-10。

广物理对照显式提供历史提交 `124ea433fe9a2ad2fd66a7360df85f6d61bab793` 的冻结源码，避免缺少历史树使 4 项对照跳过。详细 guard/变异断言、随机种子、每次计时与源码哈希见同目录 `validation-summary.json`。

## 公平计时

基线与修复版只切换 `scene_simulator.py`，同一进程共享未改依赖；每个工作负载先暖机一对，再以固定种子随机相邻顺序测 11 对。构造、导入、哈希与正确性判断在计时区间之外；两侧同样暂停 GC；计时不启用 tracemalloc。每项为 400 ticks，以下是该次机器实测中位数，未设置 CI 性能通过阈值。

| 工作负载 | 实体 | 基线 ms | 修复 ms | 修复/基线 | 增量 μs/实体/tick |
|---|---:|---:|---:|---:|---:|
| kinematic_commands | 64 | 106.227 | 117.863 | 1.1095 | 0.4545 |
| dynamics_free | 64 | 450.753 | 460.457 | 1.0215 | 0.3790 |
| dynamics_commands | 64 | 124.069 | 178.255 | 1.4367 | 2.1166 |
| dynamics_resume_8 | 8 | 34.794 | 38.123 | 1.0957 | 1.0404 |
| dynamics_resume_64 | 64 | 297.577 | 333.737 | 1.1215 | 1.4125 |
| dynamics_sparse | 64 | 444.700 | 447.200 | 1.0056 | 0.0976 |
| dynamics_singlebox | 1 | 6.071 | 6.485 | 1.0683 | 1.0370 |

最密集命令路径需要额外复制初值、换算与校验，成本如表保留；原始版错误的恢复轨迹仅作为性能参照。原始/修复正常自由动力学与运动学控制的端点指纹一致；受修复影响的恢复工作负载另外用独立端点 oracle 判断。

内存用独立 `--mode memory` 进程测量：4 个工作负载 × 400/4,000 ticks × 每侧 3 次，记录 current/peak、诊断历史长度与物理 state 数量，并单独排空该基准场景的诊断日志，读出仍存活的仿真状态。完整 Windows 门禁同样独立执行，真实脑库固定为 `98d6b899533a82fb1047f952987229c62a9929fb`。二者在本代码冻结时仍在运行，完成结果单独封存，不作为上表已完成结果。

## 来源绑定

- 代码冻结提交：`6d1048f6df5610de1e71c346e1f95182d1dfb84a`，基于 main `6b73082406d40a630a44cd68559c1f66b3652390`。
- 原始 scene LF SHA256：`45caf73deff9d23fa68ead8a4cbac2302587bba63bb2f161853e9d6fd8fa6151`。
- 修复 scene LF SHA256：`984d8abeaaa5a4b9d50e6d604758ac128d120f6b3bd56289b8b2a665a7bd1046`。
- rebase 前后 scene/physics/coord_contract/benchmark 的 LF 字节哈希均一致。
- 较早的“以已取整公开速度重锚”尝试与其数值/性能记录已标为中间证据；本目录只描述修正后的真实端点差实现。
