# 无接触碰撞箱不改写自由运动末速度

基线：`d74848fb33dbff1d719959df19d1266a1ac60f44`。

`SceneSimulator._resolve_collisions` 原来只要登记了至少两个碰撞箱，即便所有
接触读数都是 `hit=False`，也把动力学实体的瞬时末速度改成 `净位移 / dt`，再把
平均速度重播种为下一 tick 初速。恒加速运动中两者不同，因此自由落体每 tick
只增加半个应有的重力速度；同 tick 内核 `physics_readout.vel` 与实体 `velocity`
也当场分叉。

本次只改变**没有参与任何已报告接触**的动力学实体：保留 `simulate` 返回的
物理末速度与原闭式初值。所有 `hit=True` 的路径继续使用既有碰撞位移/速度及
重播种处理；运动学路径、内核接触判据与碰撞算法不变。

## 独立解析判据

预期由测试自身的 `p(t)=p0+v0*t+a*t*t/2`、`v(t)=v0+a*t` 给出，不调用本仓
`physics.simulate` 产生答案。例：`g=10, dt=.1, p0_y=100, v0_y=0`，实体与
30 米外的墙分别登记半宽 `.25` 的碰撞箱，十个 tick 均无接触：

| 时刻 | 基线 y / vy | 修复后及解析值 y / vy |
|---|---|---|
| tick 1 | 99.95 / -0.5 | 99.95 / -1 |
| tick 10 | 97.25 / -5 | 95 / -10 |

守卫覆盖三种 up 轴、四种 dt、两种重力，以及 0/1/2 箱对照、OBB、第三个
自由实体与另一对真实碰撞共存、两种插入序、真实接触、初始穿透与运动学
位移速度。额外边界保留 t=0 接触、完全重叠、静态重叠及“自由端点恰好等于
推离端点”的既有语义。末点恰好相切按当前内核 `hit=False` 契约处理。

`41 passed`；原基线使用同一守卫为 `32 failed, 9 passed`。六个具名变异各
导致指定用例恰好一条断言失败，均无 errors/skips，包括恢复平均速度、无接触
重播种、跳过真实接触处理、删除接触重播种、丢失运动学位移速度，以及错误地
只凭位置相等跳过已报告接触。

## 复跑

```text
python -B -X utf8 -m pytest tests/test_scene_collision_freeflight_velocity.py -q
python -B -X utf8 tools/testing/replay_save_guards.py --spec tests/fixtures/scene_collision_freeflight_velocity_mutations.json --baseline d74848fb33dbff1d719959df19d1266a1ac60f44 --mutate --output <新证据目录>
python -B -X utf8 bench/bench_scene_collision_freeflight.py --baseline d74848fb33dbff1d719959df19d1266a1ac60f44 --repeats 11 --ticks 80 --output <新基准目录>
python -B -X utf8 tools/verify_manifest.py --strict --quiet
python -B -X utf8 tools/time_core_lint.py --quiet
```

基准保留 baseline/fixed 源文件、依赖 SHA-256、各 worker 原始输出与全部样本。
每个来源/场景/模式使用独立解释器，任务顺序固定种子随机化；一轮预热、11 轮
时间测量、3 轮内存测量；构造/导入/哈希/oracle 均在计时区间外，两腿同样暂关 GC。
内存是 `tracemalloc` 记录的 `step` 内 Python 分配峰值，不代表进程 RSS 或原生库
总内存；性能读数是共享宿主上的描述性测量，不设时间门槛。

七个基准场景包括默认运动学、有箱运动学、默认动力学、一个箱、8/24 个无接触
动力学箱，以及碰撞对与第三个自由实体混合。默认/一箱控制的完整规范输出哈希
要求两腿一致；有缺陷基线的 oracle 红数如实记录，不将其视为通过。

## 查重与边界

全状态上游 Issues/PR 和 bounty 标题正文快照分别检查了 1307 / 2194 项。
最近邻 PR #1162 与 bounty #2027 处理的是 beta/radius/mass 丢失及 static
重播种；不包含本次“无接触平均速度覆盖末速度”问题。

已有切向滑行问题（bounty #2362）与物理预测层已知缺测年龄待办没有纳入本件。
本次不改变已有接触后的平均位移速度近似，不新增弹性反弹、冲量、摩擦或角动量
模型，不宣称所有接触动力学已符合连续物理。

## 本机验证读数

- Python 3.12 / Windows：上述守卫及仿真/碰撞/环境/时空一致性/manifest/time-core
  邻接组合 **243 passed、0 failed、0 skipped**，221.30 秒。
- strict manifest **64/64 MATCH**；时间核 **23 条登记、0 新违规、0 stale**。
- 隔离 worker 中 fixed 全部解析检查通过；缺陷基线的 8/24 箱与混合场景分别保留
  16/48/2 个终点位置或速度错误。默认运动学、有箱运动学、默认动力学和单箱控制
  的规范输出哈希两腿相同。

| 场景 | 隔离 worker 延迟 fixed/base | 跟踪峰值 fixed/base |
|---|---:|---:|
| 默认运动学 | 0.981 | 1.000 |
| 8 箱运动学 | 0.852 | 1.000 |
| 默认 8 实体动力学 | 0.907 | 1.000 |
| 单箱动力学 | 0.914 | 1.000 |
| 8 箱无接触动力学 | 1.008 | 1.000 |
| 24 箱无接触动力学 | 1.194 | 0.997 |
| 碰撞对 + 自由实体 | 0.944 | 0.999 |

24 箱隔离读数有约 19% 延迟增加，因此另做相邻、随机交换顺序的 11 对测量：
其“中位延迟之比”为 1.160，“逐对延迟比的中位数”为 1.016，原始单次读数
在约 0.93–1.46 秒间波动。不能由这些共享宿主样本声称统一加速，也不能把
描述性成本隐藏为零；保留全部样本供复核。其余相邻配对场景的中位延迟之比
在 0.985–1.017。该修复以恢复物理正确性为目标。
