# 运动算术溢出的隔离与有限均值

日期：2026-10-28

## 问题与修复

有限坐标不保证中间运算有限。公开观测 (-1e308,0,+1e308) 的两段真实速度均为 1e308，
原求和再平均溢出为 Infinity，进入 UnifiedWorldModel/WorldLearner 学得参数；
(-1e308→+1e308) 的位移溢出还会给出 NaN 方向，严格 JSON 编码失败。

本件只修原生 motion.py：坐标闸要求三轴可转换且有限，捕获转换溢出；
真实间隔下原始差溢出时尝试先除时间再相减；无法表示的速度/模长/权重配对拒绝使用。
有效速度加权和溢出时按峰值缩放计算有限均值；正常数值路径与长间隔权重不变。
最近方向查询拒绝不可用坐标及溢出向量，不返回 NaN 方向。

motion_detail 仅在算术拒绝发生时增加 arithmetic_pairs_dropped；没有可用估计时
reason=invalid_motion_arithmetic，estimate=None/adopted=False。坏原始读数与坏运算分开计数，
保留其余有效配对，不伪造零速。原四元组、_estimate_detail 三参签名、上下文新证据计数保持。
motion.py 是原生文件，无须修改导出清单；派生清单仍通过严格校验。

## 最终集中验证

main 基线 `02caf8bfe6d53fb33981d2f51c0ebceef9a0957f`；最终固定源码验证提交 `926ed42f13ba330196a7c0b786103eaf0bc93f9e`。
Windows / Python 3.12.10 / SQLite 3.49.1；最终批次为 windows-final，未跑本地全套。

| 项目 | 实测 |
| --- | --- |
| main，同一守卫 | 13 failed、5 passed；0 errors / 0 skipped |
| 修复后 | 18 passed；0 failed / 0 errors / 0 skipped |
| 具名变异 | 4/4 各恰好 1 failed，命中指定断言；0 errors / 0 skipped |
| 相关回归 | 264 passed、2 skipped |
| 派生清单 | 64/64 MATCH |
| 时间核 | 0 新违规；23 条既有登记 |

具名变异：keep_unrepresentable_velocity_pair、sum_speeds_before_stable_mean、drop_recoverable_divided_endpoint、return_overflowed_recent_direction，分别放过不可表示配对、恢复溢出求和、关闭可恢复间隔路径、
放过最近方向中的非有限向量。每个只跑指定用例并命中指定断言，不计 errors/SKIP 为成功。
覆盖可表示均值、可恢复真实间隔、不可表示配对/模长、有效坏配对混合、坐标形状/转换、
上下文证据、正常密集/稀疏/静止、Unified/WorldLearner/真实 core 学得参数与 SQLite 零额外写入。

基线原始读数：均值为 Infinity；真实间隔 10 的速度仍为 Infinity/NaN 方向；
无法表示的配对也被 adopted=True。修复守卫核对前两者为 1e308、2e307，
不可表示配对为 None 且带显式诊断。极值为确定性合成输入，不是实际设备轨迹或性能收益基准。

## 首轮发现与既有守卫维护

首轮新守卫已通过，但相关回归有 2 failed、262 passed、2 skipped：
一个旧量具替身只接三参，发现新增诊断参数破坏私有替换口；现恢复三参并把行构造单独收敛。
另一条旧 G5 自证假定旁路坐标闸后坏值必能污染，新增算术防线后该假定不成立。
本件将 G5 改为断言旁路第一道闸时第二道仍拒绝坏值；G4 的原有限性/诊断断言保留，
新变异件独立点红算术闸。没有删除测试、放宽阈值、加豁免或将失败标为 SKIP。
既有守卫文件的最终指纹也列入本批源指纹；首轮失败日志保留在本地，不冒充最终结果。

## 复跑与清理状态

仓根安装 .[full,dev] 后：

```text
python -X utf8 -m pytest tests/test_world_motion_finite_arithmetic.py tests/test_issue418_sparse_motion_guard.py -q
python -X utf8 tools/testing/replay_save_guards.py --spec tests/fixtures/world_motion_finite_arithmetic_mutations.json --baseline 02caf8bfe6d53fb33981d2f51c0ebceef9a0957f --mutate --output <新证据目录>
python -X utf8 tools/verify_manifest.py --strict --quiet
```

重放只恢复规格中的 motion.py，其余依赖取运行检出；严格重现应检出最终固定验证提交。
[最终证据](world-motion-finite-arithmetic-20261028/)。公开日志仅归一工作区/解释器前缀、换行与行尾空白，原日志留本地。
最终批次的临时副本已清理。初轮的 <workspace>/outputs/world-motion-finite-arithmetic-20261028/windows/owned-test-temp
仍保留：自动审批审查拒绝了清理命令，只返回 blocked by policy；没有绕过或重试该删除。
因此不声称本轮所有测试目录都已清理。

## SKIP 与未测边界

test_scene_relations_round_trip_real_brain；test_experience_real_brain_restart_preserves_predictive_relations，均为本地未配置脑端依赖，不计 passed。本轮本地 Linux 重放/变异未执行，远端 Linux CI 单列。
真实设备、在线脑/MCP、DSH、生产数据、性能和并发未测。

不是任意精度计算；不恢复全部极值浮点信息，不校准极值坐标的预测范围，
也没有重写关系余弦内核、位置/旋转生成算法或原始历史。未改阈值、CI、依赖及脑端代码。
按数值鲁棒性局部交付申报，不声称整体运动建模或性能提升；human review pending。
