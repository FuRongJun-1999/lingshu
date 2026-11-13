# 非整数周期远期预测相位回绕

日期：2026-10-27

## 触发与修复

显式 set_period_caliber(noninteger=True) 后，位置检测器可返回浮点周期。
_phase_at 只减去一次周期，horizon 大于周期时越过历史末端，随后被钳为最后观测值；
输出仍标 periodic，目标时刻仍前进，位置却停止变化。默认整数周期分支本已有回绕。

本件只在 _phase_at 的浮点分支、h > P 时按原浮点周期回绕；保留小数相位及线性插值，
整周期落在末端观测锚点。整数分支、h <= P 的路径、默认关闭的非整数开关、检测阈值、
范围/置信度口径和生成链优先级保持原代码。仅同步派生清单指纹，无运行依赖或脑端改动。

## 固定源码与实测

main 基线 `3090b87121dedc920d0cdae9529ff928cce9da29`；固定源码验证提交 `e6b1910c4b5427abf120d596f09c812752de171f`。
Windows / Python 3.12.10 / SQLite 3.49.1；一次集中定向批次，未跑本地全套。

| 项目 | 实测 |
| --- | --- |
| main，同一守卫 | 11 failed、7 passed；0 errors / 0 skipped |
| 修复后 | 18 passed；0 failed / 0 errors / 0 skipped |
| 具名变异 | 4/4 各恰好 1 failed，命中指定断言；0 errors / 0 skipped |
| 相关回归 | 213 passed、2 skipped |
| 派生清单 | 64/64 MATCH |
| 时间核 | 0 新违规；23 条既有登记 |

四个变异为 skip_float_phase_wrap、truncate_fractional_period、lose_whole_cycle_anchor、
drop_fractional_interpolation。只运行指定用例并核对指定断言，不把导入错误/未收集当成功。
守卫覆盖跨一周期与多周期、41/50/64/130 步、非整数与整数值浮点周期、小数插值、
整周期锚点、空历史、短期对照与默认整数模式。

独立解析真值为 [10+cos(2πt/40.5),1.5,10+sin(2πt/40.5)]，只喂 t=0..95 的观测。
公开 generate(horizon=50) 检出 P=40.5：旧输出等于 t=95 的最后观测，点误差
1.3440157211110424；修复后读取已有周期相位，点误差小于 1e-12。
基线与修复探针完整读数另列，其他目标受历史线性插值误差影响，守卫容差 0.004。
这是确定性合成场景；不作为一般精度或性能收益基准。

真实 SpacetimeMemoryEngine.world_model 的 perceive/generate 门面同样通过，SQLite
total_changes 不变。counterfactual 的 50 步两候选使用正确相位，保留
provenance=counterfactual / verification=unverified / committed=false，原实例和预测缓存不变。
候选正确使用相位，不等于真实观测、因果证明或实际设备验收。

## 复跑与证据

仓根安装 .[full,dev] 后：

```text
python -X utf8 -m pytest tests/test_world_period_phase_wrap.py -q
python -X utf8 tools/testing/replay_save_guards.py --spec tests/fixtures/world_period_phase_wrap_mutations.json --baseline 3090b87121dedc920d0cdae9529ff928cce9da29 --mutate --output <新证据目录>
python -X utf8 tools/verify_manifest.py --strict --quiet
```

重放只恢复规格中的 world_model，其他依赖取运行检出；严格重现应检出固定源码验证提交。
独立数值例可复用守卫里的 circle、observe，显式开启 noninteger 后调用 generate(50)。
运行期全局配置在测试/探针结束时恢复。临时副本已清理；
公开日志仅归一工作区/解释器前缀、换行和行尾空白，原日志留本地。
[完整证据](world-period-phase-wrap-20261027/)。

## 范围与未测

相关回归两条 SKIP：test_scene_relations_round_trip_real_brain、
test_experience_real_brain_restart_preserves_predictive_relations，本地未配置脑端，不计 passed。
本轮没有启动本地 Linux/WSL；Linux 本地重放、变异和探针未执行，远端 Linux CI 单列验收。
真实设备、在线脑/MCP、DSH、生产数据、噪声/变周期泛化、长时置信度校准、性能及并发未测。

没有修位置的稀疏采样/观测年龄换算、周期误检、突变模型或多步范围校准；
只修已检测周期的跨周期取址。没有修改 generate 的 horizon 输入规则、阈值、CI 或豁免。
与旋转时钟 [#1007](https://github.com/FuRongJun-1999/lingshu/pull/1007) 和旋转测量
[#1056](https://github.com/FuRongJun-1999/lingshu/pull/1056) 的修复分开；本件独立基于 main。
一项独立 PR/intake；定级、验收与结算由维护者决定，human review pending。
