# 世界模型旋转预测的 tick 时间轴修复

日期：2026-10-26

## 问题与实现

旋转缓冲已经记录 (tick, quaternion)，预测却丢掉 tick，以观测条数计算角速度和周期，
并从最后旋转样本只外推 horizon。显式稀疏 tick、不等间隔、旋转缺测期间的空帧或位置帧
都会让旋转角与同一预测里的 target_tick 不一致。滑窗平均角速度还被误当作窗口末端速度。

本件只修改既有 _rot_predict：用真实间隔求角速度，按滑窗中点估加速度，校正末端速度，
从最后旋转观测推进到 model.tick + horizon；等间隔周期换算为世界 tick，并在采样格插值/回绕。
不等间隔不进入索引周期检测器，仍可走局部速度/加速度外推。新增内部时钟和单位诊断。
位置预测、身份策略、解缠、既有阈值、运行依赖均不改。无旋转历史的输出与原位置链逐字段一致。

## 基线与实测

main 基线 `c3f9178eb70c8cd43d317a33fb7989ef66b0f001`；固定源码验证提交
`ee83c19d88767e4883109e800d539def3d788999`。
Windows / Python 3.12.10 / SQLite 3.49.1，执行一个有效定向批次，未跑本地全套。

| 项目 | 实测 |
| --- | --- |
| 原 main，同一守卫 | 16 failed, 4 passed；0 errors / 0 skipped |
| 修复后 | 20 passed；0 failed / 0 errors / 0 skipped |
| 具名变异 | 5/5 各恰好点红指定用例，1 failed / 0 errors / 0 skipped |
| 相关回归 | 159 passed, 2 skipped, 57 warnings in 4.67s |
| 登记清单 | 64/64 MATCH，0 mismatch / bytes_mismatch |
| 时间核 | new_violations 为空；23 条既有登记保留 |

旋转真值来自独立合成函数 angle(t)=0.05t、accelerated(t)=0.05t+0.005t² 和三角周期函数，
由这些函数生成四元数后实际送入公开 perceive/generate，而非直接调用新增计算片段。
角度断言容差 1e-8 rad；只说明这些固定场景，没有宣称一般精度收益。

| 场景 / 目标 tick | 旧预测 rad | 修复后的预期值 rad（守卫通过） |
| --- | --- | --- |
| 匀速，每 2 tick 采样 / 11 | 0.60 | 0.55 |
| 匀速，不等间隔 / 13 | 0.85 | 0.65 |
| 匀速，最后旋转样本为 tick=5、当前 tick=10 / 11 | 0.30 | 0.55 |
| 匀速，只有新位置帧 / 17 | 0.70 | 0.85 |
| 加速，单位间隔 / 7 | 0.575 | 0.595 |
| 加速，每 2 tick 采样 / 12 | 1.60 | 1.32 |
| 加速，不等间隔 / 13 | 2.085 | 1.495 |

旧值逐条见 baseline 日志，固定场景的修复值由同一守卫断言。另覆盖大整数时钟原点、
半个采样间隔的周期相位、跨多个周期、缺测后周期、真实 core 门面及隔离 counterfactual。
core 生成不增加 SQLite total_changes；试演保持原实例指纹与原预测对象，候选不提交为观测。
位置 verify 的评分不作为旋转命中的证据。

五个变异：ignore_rotation_observation_age、treat_samples_as_ticks、
use_window_velocity_as_endpoint_velocity、skip_sample_phase_conversion、
infer_index_period_on_nonuniform_samples。每个仅跑规格指定的用例，核对指定断言，
不把导入错误、未收集或普通异常当守卫成功。源码、规格、逐条输出和指纹见[证据目录](world-rotation-tick-clock-20261026/)。

## 复跑与证据范围

```text
python -X utf8 -m pytest tests/test_world_rotation_tick_clock.py -q
python -X utf8 tools/testing/replay_save_guards.py --spec tests/fixtures/world_rotation_tick_clock_mutations.json --baseline c3f9178eb70c8cd43d317a33fb7989ef66b0f001 --mutate --output <新证据目录>
python -X utf8 tools/verify_manifest.py --strict --quiet
python -X utf8 tools/time_core_lint.py --json
```

仓根安装 .[full,dev]。重放器只把规格列出的 world_model 源码换回 baseline，其余依赖取运行检出；
严格重现原记录应检出固定验证提交。临时副本在输出目录内创建并已清理。
公开日志仅归一工作区/解释器前缀、换行及行尾空白，原日志保留本地。

判定为部分验证。相关回归的两条 SKIP：test_scene_relations_round_trip_real_brain、
test_experience_real_brain_restart_preserves_predictive_relations，依赖本地未配置脑端，未计入 passed。
本地 WSL 在 Python 启动前报 Wsl/Service/CreateInstance/CreateVm/HCS/ERROR_NOT_FOUND，
Linux 本地重放和变异未运行；Linux 完整回归以后续 PR 当前 head CI 单独验收，不冒充本机重放。

真实设备、在线脑/MCP、DSH、生产数据、大规模性能及并发未测。保留既有单轴投影/解缠的假设，
不解决多轴刚体、采样混叠、噪声或运动突变；不宣称所有旋转模式都已验收。
未改 CI、门槛或豁免，未提交脑端改动。本件独立于 #1005 观测帧格式修复，
不携带其实现、守卫或重复申报其贡献。最终验收与结算由维护者决定；human review pending。
