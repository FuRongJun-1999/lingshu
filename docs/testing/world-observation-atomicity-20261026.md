# 世界模型整帧观测失败时的状态保全

日期：2026-10-26

## 问题与修复

默认 legacy 的 perceive 逐条写入节点，再展开 conditions 等字段。后一条格式错误时，
前一条节点、置信和身份随机数已经变化，tick 已推进，但观测快照与历史尚未提交。
同一个 tick 的正确帧随即被“必须晚于当前时刻”拒绝，不能正常重试。
第一帧即失败时，预测包装还可能提前建立轨迹和误差缓冲；conservative 也有这项初始化副作用。

本件将整帧格式检查与数值转换移到生效轨迹包装的首部，早于懒缓冲、RNG 和 tick 更新。
conservative 的既有严格测量校验提取为共享函数，关联算法与未决观测规则不变。
不复制整个长期模型，不引入运行依赖；修复集中在 world_model 与 identity 两个既有模块。

## 基线与本地验收

原始 main 基线 `a4c8207996f2cf2e41db792a918a3b484abef382`；
源码验证提交 `e3453b8c8420e7c7c2333f6842d2ae73d6f7d447`。
同一守卫、同一基线来源分别跑原实现与修复，未因导入错误或 SKIP 记为成功证据。

| Windows / Python 3.12.10 / SQLite 3.49.1 | 实测结果 |
| --- | --- |
| 原 main 守卫 | 14 failed, 2 passed；0 errors / 0 skipped |
| 修复后守卫 | 16 passed；0 failed / 0 errors / 0 skipped |
| 相关回归 | 126 passed, 2 skipped, 49 warnings in 3.29s |
| 具名变异 | 3/3 各恰好点红指定用例：1 failed / 0 errors / 0 skipped |
| 登记清单 | 64/64 MATCH；0 mismatch / bytes_mismatch |
| 时间核 | new_violations 为空；保留 23 条既有登记 |

守卫覆盖：后一条 conditions/坐标/类别/维数出错，完整实例指纹和外持对象不变；
匿名分配 RNG 不漂移；修正后同 tick 重送，与未失败的独立对照身份一致；
原预测对象保留，重送真实输入后仍在原 target_tick 取得 1/1 核验命中；
默认传感器单次读取；真实 core 门面失败时 SQLite total_changes 不变；
空帧、首帧 tick=0 和 legacy 转换及非有限值诊断兼容。
“真实输入”在此指独立提供给公开 Python/core 入口的合成观测，不是实机传感器证据。

变异名称与指定断言：

- skip_frame_preflight → WM_REJECTED_FRAME_CHANGED_STATE。
- validate_first_row_only → WM_REJECTED_FRAME_CHANGED_STATE。
- initialize_buffers_before_validation → WM_FIRST_REJECTION_INITIALIZED_BUFFERS。

相关回归的两条 SKIP 是 test_scene_relations_round_trip_real_brain 和
test_experience_real_brain_restart_preserves_predictive_relations，均依赖本地未配置的脑端。
SKIP 不计入 passed。相关文件、源指纹、原始逐条输出见[同名证据目录](world-observation-atomicity-20261026/)。

## 复跑

在仓根安装项目的 .[full,dev]，然后运行：

```text
python -X utf8 -m pytest tests/test_world_observation_atomicity.py -q
python -X utf8 tools/testing/replay_save_guards.py --spec tests/fixtures/world_observation_atomicity_mutations.json --baseline a4c8207996f2cf2e41db792a918a3b484abef382 --mutate --output <新证据目录>
python -X utf8 tools/verify_manifest.py --strict --quiet
python -X utf8 tools/time_core_lint.py --json
```

baseline 需要完整 Git 对象；重放器只改输出目录内的临时源码副本，结束即清理。
本轮 Windows 只执行一个有效定向批次，未跑本地全套。日志只归一工作区/解释器前缀、
换行及行尾空白，保留断言、文件行号和失败读数。源指纹可对照固定验证提交。

## 边界与交付判定

判定为部分验证。WSL 在启动 Python 之前报
Wsl/Service/CreateInstance/CreateVm/HCS/ERROR_NOT_FOUND，故本地 Linux 重放未运行；
不改宿主服务/虚拟机配置，后续 Linux 完整回归另以 PR 当前 head 的 CI 为准。
在线脑/MCP、DSH、真实相机/机器人、生产数据、大规模性能及并发输入均未测。

本件是格式预检查，不是通用运行时事务：不保证自定义转换代码、并发改输入或后续任意异常回滚。
legacy 非有限坐标仍由既有运动诊断处理，不能把“格式被接受”写成“测量已经可信”。
本件覆盖的格式拒绝以 ValueError 报告；既有策略的其他异常未作统一。
不修改 CI、阈值或豁免，不宣称观测精度/预测性能提升，不提交脑端改动。

一个独立修复对应一个 PR 和一个 intake，维护者决定最终定级、验收与结算；human review pending。

交付前同步 main `c3f9178eb70c8cd43d317a33fb7989ef66b0f001`，无冲突；本件两份生产源码、守卫和变异规格与固定验证提交逐字节相同。
合入的其他 core/world3d 修复不计为本件贡献。当前完整集成以 PR head 的 CI 为准，未重跑稳定的本地定向批次。
