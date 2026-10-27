# 世界模型四元数归一化与无效旋转隔离

日期：2026-10-26

## 问题与修复

旋转归一化直接平方求模：大数溢出后返回全零，小数下溢后被改写为单位姿态。
NaN/Inf 未检查就写进旋转缓冲，一条坏读数即可让后续 predicted_rot 成为 NaN；
零四元数也被当作已测得单位姿态。不可转换的大整数会在位置已更新后从旋转层抛出 OverflowError。

本件只改 _rot_qnorm 与 _perceive_with_rot：四项有限非零值先按最大分量缩放再归一化，
转换异常归为该通道的拒绝，保留同帧有效位置。拒绝不进入旋转缓冲/已接受计数；
回执给出已解析实体、原因与帧内来源序号，legacy 不采信上游索引覆盖，conservative 保留原序号。
没有旋转或没有拒绝时不增加拒绝字段；无旋转历史仍走原位置输出。

## 最终定向批次

main 基线 `c3f9178eb70c8cd43d317a33fb7989ef66b0f001`；固定验证提交
`3d8e7ed0884f9f91487af9cd03434268c0e92e92`。
Windows / Python 3.12.10 / SQLite 3.49.1；最终批次反映来源索引边界补充后的代码，
初轮日志另留本地，不以初轮计数替代最终读数。未跑本地全套。

| 项目 | 实测 |
| --- | --- |
| 原 main，同一守卫 | 21 failed, 4 passed；0 errors / 0 skipped |
| 修复后 | 25 passed；0 failed / 0 errors / 0 skipped |
| 具名变异 | 6/6 各恰好点红指定用例，1 failed / 0 errors / 0 skipped |
| 相关回归 | 159 passed, 2 skipped |
| 登记清单 | 64/64 MATCH，0 mismatch / bytes_mismatch |
| 时间核 | 0 新违规；保留 23 条既有登记 |

守卫覆盖有限缩放到 1e308、1e-320、负号等价、旋转流预测与单位流对照；
NaN、正负 Inf、零、转换溢出、文本、形状和布尔值的隔离；只有坏测量不声称旋转预测；
新有效样本继续接入、无须清空整个缓冲；原始/已过滤序号、真实 core 门面和无旋转输出对照。
六个变异为 accept_nonfinite_quaternions、invent_identity_for_zero_quaternion、square_before_scaling、
let_numeric_conversion_overflow_escape、drop_rejection_receipt、use_filtered_observation_index。
每个只跑指定用例并核对指定断言，不把导入错误、未收集或普通异常当成功。

固定输入的旧实现实测：同姿态乘 1e300 后归一化为全零，乘 1e-300 后变成单位姿态；
零四元数也返回单位姿态；一条 NaN 旋转进入 7 点缓冲后生成 NaN。
修复用四个等分量的严格归一化断言、有效旋转流对照与后续预测严格 JSON 编码核对，
对浮点已丢失的信息不作恢复承诺。真值与控制流是合成数据，不是实机精度基准。

## 与旋转时间修复的组合

以本件固定验证提交和 [#1007 固定 head](https://github.com/FuRongJun-1999/lingshu/pull/1007)
`f411d1e0a73fbadf36e0b5b315b9e0b1421bc240` 做真实源文件三方合并，统一 CRLF/LF 后无冲突。
隔离副本执行本件 25 条与既有时间守卫 20 条：45 passed、0 skipped。
合成源、两个守卫和执行工具指纹见同目录 combination.json；临时副本已清理。

这只证明隔离源码组合，不是正式 Git/清单/全套合并验收；Linux 组合未跑。
本 PR 不携带或重复申报 #1007 的生产改动或 20 条守卫。

## 复跑

在仓根安装 .[full,dev] 后运行：

```text
python -X utf8 -m pytest tests/test_world_rotation_quaternion_validity.py -q
python -X utf8 tools/testing/replay_save_guards.py --spec tests/fixtures/world_rotation_quaternion_validity_mutations.json --baseline c3f9178eb70c8cd43d317a33fb7989ef66b0f001 --mutate --output <新证据目录>
python -X utf8 docs/testing/world-rotation-quaternion-validity-20261026/replay_combination.py --output <新组合目录>
python -X utf8 tools/verify_manifest.py --strict --quiet
```

基线重放只恢复规格列出的 world_model 源码，其余依赖取运行检出；严格重现原记录应检出固定验证提交。
组合工具需要本件与 #1007 固定 Git 对象（缺少时先取相应贡献分支），不联网；
只在给定输出目录内生成副本并清理。公开日志仅归一工作区/解释器前缀、换行和行尾空白，
原日志留本地。[完整证据](world-rotation-quaternion-validity-20261026/)。

## SKIP 与边界

判定为部分验证。相关回归的两条 SKIP 为 test_scene_relations_round_trip_real_brain、
test_experience_real_brain_restart_preserves_predictive_relations，依赖本地未配置脑端，不计 passed。
WSL 在 Python 前报 Wsl/Service/CreateInstance/CreateVm/HCS/ERROR_NOT_FOUND，
Linux 本地重放/变异未运行；后续当前 head 的 Linux CI 单独验收，不冒充本机重放。

原始坏测量保留在 observations/身份审计中；严格 JSON 断言只覆盖后续预测。
本件预防新的坏旋转写入，不迁移已污染的旧缓冲，不改变位置的既有非有限值诊断，
不修改旋转时钟、多轴假设、阈值、依赖、CI 或豁免，不提交脑端改动。
真实设备、在线脑/MCP、DSH、生产数据、大规模性能和并发未测。
一个独立修复对应一个 PR/intake，维护者决定定级、验收及结算；human review pending。
