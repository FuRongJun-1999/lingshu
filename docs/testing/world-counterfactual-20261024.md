# W3 第一增量：隔离关系条件试演

日期：2026-10-24

当前 generate 会提交预测缓存，并可能更新模式、关系和诊断；它是有状态预测入口。为了比较另一种关系条件而直接在真实实例上生成，会替换原本等待后续观测核验的预测。本件新增显式 counterfactual 入口，在两份已观测模型副本中比较原预测与 seek/flee 候选；普通入口维持原行为。新增功能的缺失不是现有 generate 的高优先级漏洞。

副本保留轨迹、误差、旋转、随机状态和已配置物理环境，屏蔽模拟器 world 及其别名；不推进物理世界，不把内部预测喂作 actual，不访问未知目标真相。候选条件先校验，在候选中替换对应来源关系，复用生效预测链；返回原/候选预测、位移差和 source_tick/target_tick，明确 provenance=counterfactual、verification=unverified、committed=false。调用和异常均不替换真实图、观测历史、预测缓存或经验。

## 基线与验收

独立基于最新 main `aadc9f13d70be7536b6b2e1c753e325b4859f66f`；双平台源码验证提交 `870f25be5919bb680825892954cd40e86bcd872d`。同一守卫在 main 为 22 failed、1 passed，具名断言报告新接口缺失；旧 generate 对照通过，无导入错误或 SKIP。基线场景另从原始 main 的 git archive 导出运行。

| 平台 | 基线 | 功能实现 | 变异自证 |
| --- | --- | --- | --- |
| Windows / Python 3.12.10 | 22 failed, 1 passed, 9 warnings in 1.11s | 23 passed, 9 warnings in 0.91s | 4/4 具名变异各点红指定用例 |
| Linux / Python 3.13.12 | 22 failed, 1 passed, 9 warnings in 1.92s | 23 passed, 9 warnings in 1.09s | 4/4 具名变异各点红指定用例 |

4 组变异是：复用真实实例生成对照、复制模拟器隐藏状态、core 丢关系条件、把候选标作已提交。每组仅跑指定用例，恰好 1 failed / 0 errors / 0 skipped，核对指定断言；不把普通异常当作守卫命中。

- Windows：`84 passed, 58 warnings in 9.07s`；0 skipped。
- Linux：`84 passed, 58 warnings in 45.67s`；0 skipped。

相关回归覆盖观测/运动、追逐、目标 tick、未观测核验、模式窗口、原子经验恢复、core 初始化与文档口径。严格清单 entries=64 match=64 mismatch=0，时间核无新增违规。每平台一个有效定向批次，未跑本地全套。

## 固定场景读数

实际通过 perceive → 当前预测链 → counterfactual；6 帧已知 ID 轨迹 actor 从 x=6 到 11，参照 goal 在 x=30。预测比较，不作为真实执行或一般精度基准。两平台一致：

| 条件 / horizon | 原预测 x | 候选 x | 差异距离 | 原真实预测缓存 |
| --- | --- | --- | --- | --- |
| 无条件 / 1 | 12 | 12 | 0 | 原对象、target_tick=8 保留 |
| seek / 1 | 12 | 12 | 0 | 原对象、target_tick=8 保留 |
| flee / 1 | 12 | 10 | 2 | 原对象、target_tick=8 保留 |
| flee / 3 | 14 | 8 | 6 | 原对象、target_tick=8 保留 |

每个案例完整实例指纹前后一致，重复调用读数一致。原生 generate 对照在原始 main 与本提交完全一致：先生成 horizon=2，再用 horizon=1 试算，会将当前预测 target_tick 从 8 替换为 7。这是旧入口的正常有状态语义，本件提供另一种隔离消费方式。守卫还实际在试演后送入 tick=8 的观测，原预测仍 2/2 核验命中；core 走真实引擎，SQLite total_changes 不变。

## 与前两轮的组合

[#905 保守身份关联](https://github.com/FuRongJun-1999/lingshu/pull/905) 与 [#906 行动反馈](https://github.com/FuRongJun-1999/lingshu/pull/906) 尚待人审，本件不包含或重复申报它们。Windows 对三份固定源码做真实三方文本合并，68 条原功能守卫加 1 条身份策略与试演联用守卫共 69 passed / 0 skipped。world_model/core/seven_layer_loop 等组合源码指纹已保存。工具首次在测试前被 Windows git archive 的 CRLF 与 git show 的 LF 差异拦下；统一换行后源码三方合并均返回 0，未丢弃一方实现或改写守卫。

此项只是隔离源码组合，不是正式 Git/清单/全套合并验收；Linux 三项组合未跑。没有宣称 SevenLayerLoop 与统一模型已变为同一运行实例。复跑工具随本件入仓，实际执行版本指纹已登记。

## 复跑与边界

```text
python -X utf8 tools/testing/world_counterfactual_scenarios.py
python -X utf8 -m pytest tests/test_world_counterfactual_isolation.py -q
python -X utf8 tools/testing/replay_save_guards.py --spec tests/fixtures/world_counterfactual_mutations.json --baseline aadc9f13d70be7536b6b2e1c753e325b4859f66f --mutate
python -X utf8 tools/verify_manifest.py --strict --quiet
python -X utf8 tools/time_core_lint.py --json
python -X utf8 tools/testing/replay_world_branches.py --revisions 870f25be5919bb680825892954cd40e86bcd872d d677833242c193832e40987faae3de5c838f36be e1d4732df8209fa34961e3809aa7529cc0d2483a --output <新输出目录>
```

安装项目 .[full,dev] 后在仓根运行；组合引用固定本地提交，缺对象时先获取对应贡献分支。相关回归文件与完整命令保存在 source-provenance.json，逐条输出、基线及变异原文在同名证据目录。公开日志只归一化工作区前缀、换行和行尾空白，原文保留本地。使用契约见[固定验证提交说明](https://github.com/STARDUSTLC666/lingshu/blob/870f25be5919bb680825892954cd40e86bcd872d/docs/world_counterfactual.md)。

判定为部分验证：已验公开 Python/core、真实预测包装链、显式物理环境诊断、成功/异常隔离、原预测后续核验和 core 数据库零写入。真实相机、在线脑/MCP、DSH、机器人、生产观测、完整物理开关全套、大规模性能及并发快照均未测。horizon 1..64 是显式计算预算，不宣称视界标定；复制成本随模型状态内存增长。仅支持可深拷贝的原生模型，扩展连接/自定义代码不属通用沙箱保证。

仅交付 W3 第一增量：不做动作序列执行/搜索、一般因果识别、完整物理分支或跨分支长期学习。未新增运行依赖或生产模块、未改脑端、宿主、CI、门禁阈值或豁免。复用现有模型及预测链，目标仓/bounty 全状态已查重；未见相同隔离 API 交付。维护者技术与理论人审待完成，human review pending。
