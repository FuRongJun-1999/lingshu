# W2 第一增量：保守身份关联

日期：2026-10-24

同类别匿名实体交叉或遮挡时，最近邻选择会把未确定的身份写成真实轨迹，继而影响置信、预测误差和隐藏目标假设。本件交付可显式启用的 `conservative` 模式，默认 `legacy` 保持原行为。

整帧只读候选检查给出 ACCEPT/DEFER/REJECT 关联回执；歧义、候选竞争或重复显式 ID 的测量保留审计，但不进入已确认快照、历史、置信或派生缓冲。只接受无竞争的唯一候选；零旧候选分配新的观测轨迹。视觉回填使用原始观测序号，SimLoop 等待主体的新鲜已确认观测再推进假设窗口。v2 经验保存策略与审计，v1 仍按 legacy 恢复；非法恢复不半替换运行态。

## 验收读数

基线 main `c6139540461aa20f72bb6aa15e1b31e83c772af3`；双平台源码验证提交 `0468db14a393ca80fa1cb54ec6789a2927154728`。这是新增功能交付，基线的 21 条失败由具名断言报告新接口缺失，1 条默认最近邻兼容对照通过；没有导入错误或 SKIP。基线回放恢复三个既有生产文件，新增纯函数模块存在但未被基线引用。完整基线场景另从 `git archive` 导出原始 main 运行。

| 平台 | 同一守卫在基线 | 新增功能 | 定点变异 |
| --- | --- | --- | --- |
| Windows / Python 3.12.10 | 21 failed, 1 passed in 1.12s | 22 passed, 17 warnings in 0.94s | 4/4 新变异；3/3 旧 ID 变异 |
| Linux / Python 3.13.12 | 21 failed, 1 passed in 1.79s | 22 passed, 17 warnings in 1.57s | 4/4 新变异；3/3 旧 ID 变异 |

4 组新变异分别为：把未决测量写入确认结果、按缩短后的序列回填视觉 ID、恢复时静默改为 legacy、用未决主体推进假设。每组只选择指定的一个用例，恰好出现 1 failed / 0 errors / 0 skipped，并核对指定断言。旧分配函数迁移后，原 3 组 ID 防碰撞变异仍逐组点红；没有削弱其条件或断言。

- Windows：`36 passed, 57 warnings in 3.17s`；无 SKIP。
- Linux：`36 passed, 57 warnings in 22.57s`；无 SKIP。

严格源码清单 `entries=64 match=64 mismatch=0`，时间核 0 新违规。每平台各一个有效定向批次；不跑本地全套，不为整理材料重跑稳定测试。Linux/DrvFS 只关闭 pytest 输出捕获，不改测试或断言。

## 固定场景对照

场景评分器通过真实 `UnifiedWorldModel.perceive` 调用评分，真值身份只留在模型外。五个固定场景为分离、交叉、逆序交叉、一个对象遮挡和显式 ID 交叉，共 9 条评分测量；不是摄像头或通用 MOT 基准。

| 模式 | 观测 | 已确认 | 正确 | 错误关联 | 暂缓 | 接受覆盖率 |
| --- | --- | --- | --- | --- | --- | --- |
| 原始 main / legacy | 9 | 9 | 4 | 5 | 0 | 9/9 |
| conservative | 9 | 4 | 4 | 0 | 5 | 4/9 |

新提交的 legacy 对照与原始 main 的场景 JSON 完全一致；保守模式保留 4 条正常/显式关联，暂缓 5 条歧义测量。不能据此宣称普遍识别率或速度提高，也不能把 DEFER 算作正确识别。

## 复跑与使用

安装项目 `.[full,dev]`，在验证提交仓根运行：

```text
python -X utf8 tools/testing/world_identity_scenarios.py
python -X utf8 -m pytest tests/test_world_identity_qualification.py -q
python -X utf8 tools/testing/replay_save_guards.py --spec tests/fixtures/world_identity_qualification_mutations.json --baseline c6139540461aa20f72bb6aa15e1b31e83c772af3 --mutate
python -X utf8 tools/testing/replay_save_guards.py --spec tests/fixtures/identity_reserved_ids_mutations.json --mutate
python -X utf8 -m pytest tests/test_wm_identity_one_to_one.py tests/test_identity_reserved_ids.py tests/test_trajectory_resolved_observations.py tests/test_world_prior_atomic_restore.py tests/test_issue243_world_facade_init_guard.py tests/test_wm_simloop_guards.py::test_ray_intersect_accepts_forward_intersection -q
python -X utf8 tools/verify_manifest.py --strict --quiet
python -X utf8 tools/time_core_lint.py --json
```

直接使用 `UnifiedWorldModel(identity_policy="conservative")`；引擎 `world_model("init", {"identity_policy":"conservative"})`，只读检查走 `identity`，策略查询/切换走 `identity_policy`。字段、资格含义、旧档兼容和截断恢复见 [`docs/world_identity_qualification.md`](../world_identity_qualification.md)。同名证据目录含逐腿原始读数、场景 JSON 和源码指纹，公开日志仅替换工作区前缀并归一化换行/行尾，原日志保留本地。

## 既有贡献与边界

保留 [#461 的视觉身份和经验入口](https://github.com/FuRongJun-1999/lingshu/pull/461)、[#857 的全帧保留 ID](https://github.com/FuRongJun-1999/lingshu/pull/857)、[#858 的已解析观测投影](https://github.com/FuRongJun-1999/lingshu/pull/858)、[#864 的原子恢复](https://github.com/FuRongJun-1999/lingshu/pull/864)。本件新增“未决身份不升级为观测事实”的可用模式，不重复申报这些已交付能力。目标仓和 bounty 所有状态的相关条目已检查，未见相同新模式交付。

判定为部分验证：已验公开 Python/核心门面、World3D 对象回填、实际 SimLoop 步进、隔离文件 read_content 恢复协议；未验真实摄像头、在线脑/MCP、DSH、私有 AEIS 或生产数据。显式 ID 仍依赖输入适配器；新轨迹不是物理新物体证明。关联资格不提升知识真实性或权限。候选开销 O(观测数×已知实体数)，未测大规模性能；未引入外观/ReID、全局最优匹配、跨遮挡多假设重关联或新坐标标定。本件是 W2 第一增量，完整 W2 与七层架构仍待后续开发。未修改大脑区、宿主配置、旧门禁/阈值或新增豁免。维护者技术审核和理论裁决待完成，human review pending。
