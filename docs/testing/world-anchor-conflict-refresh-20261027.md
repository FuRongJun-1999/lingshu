# 锚点冲突登记后的即时状态刷新

日期：2026-10-27

## 问题与修复

公开 core.world3d("verify_conflict") 已登记真实冲突并清零冲突通道证据，却只读取旧缓存。
已经 ACCEPT_strong/stable 的锚点仍在返回值、anchor_state 和 verification_summary 中保持已接受，
旧稳定轮数和 tier_basis 也未清理，直到另一次 verify 才更新。

本件只改 AnchorVerification.channel_conflict_detect：冲突入账后立即调用现有 verify_anchor，
按原全称量闸重新判定；不改阈值、不添加新资格规则、不向可信度注册表记新样本。
相同观测（大小写/空白归一后相同）保持原空操作；未知锚点仍由图层拒绝。

## 固定源码与集中验证

main 基线 `02caf8bfe6d53fb33981d2f51c0ebceef9a0957f`；固定源码验证提交 `0fd1f33cf55bdd77923221fffa2b964cc3f734f1`。
Windows / Python 3.12.10 / SQLite 3.49.1，一次集中定向批次；未跑本地全套。

| 项目 | 实测 |
| --- | --- |
| main，同一守卫 | 7 failed、3 passed；0 errors / 0 skipped |
| 修复后 | 10 passed；0 failed / 0 errors / 0 skipped |
| 具名变异 | 3/3 各恰好 1 failed，命中指定断言；0 errors / 0 skipped |
| 相关回归 | 103 passed、0 skipped |
| 派生清单 | 64/64 MATCH |
| 时间核 | 0 新违规；23 条既有登记 |

具名变异：skip_conflict_refresh、retain_stable_count_after_refutation、return_stale_conflict_reply。分别阻断状态刷新、稳定轮数清零或让返回值仍使用旧快照，
每个只跑指定用例并核对指定断言；不是将导入错误、未收集或普通异常当作验证成功。
覆盖 strong/stable、稳定时间戳与排序依据、即时摘要、真实 core 与语义锚点图入口。
控制项覆盖一致观测、未知锚点、其他锚点、注册表计数与证据序号不变。
稳定轮数由注入时钟生成合成证据，不等待或冒充设备采样。

基线原始 core 读数：冲突前 ACCEPT_strong；冲突后 tactile=0、conflict_count=1，
但 confirmation 仍是 ACCEPT_strong，旧排序依据仍称 tactile evidence=0.9。
修复守卫要求当次返回 NOT_ACCEPTED、稳定轮数清零、排序依据为空；SQLite 写入计数不变。

## 复跑

仓根安装 .[full,dev] 后：

```text
python -X utf8 -m pytest tests/test_world_anchor_conflict_refresh.py -q
python -X utf8 tools/testing/replay_save_guards.py --spec tests/fixtures/world_anchor_conflict_refresh_mutations.json --baseline 02caf8bfe6d53fb33981d2f51c0ebceef9a0957f --mutate --output <新证据目录>
python -X utf8 tools/verify_manifest.py --strict --quiet
```

重放只恢复规格中列出的生产源码，其他依赖取运行检出；严格重现应检出固定源码验证提交。
临时副本只在输出目录内创建并已清理；公开日志仅归一工作区/解释器前缀、换行与行尾空白。
原始日志留本地，[公开证据](world-anchor-conflict-refresh-20261027/)。

## SKIP 与未测边界

本批相关回归无 SKIP。跳过项不计 passed。本轮未做本地 Linux 重放或变异，远端 Linux CI 单独核对。
真实设备、在线脑/MCP、DSH、生产数据、性能和并发未测。这里验证的是冲突状态一致性，
不证明冲突内容在现实中为真，不更改冲突解除、证据接收或租约策略。
未修改 CI、豁免、阈值、依赖或脑端代码；human review pending。
