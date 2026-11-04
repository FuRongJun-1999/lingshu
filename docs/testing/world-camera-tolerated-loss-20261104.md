# 相机容差内焦距损失显式报告

相机标定桥用单一 fx 表达 Camera3D 的水平和纵向投影，却只在 fx/fy 相对差超过容差时记录损失。fx=1000、fy=1005、容差 1% 时允许转换，并错误返回 lossy=[]、representable_exactly=True；世界点 (0,1,10) 的纵向投影相对标定内参有 0.5 px 残差。

任何非零 fx/fy 差别都写入已有 lossy 报告并将 representable_exactly 置假。strict 的拒绝门槛仍是原有 tol_frac；容差内继续允许近似，投影、fov 公式与相机表达能力不变。

对应 README 的真实相机接入及 camera_bridge 的显式报告损失契约；容差决定是否接受近似，不能证明数学上无损。

基线 `891c28a7471a31275cc6d37a4468b62e49fe0694`；固定功能验证源码 `5343b0e32a3c63c7a2367e54fd012ebe9fbc6cdb`。建议 P2/medium/B，最终定级、验收与结算由维护者决定。

## 集中验证

[原始证据](world-camera-tolerated-loss-20261104/)；Windows / Python 3.12.10 / SQLite 3.49.1。

| 检查 | 实测 |
| --- | --- |
| 基线同一新守卫 | 4 failed、3 passed；0 errors、0 skipped |
| 修复后新守卫 | 7 passed；0 errors、0 skipped |
| 具名定点变异 | 2/2，各恰好一个指定断言失败；0 errors、0 skipped |
| 邻接回归 | 58 passed、0 skipped |
| 严格清单 | 64/64 MATCH |
| 时间核 | 0 新违规，23 条既有登记 |

具名变异：omit_tolerated_focal_loss、force_exact_report。邻接跳过：无；跳过不计 passed。
camera_bridge.py 不在导出清单登记面，严格清单仍逐项核验；未新增模块或修改既有测试、CI、容差和依赖。

## 复跑与边界

```text
python -X utf8 -m pytest tests/test_camera_tolerated_loss.py -q
python -X utf8 tools/testing/replay_save_guards.py --spec tests/fixtures/camera_tolerated_loss_mutations.json --baseline 891c28a7471a31275cc6d37a4468b62e49fe0694 --mutate --output <新证据目录>
python -X utf8 tools/verify_manifest.py --strict --quiet
```

实际调用 Camera3D.project，验证 0.5 px 残差与逆转换 fy；使用合成内参和世界点，未接入物理相机、DSH 或生产数据。不是远程安全边界或 P0。
全状态 705 Issues / 471 PRs 已核对。#1194/#1195 是既有相机能力；#1228 修复的是畸变五键和标定残差闸，本件只修焦距容差内的损失报告，不重复申报那两项。
本地 Linux 重放与变异未跑；远端 CI 单列。未运行压力测试或真实设备验收。
本批临时数据库和隔离副本已清理；公开日志仅脱敏路径并归一换行/行尾空白，本地保留原始结果。
一个独立 PR 对应一个 intake，human review pending。
