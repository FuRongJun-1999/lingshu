# 导出目标误指源库的保护验证

基线：`40fceb69e79d0f481455f05ea8581b6f6f5e5816`。
目标为 lingshu 身体侧 SQLite 存储；不改 dsh-memory 大脑区。

## 复现与修复

`export_all` 的原子替换、`export_trajectory` 的直写都接受调用方指定的路径，
没有排除当前连接正在使用的数据库。`export_structured_trajectories` 委托轨迹导出，
也会命中同一问题。目标在 `LINGSHU_EXPORT_OUT_ROOT` 内仍可破坏源库。

本件只排除当前 SQLite 连接 `PRAGMA database_list` 报告的数据库及对应
`-wal` / `-shm` / `-journal` 文件；用 realpath 和 samefile 识别软硬链接。
内存库无源文件，仍可导出。普通 JSON 备份仍可覆盖，#698 的返回值契约保持；
#746 的可选根闸、#330 的原子写与 #498 的权限/符号链接处理继续沿用。

## 原始读数

探针先写入合成节点并 `PRAGMA wal_checkpoint(TRUNCATE)`，再把**这份临时库**
误传为导出路径，关闭引擎并用独立连接检查文件头、完整性和节点是否存活。

| 平台/入口 | 基线 | 修复 |
| --- | --- | --- |
| Windows / export_all | OS PermissionError，源库存活；没有应用层源库拒绝 | ValueError 源库保护，源库存活 |
| Windows / export_trajectory | 返回成功，重开 `file is not a database`，合成节点不可读 | ValueError 源库保护，源库存活 |
| Linux / export_all | 返回成功，重开 `file is not a database`，合成节点不可读 | ValueError 源库保护，源库存活 |
| Linux / export_trajectory | 返回成功，重开 `file is not a database`，合成节点不可读 | ValueError 源库保护，源库存活 |

原始字段在 `windows-summary.json` / `linux-summary.json` 的 `readings` 中，
逐项含 `returned_success`、`before_header_hex`、`after_header_hex`、`reopen_error`、
`fact_survived`。Windows 原版 export_all 并未复现数据破坏，不把其 OS 拒绝算作破坏成功。
文件头以 16 字节的 hex 无损表示；OS 异常保留真实类型、消息与 errno/winerror，
不公开本机临时文件名。此表示亦避免 JSON 中的 CRLF 转义被扫描器误判为 UNC 路径。

## 复跑与守卫

```text
python -X utf8 -m pytest tests/test_export_source_path_guard.py -q
python -X utf8 tools/testing/check_export_source_path_guard.py --baseline 40fceb69e79d0f481455f05ea8581b6f6f5e5816 --mutate
```

第二条在临时包副本中运行当前实现、git show 固定基线、各条具名变异；工作树不改写。
输出 stdout 日志、JUnit、源码/守卫 SHA256。超时、收集错误、意外跳过都不算变异命中。

| 平台 | 基线 | 修复 | 定点变异 |
| --- | --- | --- | --- |
| Windows / Python 3.12.10 | 14 failed / 3 passed | 17 passed | 5/5 指定断言失败 |
| Linux / Python 3.13.12 | 14 failed / 3 passed | 17 passed | 5/5 指定断言失败 |

17 条覆盖三个公开入口、三类辅助文件（含尚不存在的 journal）、配置根内源库、
硬链接/符号链接、cwd 改变、附加数据库，以及普通备份覆盖/内存库导出正对照。

| 变异 | 指定用例/断言 |
| --- | --- |
| `export_all_not_wired` | G1 `[all]`：source path was not rejected before filesystem write |
| `trajectory_not_wired` | G1 `[trajectory]`：同上 |
| `sidecars_not_protected` | G2 `[-journal-all]`：同上 |
| `hardlink_identity_not_checked` | G4 硬链接：同上 |
| `attached_database_not_enumerated` | G7 附加数据库：同上 |

扩大回归（上述新守卫加 #330、#383、#498、#698、#746、#57 及登记守卫）：
Windows **108 passed / 1 failed / 6 skipped**；Linux **114 passed / 1 skipped**。
两个平台均计入新守卫，不能与 17 passed 重复相加。
Linux 跳过的是 #746 仅适用于 Windows 的大小写不敏感用例；Windows 6 条跳过是
#498 的 A5/A6/B5/B6/C5/C6（POSIX 权限位/符号链接，原有静态平台标记）。
Windows 的 1 failed 详见下方既有平台探针记录。

另跑 `tests/test_core_export_all.py` 为 **26/26 PASS**（脚本式）；
`tools/verify_manifest.py --strict --quiet` 为 **64 MATCH，0 MISMATCH，exit 0**。
扩大回归原始输出为同目录 `windows-regression.txt` / `linux-regression.txt`。
WSL 复用既有环境：SQLite 3.46.1、pytest 9.0.3、numpy 2.3.5、Pillow 12.2.0；
Pillow 低于项目声明下界，该次读数仅为补充验证，声明依赖矩阵以 PR CI 为准。

## 诚实边界

- 已验：上述生产入口、源库重开与合成节点存活、基线对照、指定断言变异、普通覆盖契约。
- Windows 扩大回归有一条**原版同样失败**的既有平台探针：
  `test_issue498_atomic_write_perm_symlink_guard.py::test_p1_符号链接量具能力探测`
  假设 Windows 不能创建符号链接；本机实际可以。未改该用例或关闭平台能力。
- 未验：真实 DSH 宿主、生产数据、掉电；本件验证独立的身体侧引擎接口。
- 不确定：部署中误填路径的发生频率；优先级、难度和影响档由维护者裁定。
- 不做：保护不属于当前连接的任意其他文件；改变默认根/备份覆盖策略；抵御有权限
  并发改名或替换目录的本机进程（本件不是文件系统沙箱，无 TOCTOU 消除承诺）。
- 人工技术审查与理论裁定仍待维护者；本机结果与全仓 PR 当前 head CI 分开记录。

查重：对照 lingshu #330、#383、#498、#698、#746 与开放 PR；前几项分别讨论
备份原子性、快照一致性、权限/软链、覆盖上报和可选根闸，未见同件源库排除修复。
