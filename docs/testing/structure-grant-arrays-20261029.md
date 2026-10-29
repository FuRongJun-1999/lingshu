# 服务启动授权名单校验

Refs lingshu #1107 的 grant 配置面；该 Issue 的业务参数、游标两面未由本件修复，故不自动关闭整条 Issue。

基线 `7e2de720b39d220fc1553cf114e3a28b4cdf274f`；固定功能验证源码 `a2a2d3784a89f089bab02706e6997d4cb0b1cfe3`。
配置 `instances: "instance-1"` 曾被 frozenset 拆成字符集，报告 instance_id="i" 会被当成授权执行结果写入。
现在 capabilities/instances 必须是非空字符串组成的 JSON 数组；类型错误、空项在 TLS/数据库初始化前报 WriteError。
instances 缺省仍为空；不修剪/拆分名称，不改 capability 词表、TLS 或数据库权限。

建议 P1/medium/B：误配置会改变授权集合，但配置只由受信部署者控制，不能据此宣称 worker 远程提权或 P0。

## 证据

[原始日志](structure-grant-arrays-20261029/)；Windows / Python 3.12.10 / SQLite 3.49.1。

| 检查 | 结果 |
| --- | --- |
| 基线同一守卫 | 10 failed、3 passed；0 errors、0 skipped |
| 修复后 | 13 passed；0 errors、0 skipped |
| 定点变异 | 3/3，各恰好一个指定断言失败 |
| 邻接回归 | 18 passed、0 skipped |
| 严格清单 | 64/64 MATCH |
| 时间核 | 0 新违规，23 条既有登记 |

变异：split_instance_string、split_capability_string、allow_empty_names。
新守卫调用真实 load_service、数据库与 dispatch，仅配置测试中的 TLS/ACL 装配使用替身；不把它称为 TLS 认证证明。
邻接 test_structure_writer.py 的临时 CA/客户端证书、真实 mTLS 子进程与授权边界用例另在本批实际通过；OpenSSL CLI 只在任务进程环境中使用，未安装工具或更改全局 PATH。
没有改旧守卫、业务阈值或 CI；没有新增模块，清单无需改动。

```text
python -X utf8 -m pytest tests/test_structure_grant_arrays.py tests/test_structure_writer.py -q
python -X utf8 tools/testing/replay_save_guards.py --spec tests/fixtures/structure_grant_arrays_mutations.json --baseline 7e2de720b39d220fc1553cf114e3a28b4cdf274f --mutate --output <新证据目录>
python -X utf8 tools/verify_manifest.py --strict --quiet
```

严格重现应检出固定功能验证提交。真实 mTLS 用例需要 OpenSSL CLI；本机通过进程级 PATH 使用已安装的 Git OpenSSL。
目标仓 697 Issue / 381 PR 全状态核对，未见 #1107 对应 PR；本件直接引用已有 Issue，不重复报同一个缺陷。
本地 Linux 重放/变异、多 OS 账户隔离部署、DSH、真实设备、生产数据未测；远端 CI 另记。
本批临时证书、数据库、测试服务与副本已清理；公开日志只脱敏路径并归一换行/行尾空白，本地保存原始结果。
一个 PR 对应一个 intake，human review pending。
