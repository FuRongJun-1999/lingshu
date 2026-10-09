# tests/ · 当前运行方式

完整门禁由 [.github/workflows/gate.yml](../.github/workflows/gate.yml) 定义：

1. 安装 `pip install -e ".[full,dev]"`。
2. 逐一运行 workflow 的 `SCRIPT_TESTS`（`python -X utf8 <文件>`），任一非零退出码即失败。
3. 使用 `python -X utf8 -m pytest tests/ -v`，并通过 `--ignore` 排除已在前一步独立执行的脚本文件。所有其他 pytest 用例都参加检查。

部分脚本在模块导入时执行或退出，直接对目录运行 pytest 不能替代上述完整门禁。脚本列表统一以 workflow 为准，不在这里维护另一份名单。

脑适配器测试需要公开 `dsh-memory` 代码目录，由 `MDCG_BRAIN_PYTHONPATH` 指定。CI 会 clone 公开仓并实际运行；只有公开仓不可达时才显示具体 SKIP 原因。它不使用真实记忆数据库。

结构服务业务回归运行：

```text
python -X utf8 -m pytest tests/test_structure_events.py tests/test_structure_writer.py -q
```

TLS 子进程测试使用 OpenSSL CLI 在临时目录生成证书与私钥，文件不会进入仓库。GitHub Ubuntu runner 提供该工具；本地需先安装 OpenSSL CLI。Windows 的临时 ACL 调整只作用于测试自身目录。

两条 `test_hex_text` 历史 known-fail 现在已通过原断言，gate 已移除对应 deselect；没有添加 xfail。旧 intake 中的 6/16、7/12 是历史导出读数，不能用来描述当前实现。

SciPy 属可选后端，未安装时其现有专属测试显式 skip；不要将 skip 记为运行通过。核心和 NumPy 路径正常运行。
