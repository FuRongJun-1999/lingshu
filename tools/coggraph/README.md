# coggraph · 认知图可视化管线

把纯 md 认知图导出为可交互知识图谱（cytoscape 查看器）的**七步确定性管线**。
**只读真源，不写回**——所有脚本经 `--root` / `MDCG_ROOT` 指定认知图根，不硬编码任何路径。

## 七步

| # | 脚本 | 输入 → 输出 |
|---|------|------------|
| ① | `export_coggraph.py` | 认知图根 → `graph.json` + `edge_report.md`（支持 `--layer`/`--bucket` 过滤；加 `--write` 才落盘） |
| ② | `derive_edges.py` | `graph.json` → `derived_edges.json`（推导边**必须**带 `derived=true` + 规则名，绝不与显式边混淆） |
| ③ | `session_chains.py` | `_audit.jsonl` → `session_chains.json`（会话顺序边） |
| ④ | `naming_report.py` | 标题集合 → `naming_report.json/.md`（四问测试命名审计） |
| ⑤ | `synonym_merge.py` | 中文标题 → `synonym_groups.json`（同义归并**只进待复核，不自动归并**） |
| ⑥ | `build_viewer.py` | `graph.json` + 多个 `--derived` 源 → `viewer_graph.json` + `index.html` + `serve.py` |
| ⑦ | `python serve.py 8788 <mdcg_root>` | 本地服务 → http://127.0.0.1:8788 |

## 一键：查看自身记忆

`view_memory.py` 把七步串成一条命令（产物缺省落 `./graph_view`，可随时重建）：

```bash
python view_memory.py --root <MDCG_ROOT>            # 构建查看器（产物落 ./graph_view）
python view_memory.py --root <MDCG_ROOT> --serve    # 构建并直接起本地服务（http://127.0.0.1:8788）
```

- **只读真源**：认知图根内任何文件都不会被修改；产物全部落在 `--out`。
- `--root` 缺省读环境变量 `MDCG_ROOT`；`--out` 与 `--port`（缺省 8788）可调。
- 需要单步调参时按上表逐条执行——一键脚本就是这七步的顺序封装。

## 依赖

- Python 3.10+；`export_coggraph.py` 需要 **PyYAML**（已在 `pyproject.toml` 的 `dev` extra 里
  声明，`pip install -e ".[dev]"` 即装齐；门禁 workflow 装 `.[full,dev]` ⇒ CI 同面）；其余纯标准库
  （`cytoscape.min.js` 由 build_viewer 首次运行自动下载）。

## 纪律（定位）

- 管线**不写回**认知图真源；产物（`graph.json` 等）是派生物，
  可随时按元数据里的 `source_sha` 指纹判断新鲜度并重生成；
- 本管线定位＝**发现工具**（用于人眼观察记忆形态：连通性 / 分布 / 命名）；
  **进入决策的数字须经认知图引擎（脑核）单点复核**——两套解析口径，以脑核为准。

## 来源

身体侧可视化项目（2026-09）原文照录；引入批次与双清单结论见
[../../docs/architecture.md](../../docs/architecture.md) 第七节登记表。
