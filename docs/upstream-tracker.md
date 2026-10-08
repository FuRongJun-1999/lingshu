# 上游同步追踪 · upstream-tracker

> 本文件是 `.github/workflows/upstream-tracker.yml` 的**输入面**：每日 cron 解析
> 下方哨兵块里登记的「上游标记」，用 `gh api` 查它们是否**已合并 / 已落地**，
> 命中即在本仓开或评一条 issue（标签 `upstream-sync`）提醒同步。
>
> 本仓是**聚合仓**（灵 × 脑 × 身），架构章程是**单向引入**：上游为真源、本仓为聚合体。
> 上游一改，本仓的适配面（如 `lingshu/world/brain_store.py`）就可能陈化——
> 本表用于把「该同步了」这件事从**人的记性**变成**机械提醒**。

## 上游清单（真实）

| 面 | 上游仓 | 与本仓的关系 | 本仓关注点 |
|---|---|---|---|
| **脑** | [`FuRongJun-1999/dsh-memory`](https://github.com/FuRongJun-1999/dsh-memory) | 认知图引擎（可安装的包依赖，不 fork） | `lingshu/world/brain_store.py` 的 MCP 对接面（`cg(op=read)` 召回/`limit→k`/`spatial.coords3d`/`stg` 状态槽位）；`tests/test_brain_store.py` 复跑 |
| **灵** | [`FuRongJun-1999/CommonTrustProtocol`](https://github.com/FuRongJun-1999/CommonTrustProtocol) | 智能论 · 共同信任协议（理论层真源） | `docs/theory/` 的引用与术语一致性 |
| **身** | 私有（未公开） | 身体侧件由本仓**单向引入**并落 `docs/intake/` | 引入批次与 `docs/intake/**` 登记一致性 |

## 登记格式（机器可读）

在下方哨兵块之间，**每行一条**，用 `|` 分隔四个字段：

```
<上游仓 owner/repo> | <kind: pr|commit> | <编号或提交 SHA> | <本地关注点>
```

- **注释行**（行首 `#`，允许前导空白）与空行被解析器跳过——示例与说明一律写成注释。
- `kind=pr` 时第三字段为 PR 编号（纯数字）；`kind=commit` 时为 SHA（≥7 位十六进制）。
- 第四字段为**本地关注点**：命中后写进 issue 正文，供同步时对照。
- 解析失败的行会作为 `FORMAT-ERR` 打印在 workflow 日志里（不静默），但不阻断。

### 哨兵块

```text
<!-- UPSTREAM-TRACKER:BEGIN -->
# —— 示例条目（全部以 # 注释，解析器跳过；**未编造任何真实 PR 号 / SHA**）——
# FuRongJun-1999/dsh-memory | pr | <PR编号> | 脑端 brain_store 对接面（M1/M2）——合并后同步适配器并复跑 tests/test_brain_store.py
# FuRongJun-1999/dsh-memory | commit | <提交SHA> | 脑端 cg(op=read) 召回面 / limit→k 语义变更——影响 scene 重建条数契约
# FuRongJun-1999/CommonTrustProtocol | pr | <PR编号> | 智能论协议变更——同步 docs/theory/ 引用与术语
#
# 启用方式：把上面某行**去注释**并填入真实编号/SHA（须先与维护者确认），保存提交即可。
# 去注释后本 workflow 下一轮 cron 即会查询并提醒。
<!-- UPSTREAM-TRACKER:END -->
```

> 注：`<PR编号>` / `<提交SHA>` 是**占位符**——本仓当前**没有**已确认的待同步上游标记，
> 故示例一律置为注释且不填具体编号（避免记录未经核实的编号导致误报）。

## 运行与权限

- **权限最小化**：workflow 只申请 `contents: read`（读本文件）+ `issues: write`（开/评 issue）。
- **令牌**：需 repo secret `GH_PROJECT_TOKEN`（`issues: write` 粒度）。
  **未配置即静默跳过**（打印一条 `::notice::`，不红）——沿用 `billion-context-dsh` 的缺省即跳过先例。
- **手动试跑**：`workflow_dispatch` 带 `dry_run=true` 只打印将执行的动作，不开/评 issue。
- **幂等**：同一条上游标记若已有 open 的 `upstream-sync` issue（以 issue 正文里的
  `<!-- upstream-ref: … -->` 标记识别），则**追加评论**而非重复开条。

## 维护

- 上游合并且本仓同步完成后：把该行移入「已同步（历史）」小节并去注释化、标注同步批次提交号。
- 新增上游：在「上游清单」表与哨兵块各补一条；若引入新的顶层上游面，同步更新
  `docs/architecture.md`。

## 已同步（历史）

<!-- 同步完成后把原条从此处记录；本仓尚无记录。 -->
（暂无）
