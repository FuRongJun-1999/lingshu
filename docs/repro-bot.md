# repro-bot —— issue 复现机器人

> **状态**：工作区草稿（未落仓）。相关件：`.github/workflows/repro-bot.yml`（外壳）、
> `scripts/extract_and_run_issue_repro.py`（执行核）。

一句话：外部报告人附了复现脚本的 issue，维护者打一个 `run-repro` 标签，
机器人在 GitHub 托管 runner 上自动执行该脚本并把**原始读数**回贴到评论。
**机器人只贴读数，不作结论**——结论仍走人工/模型侧的既有流程。

## 用法（维护者视角）

1. 确认目标 issue 的正文里有 ` ```python ` 围栏代码块（第一个 python 块会被执行；
   抽取规则见下）。
2. 给 issue 添加 `run-repro` 标签。**只有允许名单内的账号**（workflow 内一处常量，
   当前仅 `FuRongJun-1999`）打这个标签才会触发。
3. 等 1–2 分钟：issue 会收到一条「自动复现读数（repro-bot）」评论；机器人同时
   打上 `repro-done`（执行核退出码 0）或 `repro-failed`（其余情况），并**
   移除 `run-repro`**。
4. 要重跑：再打一次 `run-repro`（标签已被移除，直接重打即可重新触发）。

## 判据（对齐 enforcement 三件）

- **触发**：`issues` 事件的 `labeled`；**门**＝标签名 `run-repro` 且打标签者
  在允许名单内（防滥用）。
- **权限**：`issues: write` + `contents: read`（最小面）。
- **执行**：checkout（main、`persist-credentials: false`）→ Python 3.12 →
  `pip install -e ".[full,dev]"` → clone 脑端公开仓（失败显式 SKIP）→
  `gh api` 取 issue 正文落盘 → 抽取 → 写入独立临时目录 `repro_issue_<n>.py` →
  限时执行（600 s）→ 截断（stdout/stderr 各 20 KiB）→ 固定模板回评 → 打状态标签。
- **模板**（末行固定声明，勿改）：
  `来源 / 抽取 / 环境 / 执行 / 退出码 → 原始输出（stdout）→ 原始错误（stderr）→`
  `「本评论仅贴读数，不作结论。」`

## 抽取规则（CommonMark 围栏口径）

- 开启：行首（允许前导空白）≥3 个反引号或波浪线；info string = 同行其后剩余文本。
- 闭合：同行定界符字符、长度不短于开启者、且该行除定界符外无其它字符
  （围栏内写的 ` ```js ` 只算代码内容）。
- **只取第一个 info 首词为 python|py|py3（大小写不敏感）的围栏块**；
  正文含多个 python 块时也只用第一个（报告的「抽取」行会给出全部块的语言分布）。
- 无 python 块 ⇒ **不执行任何东西**，显式失败并回评说明原因。

## 边界（当前口径下的已知行为）

- 多代码块：只取第一个 python 块，其余块不执行。
- 脚本不自包含（依赖报告人环境里"由运行方提供"的变量）：原样执行会报
  `NameError` 之类的错误——**读数照贴**（机器人不替报告人补跑环境，这是有意为之；
  报告人应在后续交互中补一个自包含脚本，再次打标签）。
- 无代码块 / 无 python 块：不执行，回评说明，打 `repro-failed`。
- 超时：600 s 未结束即终止，读数照贴（退出码行写「超时终止」）。
- 脑端 clone 失败：显式 `::warning::` 并**不传** `MDCG_BRAIN_PYTHONPATH`——
  依赖脑端的脚本会自行报错，读数照贴（不静默、不假绿）。

## 接受的残余风险

本功能的本性是**在托管 runner 上执行不可信代码**。缓解与残余如实列明：

**已缓解**：
- **无 secret**：job 只持有被收窄权限的 `github.token`（issues: write + contents: read），
  且该令牌不进被测脚本的进程环境（执行步骤不挂 GH_TOKEN；执行核对子进程 env 清洗：
  剔除 `GITHUB_*/GH_*/ACTIONS_*/RUNNER_*/INPUT_*` 与含 `TOKEN|SECRET|PASSWORD|CREDENTIAL`
  的键）；checkout 不落凭据（`persist-credentials: false`）。
- **限时**：脚本内 600 s 超时终止 + job 30 分钟上限。
- **一次性宿主**：GitHub 托管 runner 用完即弃；**绝不用自托管 runner**（同 gate.yml 口径）。
- **输出截断**：stdout/stderr 各 20 KiB + 动态围栏（防输出撑破 markdown）。
- **不改仓**：脚本只写独立临时目录；不动检出仓内任何文件。

**未缓解而接受的**：
- 被测脚本仍可自由外联、耗资源、探测宿主（一次性 runner 的资源上限内）。
- 超时杀死的是直接子进程；其孙/后台进程在 runner 销毁前可能短暂残留（同上）。
- 报告原样贴出被测脚本的输出——其中可能含被刻意构造的误导文本；**读数的解释权
  始终在人**，repro-bot 不背书其内容（模板末行固定声明即此边界）。
- 执行步骤与回评步骤同 job 同 runner：理论上被测代码可尝试伪造
  `$GITHUB_OUTPUT`/`$GITHUB_ENV`（路径含随机后缀，且相关环境变量已从其 env 剔除；
  最坏后果=状态标签打反，报告内容仍由执行核生成）——记为接受项。

## 失败排查

- **没触发**：标签名不是 `run-repro` ／ 打标签者不在允许名单 ／ workflow 未在默认分支。
- **打了 `repro-failed`**：看评论里的退出码与原始输出；读数本身可能就是"预期失败"
  （如脚本依赖运行方预置）。
- **评论缺失且 job 红**：看 Actions 日志（多半是执行核自身异常或 `gh api` 失败）。
