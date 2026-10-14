# repro-bot —— issue 复现机器人

> **状态**：已随仓落仓（引入提交 `8a2de8f`「ci: issue 复现机器人落仓」——本件与 workflow、执行核同批落仓）。相关件：`.github/workflows/repro-bot.yml`（外壳）、
> `scripts/extract_and_run_issue_repro.py`（执行核）。

一句话：外部报告人附了复现脚本的 issue，维护者打一个 `run-repro` 标签，
机器人在 GitHub 托管 runner 上自动执行该脚本并把**原始读数**回贴到评论。
**机器人只贴读数，不作结论**——结论仍走人工/模型侧的既有流程。

## 用法（维护者视角）

1. 确认目标 issue 的正文里有 ` ```python ` 围栏代码块（第一个 python 块会被执行；
   抽取规则见下）。**执行的是打标签那一刻的正文快照**——打完标签后再改正文不会
   改变本次执行（见下「正文快照」）。
2. 给 issue 添加 `run-repro` 标签。**只有允许名单内的账号**（workflow 内一处常量，
   当前仅 `FuRongJun-1999`）打这个标签才会触发。
3. 等 1–2 分钟：issue 会收到一条「自动复现读数（repro-bot）」评论；机器人同时
   打上 `repro-done`（执行核退出码 0）或 `repro-failed`（其余情况），并**
   移除 `run-repro`**。
4. 要重跑：再打一次 `run-repro`（标签已被移除，直接重打即可重新触发）。

## 判据

- **触发**：`issues` 事件的 `labeled`；**门**＝标签名 `run-repro` 且打标签者
  在允许名单内（防滥用）。
- **权限**：`issues: write` + `contents: read`（最小面）；且 `issues: write` 只出现在
  持令牌的 `report` job。
- **三 job 分离**（#241）：`prep`（唯一取正文，取自**打标签时刻的事件载荷快照**，
  **不持令牌**）→ `exec`（**无** issue 令牌，跑不可信代码）→ `report`（唯一回评/打标签，
  带令牌）；job 间只经 artifact 交换正文与报告。
- **正文快照**（#144①）：执行的正文＝**打标签那一刻**的事件载荷（`$GITHUB_EVENT_PATH`
  的 `.issue.body`），**不在运行时重取**——维护者审的与机器人跑的必须是同一份；
  打标签后报告人再改正文不影响本次执行（要跑新版正文须重打 `run-repro`）。
- **执行**（`exec` job）：checkout（main、`persist-credentials: false`）→ Python 3.12 →
  `pip install -e ".[full,dev]"` → clone 脑端公开仓（失败显式 SKIP）→
  取回正文 artifact → 抽取 → 写入独立临时目录 `repro_issue_<n>.py` →
  限时执行（600 s）→ 截断（stdout/stderr 各 20 KiB）→ 上传报告 artifact。
- **回评**（`report` job）：取回报告 artifact → 固定模板回评 → 打状态标签。
- **模板**（末行固定声明，勿改）：
  `来源 / 抽取 / 环境 / 执行 / 退出码 → 原始输出（stdout）→ 原始错误（stderr）→`
  `「本评论仅贴读数，不作结论。」`
  报告里一切来自正文的文本（如块语言分布）一律包成**行内代码**入报告，防注入（#363）。

## 抽取规则（CommonMark 围栏口径）

- 开启：行首（允许前导空白）≥3 个反引号或波浪线；info string = 同行其后剩余文本。
- 闭合：同行定界符字符、长度不短于开启者、且该行除定界符外无其它字符
  （围栏内写的 ` ```js ` 只算代码内容）。
- **HTML 注释（`<!-- … -->`）内的围栏不算块**（#235）——注释在网页上不可见，其中
  的代码不得被抽取执行；检测按掩掉注释后的可见文本判，代码仍取原文行。
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
- **无 secret**：`github.token` 被收窄到 issues: write + contents: read，且**只出现在
  `report` job**——`prep` 不调 API（只读本机事件载荷文件）故不持令牌；跑不可信代码的
  `exec` job 任一步骤都不挂 `GH_TOKEN`；
  执行核对被测子进程 env 另做清洗（剔除 `GITHUB_*/GH_*/ACTIONS_*/RUNNER_*/INPUT_*`
  与含 `TOKEN|SECRET|PASSWORD|CREDENTIAL` 的键，注意这只是封子进程自己的 env、
  不是 job 隔离）；checkout 不落凭据（`persist-credentials: false`）。
- **正文快照**（#144①）：正文取自打标签时刻的事件载荷，不运行时重取 ⇒ 消除
  「审的正文 vs 跑的正文」之间的 TOCTOU 窗口（旧实现 `gh api … --jq .body` 在
  checkout/装依赖/clone 之后重取，报告人可在窗口内换掉被执行的脚本）。
- **限时**：脚本内 600 s 超时终止 + job 30 分钟上限。
- **一次性宿主**：GitHub 托管 runner 用完即弃；**绝不用自托管 runner**（同 gate.yml 口径）。
- **输出截断**：stdout/stderr 各 20 KiB + 动态围栏（防输出撑破 markdown）。
- **不改仓**：脚本只写独立临时目录；不动检出仓内任何文件。

**未缓解而接受的**：
- 被测脚本仍可自由外联、耗资源、探测宿主（一次性 runner 的资源上限内）。
- 超时杀死的是直接子进程；其孙/后台进程在 runner 销毁前可能短暂残留（同上）。
- 报告原样贴出被测脚本的输出——其中可能含被刻意构造的误导文本；**读数的解释权
  始终在人**，repro-bot 不背书其内容（模板末行固定声明即此边界）。
- **执行与回评已分属不同 job / 不同 runner**（#241）：跑不可信代码的 `exec` job
  不持有 `issues: write`、任一步骤不出现 `GH_TOKEN`；正文经 artifact 进、报告经
  artifact 出，回评/打标签只在带令牌的 `report` job。故被测代码写
  `$GITHUB_ENV`/`$GITHUB_PATH` 或残留孙进程**够不到**带令牌的那台 runner。
  （注意：执行核 `_child_env` 的 env 清洗只封被测子进程**自己的** env，同 uid 仍可
  读父进程 environ 把被清洗的键取回——**它不是 job 隔离**，隔离靠上面的分 job。）
  残余：报告文件在 `exec` 上生成，若被测代码在回评前改写它，改写的也只是**报告
  内容**（仍由执行核模板约束），够不到令牌面。
- **exec runner 自身的作业令牌（#144② 报告人路径(b)，未缓解）**：`exec` job 的
  checkout 仍需要一个作业令牌（`permissions` 已收窄为 `contents: read`），该令牌
  存在于 runner 内部进程（`Runner.Worker`）的内存中；被测代码与它同 uid，理论上
  可在**本步内**尝试从 runner 进程内存取回。分 job 不改变这一点（它针对的是
  「同 job 后续步骤」与「残留孙进程」两条路）。**影响上界＝一个 `contents: read`
  令牌**（exec job 无 `issues: write`），故不能借此改 issue；本项作为残余如实记录。

## 失败排查

- **没触发**：标签名不是 `run-repro` ／ 打标签者不在允许名单 ／ workflow 未在默认分支。
- **打了 `repro-failed`**：看评论里的退出码与原始输出；读数本身可能就是"预期失败"
  （如脚本依赖运行方预置）。
- **评论缺失且 job 红**：看 Actions 日志（多半是执行核自身异常或 `gh api` 失败）。
