# 第 8 项：场景整合运行与验收

第 8 项的工程交付是 **共享场景消费与候选推演 v0.1**：一次选取/召回已有图记录，
角色、世界、故事、历史、因果五个任务共用来源、四栏条件、资格和作用域。
身体侧使用现有 core/WorldModel；脑侧沿用既有 MCP。此入口不依赖前端或第 9 项插件。
设计取舍及理论核见 [整合方案](plans/场景推演整合_v0.1.md)。

## 安装和第一条命令

在本 PR 检出目录，用 Python 3.10 以上（本地验证 3.12）安装：

```sh
python -m pip install -e .
python -X utf8 -m lingshu.world.scenario_cli demo --mode all
```

取得交付包时，也可以在解压目录安装其中的 wheel：

```sh
python -m pip install --no-deps ./lingshu-0.0.1-py3-none-any.whl
python -X utf8 -m lingshu.world.scenario_cli demo --mode all
```

这条命令只在内存构造合成灯塔场景：Alice 的身份、子知识、memory/anchors/values、
指定会话、8 条因果边、实体现值，以及供隔离验收的另一角色/另一会话/合成史料。
没有脑端前置，没有用户数据，也不创建记忆库文件。无需 numpy/Pillow 即可输出五种 JSON 视图。

可单独选择 `--mode roleplay|world|story|history|causal`。因果视图默认深度 16、路径数 32、
展开数 4096，可用 `--max-depth` / `--max-paths` / `--max-steps` 调整；截断会留缺口。

## 五种输出怎样使用

| mode | 可用输出 | 使用方式 |
|---|---|---|
| roleplay | identity/lore/memory/anchors/values/turns | 上层角色调用消费同角色、同会话材料；缺角色定义留缺口 |
| world | entities，含位置、状态和 node_id | 构造现有 WorldModel，来源仍由 facts 按 node_id 查询 |
| story | outline/background/cyclic | 用因果/时序约束排列素材；静态设定保留为背景 |
| history | evidence_timeline/unresolved/time_basis | 分开合格现实来源证据和待核材料；虚构内容不进入 |
| causal | paths/truncated/cyclic/steps | 保留路径上的每条边、四栏条件与来源；hypothesis 明示待核 |

顶层格式为 `{"schema_version": 1, "fixture": true/false, "views": {...}}`。
每个视图有 `state`、`domain`、`domains`、`scope`、`facts`、`edges`、`gaps` 和 `payload`。
`domain` 是当前材料的 real/role/mixed 分类；混合故事的 `fiction` 为 null，
每个提纲项另附 domain，不把现实材料整体包装成角色世界。

**演示全部为 DEFER 是预期行为**：fixture 和本地未裁决材料不自动升格成事实。
ACCEPT 仅表示本层组装检查通过，不验证新事实。REJECT 不进入材料；
缺来源/条件、正文截断、资格待定、召回或推演超限都会留缺口。
结果可成功返回 DEFER/BLINDSPOT/REJECT；进程退出码与资格不是同一维度：
0 = 调用成功，2 = 命令参数无效，1 = 数据、依赖或传输错误。错误写 stderr，成功 JSON 用 UTF-8 写 stdout。

## 消费已有本地图

```sh
python -X utf8 -m lingshu.world.scenario_cli local --db ./scenario.db --node-id role-node --node-id lore-node --node-id action-node --node-id outcome-node --role-id alice --session-id conversation-1 --mode all --start-id action-node --end-id outcome-node
```

必须显式选取节点，不按标签猜选取集。输入须为既有 LayeredStore SQLite schema，
不是脑端 Markdown 根，也不是任意 SQLite 数据库。
CLI 用 SQLite `mode=ro` + query_only + 同一读事务取得快照；不运行 LayeredStore 构造器，
不建表、不迁移、不更新访问计数。文件不存在或 schema 不兼容时直接失败。
底层出边读取仍沿用既有全出边接口；候选边数预算不等于 SQL I/O 硬限额。

库已有角色/会话字段时需声明对应作用域；不声明就不会猜测可见性。
本地节点没有脑端资格裁决，原值保留 BLINDSPOT，仍可消费为明确待核的材料。

## 消费脑端 MCP

提供包含 `md_cg/` 的脑包目录和已存在的目标根；安装方式由脑仓维护，灵枢不复制脑代码。

```sh
python -X utf8 -m lingshu.world.scenario_cli brain --root ./isolated-brain --brain-path ./dsh-memory --query "灯塔" --k 64 --role-id alice --session-id conversation-1 --mode all --start-id action-node --end-id outcome-node
```

如脑端需要身份配置，可追加 `--brain-env-file ./local-identity.json`。
该库外 JSON 沿用适配器 extra_env 口径，把 `MDCG_*` 键映射为字符串或 null，例如
令牌/租户/登记根等由调用者按脑端实际部署提供。本入口不打印身份值，不继承宿主 MDCG_*；
`MDCG_ROOT`、`MDCG_STG_STATE` 是保留键，不能经身份文件覆盖命令的根或关闭所需状态读取。
未提供文件时沿用脑端匿名身份，能否读取由脑端授权决定。不要把身份文件提交到仓库。
`--brain-python` 指定脑端解释器，`--timeout` 控制每次 MCP 请求等待秒数（默认 30）。

一次 `cg(op=read)` 召回建立快照，五个视图不各自召回五遍。
现实空间实体状态继续使用现有 `stg(op=state_chain)` 投影，并保留状态事件证据。
只在单实体名、单现值且非角色知识时投影；歧义留缺口，角色同名实体不继承宿主状态。
本入口不调用 remember、link 或 state_event。脑侧读取仍可能维护自己的统计/索引，
因此不宣称 MCP 整体零落盘。消费过滤也不替代脑侧身份/租户/密级授权。

`local`/`brain` 的 all/causal 必须提供起点和终点。其余单任务不需要因果端点。
选取超过 `--max-nodes` 时报错，边超过 `--max-edges` 时截断并留缺口；默认 256/1024。

## 世界模型呈现

源代码检出目录安装 `.[full]`，或对交付 wheel 安装 full extra：

```sh
python -m pip install -e ".[full]"
python -X utf8 -m lingshu.world.scenario_cli demo --mode world --render ./lighthouse.png
```

500 × 500 PNG 使用既有 WorldModel 渲染。也可在 all 模式加 render。
未知状态在 JSON 保留 null 和缺口，neutral 只是呈现占位，图像不提供事实验证资格。
输出只写显式给出的 PNG，不能覆盖输入数据库或写入脑根。

## Python 接入

```python
from lingshu.world.scenario import ScenarioContext

context = ScenarioContext.from_store(
    engine.store, selected_ids, role_id="alice", session_id="conversation-1")
material = context.view("roleplay")
outline = context.view("story")
paths = context.view("causal", start_id=action_id, end_id=outcome_id)
model = context.to_world_model()  # 此时才需要 full extra
```

已有 MCPClient 的调用方使用 `ScenarioContext.from_brain(client, query, k=64, ...)`；
由调用方负责连接、身份和关闭。每次 view 都返回独立 JSON 副本，改变返回值不会写回原图。

## 交付验收与范围

```sh
python -m pip install -e ".[full,dev]"
python -X utf8 -m pytest tests/test_scenario_context.py tests/test_scenario_cli.py tests/test_memory_world_scenarios.py -q
```

设置 `MDCG_BRAIN_PYTHONPATH` 为含 md_cg/ 的脑包目录后，3 条真实 MCP 验收会在
pytest 隔离根写入合成前置，再测试角色/现实隔离、状态事件→场景→WorldModel、
以及 CLI 显式根与五视图的完整调用。缺前置则明确 SKIP；正式交付证据必须标清是否真实跑过。
其他验收覆盖持久化重开、8 深度、循环/平行路径/预算、源与条件、输出复制、
多实体名歧义、混合域标记、只读库摘要/计数不变、错误出口和无第三方依赖运行。

本件完成的是第 8 项的**后端整合件 v0.1**。角色回复/故事正文的生成后端、
历史年代抽取、完整考古研究和在役权限/数据验收不在这个整合件内，不能据此声称全部 AI 应用已完工。
上层消费时须保留资格、domain 和 gaps，不把提纲/待核路径变成已验证事实。
接口分位与可见性取舍仍交项目理论核，默认调用链不变；项目最终确认后再合并。
第 9 项插件独立安装和前端均另件处理。
