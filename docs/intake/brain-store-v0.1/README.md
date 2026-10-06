# 世界模型 × 脑 适配器 v0.1（brain_store）

> 引入日期：2026-10-06 ｜ 来源：**本仓新件**（非上游导出）——依 0.8.0「身体×脑」对接
> 设计四裁定之一「适配器归属＝身体侧」（对端 `dsh-memory` 仓
> `docs/plans/灵枢身体×脑_世界模型对接设计_v0.1.md` §四，使用者已批准按推荐执行）。
> 状态：**M1 读向 + M2 写向最小闭环已贯通**（隔离库端到端 10/10，可复跑）。

## 一、内容

| 件 | 说明 |
|---|---|
| `lingshu/world/brain_store.py` | 脑版 store/engine 适配器：最小 MCP stdio 客户端 + `BrainStore`（`get_nodes_by_tag` 读 + 状态事件写垫片）+ `BrainEngine`（`add_perception` → `mdcg_remember`，`spatial.coords3d` 直存）+ `connect()` 与两条便捷函数 |
| `tests/test_brain_store.py` | 隔离库端到端：`ingest_scene`（零改动）→ 世界重建（`load_world_from_memory` 零改动）逐项断言；tag 过滤对照 |

## 二、对接口径（四条裁定如何落在这件里）

1. **归属**：适配层在本仓（`world/`），对端零脑改承接（其发布门禁已含组合冒烟腿）；
2. **坐标**：`spatial_coords {x,y,z}`（米）↔ 脑节点 `frontmatter.spatial.coords3d` 直存；
3. **tag 过滤**：`cg(op=read)` 取候选后在**适配器侧**按 tags 过滤（裁定三：零脑改）；
4. **状态**：身体侧状态写经 `store.conn` 垫片翻译为脑侧 `cg(op=state_event)` 记账；
   读回时状态取自**槽位投影**（`stg(op=state_chain)`，slot＝「状态」）——与脑侧
   「事件是源、槽位是投影」同构。

## 三、双清单

**CLEAN（0 命中）** —— 机械扫描（凭据/令牌、本机绝对路径、邮箱、本地 IP）：2 文件 0 命中，
记录见 `AUDIT.md`。隐私面：不含任何记忆数据、私有模块名与私有资产；连接参数一律
经环境/参数注入（仓内不写本机路径字面量，章程六）。

## 四、复跑

```bash
# 前置：MDCG_BRAIN_PYTHONPATH = 脑包所在目录（须含 md_cg/ 与仓根 utf8_boot.py）
MDCG_BRAIN_PYTHONPATH=<脑包目录> python -X utf8 tests/test_brain_store.py
# → SUMMARY 10/10；VERDICT=PASS（隔离库，临时根断言，零在役写入）
```

## 五、如实边界（MVP）

- 读候选受脑侧 `cg(op=read)` 召回面与 `limit` 约束（非全库枚举；`seed_query`
  缺省＝「场景实体」＝`ingest_scene` 内容定式词）；
- 状态事件的 `old` 恒为 None（legacy `ingest_scene` 只传新值；变迁史由事件序承担）；
- 垫片只翻译 `UPDATE nodes SET state_attributes` 一条 legacy 语句形态（其它 SQL 抛错）；
- 本件为身侧新件，未经独立复核前不外推「产品可用」——复跑命令与断言即判据。
