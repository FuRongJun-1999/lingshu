# 世界反馈与重建预算

本说明收束 issue #1/#2 回复中的两个使用契约。

## 没有证据的轮次

`UnifiedWorldModel.verify()` 和七层 L5 的 `total` 只数可以比对实际位置的预测。
`total == 0` 时 `hit_rate` 返回 `None`（JSON `null`），含义是没有验证读数。
有证据但全部未命中时，读数仍是 `0.0`。

`verify_run`、`SimulationLoop`、`SevenLayerLoop` 只把有验证读数的轮次纳入
原来的按轮平均；轮次 WAL/审计仍保留，包括 `total=0` 的轮次。
推演 WAL 的 episode 带 `hits/total/pending`，可区分失分、遮蔽和缺少预测。
推演汇总给出 `verified_ticks` 与 `unverified_ticks`，七层报告的 L5 同样给出计数。
尚无任何有效读数的汇总及无可比较分段的提升量也返回 `None`。

兼容性变化：调用者不能再把这些可能缺失的读数无条件当作 float 运算。
有读数的数值、按轮平均方式与假设生长/回退策略保持原契约。
认知载荷中的“最近已验证轮次命中率”明确是最近一次有效读数；完全没有有效读数
时显示“暂无验证读数”，并附有效轮次计数。

## 重建世界的预算

```python
world = load_world_from_memory(store, limit=300)
world = load_world_from_brain(agent, limit=300)
```

两个入口均默认 `limit=200`，可按场景规模指定读取预算。
脑侧预算通过 `cg(op=read, k=...)` 传输，适配器随后过滤标签；它不是最终实体数，
也不保证完整枚举世界。返回较少实体可能来自召回、标签过滤或同名实体合并。
脑侧状态投影当前还有独立的 500 条预算，因此增大候选预算也不能承诺全量状态恢复。

完整世界快照需要脑端提供稳定枚举/cursor/快照语义，之后才能做有保证的分页。
当前读取面没有这些信息，不自动推断截断，也不因候选数刚好等于预算而报错。

`BrainStore.get_nodes_by_tag` 保留原有最小值 1，`limit=0` 不表示空查询。
本次将此行为明确写入接口说明；零预算是否代表禁用查询，留待 store 契约统一。
MCP 参数按操作翻译：read 的候选数是 `k`，state_chain 的条数是 `limit`。
脑端 read 的预算式读取还使用 `limit` 表示近期事件条数，不能全局改名或直接加别名。
