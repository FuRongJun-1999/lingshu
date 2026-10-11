# 世界模型指标 v2 迁移

本 PR 变更了评估口径及无样本返回值。使用旧字段名的外部脚本需要选择指标含义、重标阈值，并处理 `None`；不应将 v1、v2 数字直接拼入同一曲线。

## 字段对应

| 接口 / 字段 | v1 | v2 / 迁移方式 |
|---|---|---|
| `eval_phase`、`evaluate`、`probe` 的 `learned_rate` | 距离落在模型自报 `bound` 内的比例 | 固定 `hit_threshold` 下的点命中；继续读取该字段需重标阈值 |
| 同接口的 `bound_coverage` | 无独立字段 | 保留旧 learned 范围覆盖的计算含义；无样本返回 `None` |
| `naive_rate` | 固定点阈值，与旧 learned 范围口径不同 | 与新 learned 使用相同点阈值；无样本返回 `None` |
| `oracle_rate` | oracle 自报范围评分 | 与 learned 相同点阈值；范围覆盖见 `oracle_coverage` |
| `gap_to_oracle` | 两个总体命中率作差，评分口径也不同 | 同实体同刻交集的点命中率差，截为非负；不可当成旧 gap 的延续 |
| `learning_curve` 的 `learned_rate` / `improvement` | 范围覆盖及其增量 | 点命中及其增量；报告带 `metric_version`、`hit_threshold` |
| `compare_policies` 的 `probe_rate` | 探针范围覆盖 | 探针点命中；保留 `bound_coverage`，报告及各策略带版本标记，各策略带实际评分阈值 |
| Unified / Seven / SimulationLoop 的 `hit_rate` / rolling rate | 范围覆盖 | 仍是范围覆盖；不要与 learner 点命中混比。Unified / Seven 的 `point_hit_rate` 单列点命中 |
| 无样本率 / 损失 | 有些返回 1.0 或 0.0 | `None`，JSON 中为 `null`；不代表成功或零误差 |

`metric_version=2` 已贯穿 `eval_phase`、`evaluate`、`probe`、`learning_curve` 及策略比较。旧版本可能没有该字段。core 门面直接转发这些报告，调用方仍能读取标记。

## 保留旧范围门槛

已有监控若本意是检查预测范围覆盖，v2 改读 `bound_coverage`，并同时记录 `mean_bound`：范围变宽也会提高覆盖。不要再把这个率和 `naive_rate` 的点命中比较。

```python
result = learner.eval_phase(15)
version = result.get("metric_version", 1)
coverage = result["bound_coverage"] if version == 2 else result["learned_rate"]
if coverage is None:
    status = "unverified"
elif coverage < required_coverage:
    status = "below_coverage_target"
else:
    status = "coverage_target_met"
```

此映射保留旧 learned 范围评分的含义，但旧“无样本=满分”不再保留；因此不是逐值兼容。旧 oracle 的范围评分、缺样本处理和总体 gap 也不能通过字段重命名完整复原。

## 采用新的点精度指标

```python
result = explorer.probe(15)
if result.get("metric_version") != 2:
    raise ValueError("This point-accuracy evaluator requires metric_version=2")
point_rate = result["learned_rate"]
display = "未验证" if point_rate is None else f"{point_rate:.1%}"
record = {
    "metric_version": result["metric_version"],
    "hit_threshold": result["hit_threshold"],
    "outcomes": result["outcomes"],
    "point_rate": point_rate,
    "bound_coverage": result["bound_coverage"],
    "mean_bound": result["mean_bound"],
}
```

重新跑旧场景并标定新的点命中门槛；旧范围门槛不能直接当点精度门槛。缓存记录和图表至少区分版本、点阈值及评分样本数。使用 `is None`，不要写 `rate or 1.0`（会把真实 0% 当成满分），也不要直接对 `None` 做减法或百分比格式化。

本仓调用审计中，`evaluate` 转发评估报告；`learning_curve` 的空样本作差已处理；策略比较转发点率、范围和实际阈值；core 门面原样转发，没有发现额外将旧 learned 率用于资格判断的仓内脚本。外部调用方不在本仓审计范围内。

## 其他调用变化

- `verify_anchor` 的读取不再产生新稳定证据。调用方提交独立证据，并满足原多通道及时间间隔门槛；可传 `observation_tick` 合并同刻证据，`evidence_id` 去重重发。
- `masked_loss` 仅实现重建最后一次真实观测；`mask_last!=1` 明确报错。它与 next-state loss 都是诊断，不是参数梯度更新。
- 空场景 Unified / Seven / SimulationLoop 的率也是 `None`。`hit_rate` 的范围含义不变，汇总或导出仍需处理无样本。
- 空间多步预测接口保留上游实现；按 `target_tick` 匹配后续观测，提前或过期观测不作验证。
