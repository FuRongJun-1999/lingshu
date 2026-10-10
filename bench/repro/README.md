# 复现说明：检索与置信度六项改动（全部默认关闭）

本目录让维护者能从零复现 PR 里的全部读数：语料自动下载（锁定版本 + sha256 校验），评测脚本只依赖本仓库代码和 `numpy`、`pyyaml`。

```bash
python bench/repro/fetch_data.py         # 约 20 MB：hive-memory-bench@72130ac、locomo-zh-500@dsh-memory 2c21e3f、KdConv@HF 460c94a
bash bench/repro/run_all.sh              # 约 10 分钟，结果写到 bench/repro/results/RESULTS.md
FULL=1 bash bench/repro/run_all.sh       # 另含 5 万条干扰、1.5 万节点 TMS
python bench/repro/negative_results.py   # 试过但没有并入的方案（含稠密向量对照）
```

本次提交时的完整输出见 `results/RESULTS.md`（`raw.jsonl` 不入库）。脚本全部固定随机种子；除延迟外，重跑读数应当逐位一致。

## 六个开关

| 开关 | 改动 | 设计文档 |
|---|---|---|
| `LINGSHU_FTS=1` | 候选生成改为 FTS5 字符二元组 + BM25，取代「无序 LIKE + LIMIT 300/500」（Refs #35） | `docs/plans/检索候选生成_FTS5_BM25_设计_v0.1.md` |
| `LINGSHU_FTS_CONTEXT=1` | 在 FTS 之上过滤问句功能词，并做时序邻接平滑 | `docs/plans/检索语境扩展_功能词与时序邻接_设计_v0.1.md` |
| `LINGSHU_RETRIEVAL_NO_TOUCH=1` | 检索命中不再刷新 `last_access`，切断 search→近因→recall 的自增强 | `docs/plans/检索不刷新近因_设计_v0.1.md` |
| `LINGSHU_CONTRADICTION=1` | M5 近重复先做极性、数值、反义差分，发现冲突就不合并、不增信（Refs #142） | `docs/plans/矛盾感知_近重复差分_设计_v0.1.md` |
| `enable_evidence_ledger()` | 证据账本：置信度由证据推导（#430） | `docs/plans/证据账本_置信度推导_设计_v0.1.md` |
| `LINGSHU_TMS=1` / `enable_tms()` | 真值维护：前提被推翻时，下游结论自动撤回 | `docs/plans/真值维护_依赖撤回_设计_v0.1.md` |

## 脚本与口径

| 脚本 | 测什么 | 数据 |
|---|---|---|
| `bench_search.py` | `store.search_content` top-10：证据保留、另一侧覆盖、块级命中、hit@10、MRR、p50 延迟 | hmb 切A/切B/全书；加 KdConv 干扰；locomo-zh |
| `bench_recall_drift.py` | `engine.recall` 连续三轮，观察近因自增强 | hmb |
| `bench_supersede.py` | 知识更新：旧值被问过之后再写入新值，新值能否排第一 | 构造 |
| `bench_correction.py` | 更正是否被 M5 吞掉、原命题是否被增信、矛盾两侧能否同时召回、同义复述是否被误拆 | 构造 |
| `bench_contradiction_fp.py` | 真实语料里 Jaccard≥0.85 近重复对的冲突误报（KdConv 后 3.1 万句为未调参的留出集） | hmb、locomo、KdConv |
| `bench_tms.py` | 随机推理 DAG：撤回召回、误撤、传播耗时 | 构造 |
| `negative_results.py` | 实体桥、PPR 图桥、文档扩展、稠密 RRF | hmb |

hmb 指标口径见 `common.py` 文件头。这是我们自建的口径，与 hive-memory-bench 仓库自带脚本的数字**不可直接比较**，只用于同一口径内的前后对比。参数只在切A 上选定；切B、全书、加干扰版本和 locomo 都是留出集。构造场景的数据生成规则写在各脚本文件头，规则对改动前后两条路径一视同仁。

## 读数注意事项

- **写入顺序会影响改动前的 legacy 路径**。legacy 是无序 `LIKE … LIMIT 300/500`，实际按 rowid 扫描，先写入的节点天然占优。`hmbscale` 按顺序写入时，小说块先于 1 万条干扰写入，legacy 因此得到 11.1/44.1，高于 FTS5+语境扩展的 10.3/33.2。把写入顺序打乱后，legacy 跌到 1.5/17.3，FTS5+语境扩展为 8.6/31.7，与写入顺序无关的 FTS5 保持 7.3/27.8。真实使用中，相关记忆不会总是最早写入，所以打乱写入那一行更接近实际。两行都保留，不隐藏。
- 在跨领域干扰下，语境扩展的平滑可能把干扰句的邻居也一并抬高，延迟也会增加（切B 下 p50 由 7.7 ms 升到 10.7 ms；单机读数，多次运行间波动约 ±3 ms）。

## 不解决的问题（如实说明）

- **另一侧覆盖**（同一题里与题面最像和最不像的两条证据同时进 top-10）：没有哪个改动带来稳定提升，不要据此宣称多跳或对照证据问题已解决。
- 约三分之一漏检的证据与问句**零共享二元组**（作者动机、叙述形式这类分析题），字面检索取不到。
- 两条已有测试与新语义冲突，只在开关打开时出现，开关关闭时全部通过：
  - `test_core_search_discrimination.py::test_document_without_query_scores_zero`：FTS 不再返回零匹配文档；
  - `test_scene_ingest_dedup.py` 的 G6：矛盾感知开启后，坐标数值不同的观测不再被合并。
  两者都需要维护者裁定测试口径。

## 为什么不用稠密向量（embedding）

试过，有效果，但没有并入。`negative_results.py` 装了 `fastembed` 就会自动跑这组对照（bge-small-zh-v1.5，与本 PR 的离线等价物做 RRF 融合）：

| 切B（留出） | 证据保留 | 块级命中 | 另一侧覆盖 |
|---|---|---|---|
| 本 PR 的离线等价物 | 31.5 | 72.4 | 5/56 |
| + 稠密 RRF | 35.1 | 78.4 | 6/56 |

不并入的理由：
1. **违反 D-005（白箱、确定性、零 LLM）**。检索排序会依赖一个不可审计的 512 维模型；出现误召回时无法像 BM25 那样逐项解释是哪个词贡献了分数。
2. **确定性**：模型权重、onnxruntime 版本、CPU 指令集不同，余弦分数的末位就可能不同，进而打破同分排序。本 PR 的读数可以逐位复现，加入稠密检索后就做不到了。
3. **依赖与开销**：新增约 90 MB 模型文件和 onnxruntime；每次查询要多一次编码（CPU 上约 10 ms，与当前整条检索路径的耗时相当）；入库要为每个节点算向量并另建存储。
4. **收益集中在块级命中**（+6），证据保留 +3.6，另一侧覆盖没有改善。最难的问题它同样解决不了。
5. 引擎里已有 `_embedding_provider` 接口但从未接通。是否引入语义层属于架构决策，应由维护者另开议题决定，不该夹在本 PR 里。

其他负结果（同一离线口径，切B 证据保留）：实体桥 30.6（低于基线，在切A 上调的参数不能迁移），PPR 图桥 32.8（+1.3；在等价设置下做过 bootstrap，95% 置信区间跨 0），文档扩展 32.8（块级命中降到 67.9）。
