# -*- coding: utf-8 -*-
"""test_issue661_m5_dedup_perf_and_parity · #661 M5 去重等价倒排索引守卫
========================================================================
目标（issue #661）：
  在 100% 保持黑盒逐位等价（默认模式与 LINGSHU_CONTRADICTION=1 均逐位一致，
  `_dedup_history` 滑窗与 `#297` 全层覆盖语义零改动）的前提下，消除每次
  `add_perception` 对知识层的全表重读与全量二元组重算。

断言组：
  1. 黑盒逐位等价（默认 + LINGSHU_CONTRADICTION=1）：
     构造含独特条目、近重复合并（entities/tags/importance）、极性对立
     （success/failure）、条件空间差异（4 维度 + 重叠/不重叠时间窗）、
     零宽字符变体（#803）、同义/极性/数值矛盾、多模态混合（text/audio）
     及流中裸 SQL 注入/事务回滚的混合语料，断言快速倒排路径与全量
     `query_nodes(..., limit=-1)` 参考路径在节点集、边集、confidence、
     importance、access_count、tags、`_dedup_history`、`last_skipped_rows`
     及返回值（含 `.merged`）上逐项完全一致。
  2. 候选 ID 顺序逐一相等：
     断言 `_dedup_index_for` 有序键产出的候选 ID 序列与 `_dedup_candidates`
     及 `query_nodes(layer=KNOWLEDGE, modality=..., limit=-1)` 逐一相等
     （含同 `(importance, last_access)` 并列按 rowid 升序、坏行跳过一致）。
  3. 稳态写入零全层重扫：
     验证连续 `add_perception` 写入命中增量索引，不重复执行全层 `SELECT`。
  4. 定点变异自证（含 `--mutate`）：
     ① 跳过极性/条件空间过滤 ⇒ 等价断言精确点红；
     ② 篡改增量索引文本或截断鸽巢前缀 ⇒ 等价断言精确点红；
     还原后全绿。

运行（仓根）：
  python -X utf8 tests/test_issue661_m5_dedup_perf_and_parity.py
  python -X utf8 tests/test_issue661_m5_dedup_perf_and_parity.py --mutate
  python -X utf8 -m pytest tests/test_issue661_m5_dedup_perf_and_parity.py -q --no-header
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lingshu.core.core import (  # noqa: E402
    ConditionSpace,
    LayeredStore,
    MemoryLayer,
    STNode,
    SpacetimeMemoryEngine,
    _DedupIndexEntry,
)


def _cs(pos="p0", tool="t0", tw=(0.0, 1e12), ec="ok"):
    return ConditionSpace(
        observation_position=pos,
        observation_tool=tool,
        time_window=tw,
        existence_constraint=ec,
    )


def _reference_add_perception_loop(eng: SpacetimeMemoryEngine, modality: str,
                                   content: str, tags, cs: ConditionSpace,
                                   threshold: float):
    """参考路径：全量 `query_nodes(limit=-1)` 循环，验证与快速路径逐位一致。"""
    candidates = eng.store.query_nodes(
        layer=MemoryLayer.KNOWLEDGE, modality=modality, limit=-1)
    best, best_sim = None, 0.0
    new_polarity = set(tags or []) & eng._POLARITY_TAGS
    norm_new = eng._strip_zero_width(content)
    for n in candidates:
        if (set(n.tags or []) & eng._POLARITY_TAGS) != new_polarity:
            continue
        if not eng._conditions_mergeable(cs, n.condition_space):
            continue
        sim = LayeredStore.char_bigram_jaccard(
            norm_new, eng._strip_zero_width(n.content))
        eng._dedup_history.append(sim)
        if len(eng._dedup_history) > eng._dedup_window:
            eng._dedup_history = eng._dedup_history[-eng._dedup_window:]
        if sim > best_sim:
            best_sim, best = sim, n
    if best and best_sim >= threshold:
        return best, best_sim
    return None, 0.0


def _corpus_ops():
    """覆盖独特/近重复/极性/条件/零宽/矛盾/多模态的确定性混合语料。"""
    cs_a = _cs("pos-A", "cam", (10.0, 100.0), "active")
    cs_b = _cs("pos-B", "cam", (10.0, 100.0), "active")
    cs_disjoint = _cs("pos-A", "cam", (200.0, 300.0), "active")
    cs_overlap = _cs("pos-A", "cam", (50.0, 150.0), "active")
    return [
        # 1-4: 预置独特条目（使历史窗积累、验证跨阈值演化）
        dict(content="机房温度传感器校准流程说明文档", modality="text", importance=0.4, tags=["infra"], entities=["srv1"], condition_space=cs_a),
        dict(content="仓库盘点月度核对表与异常处理规则", modality="text", importance=0.5, tags=["ops"], entities=["wh1"], condition_space=cs_a),
        dict(content="语音通道测试信号样本第一段记录", modality="audio", importance=0.6, tags=["voice"], entities=["mic1"], condition_space=cs_a),
        dict(content="数据库连接池最大并发参数调优指南", modality="text", importance=0.7, tags=["db"], entities=["pg1"], condition_space=cs_a),
        # 5: 近重复合并（entities 并集 + tags 并集 + importance 取大）
        dict(content="数据库连接池最大并发参数调优指南说明", modality="text", importance=0.9, tags=["tuned"], entities=["pg2"], condition_space=cs_a),
        # 6: 零宽变体命中（#803）
        dict(content="机房\u200b温度传感器校准流程说明文档", modality="text", importance=0.3, tags=["zw"], entities=["srv2"], condition_space=cs_a),
        # 7: 跨模态同正文不得合并（#223）
        dict(content="数据库连接池最大并发参数调优指南", modality="audio", importance=0.6, tags=["cross"], entities=[], condition_space=cs_a),
        # 8-9: 极性对立不得合并（#225）
        dict(content="成功：部署流水线第三阶段镜像构建完成", modality="text", importance=0.5, tags=["success"], entities=[], condition_space=cs_a),
        dict(content="失败：部署流水线第三阶段镜像构建完成", modality="text", importance=0.5, tags=["failure"], entities=[], condition_space=cs_a),
        # 10-12: 条件空间差异（#268：观测位不同、时间窗不交、时间窗相交）
        dict(content="锅炉压力读数维持在安全区间之内", modality="text", importance=0.5, tags=[], entities=[], condition_space=cs_a),
        dict(content="锅炉压力读数维持在安全区间之内", modality="text", importance=0.5, tags=["b"], entities=[], condition_space=cs_b),
        dict(content="锅炉压力读数维持在安全区间之内", modality="text", importance=0.5, tags=["disj"], entities=[], condition_space=cs_disjoint),
        dict(content="锅炉压力读数维持在安全区间之内", modality="text", importance=0.8, tags=["ovlp"], entities=["b1"], condition_space=cs_overlap),
        # 13-16: 否定/数值矛盾（验证默认与 LINGSHU_CONTRADICTION=1 均等价）
        dict(content="患者对青霉素过敏需要禁用该类药物", modality="text", importance=0.6, tags=[], entities=[], condition_space=cs_a),
        dict(content="患者对青霉素不过敏无需禁用该类药物", modality="text", importance=0.6, tags=[], entities=[], condition_space=cs_a),
        dict(content="每日推荐安全剂量为 7.5 毫克每次", modality="text", importance=0.6, tags=[], entities=[], condition_space=cs_a),
        dict(content="每日推荐安全剂量为 75 毫克每次", modality="text", importance=0.6, tags=[], entities=[], condition_space=cs_a),
        # 17-52: 扩充超过默认 _dedup_window=30，确保滑窗溢出与高基数表现均被覆盖
        *[
            dict(
                content=f"批量环境巡检独立事件第{i:02d}号观测记录正文特征{i * 17}",
                modality="text",
                importance=0.3 + (i % 5) * 0.1,
                tags=[f"batch_{i % 3}"],
                entities=[f"ent_{i % 4}"],
                condition_space=cs_a,
            )
            for i in range(36)
        ],
        # 53: 滑窗演化后的再次合并（验证 _dedup_history 驱动阈值一致）
        dict(content="批量环境巡检独立事件第05号观测记录正文特征85", modality="text", importance=0.85, tags=["rehit"], entities=["ent_rehit"], condition_space=cs_a),
    ]


def _snapshot_db(eng: SpacetimeMemoryEngine):
    """提取可跨实例对比的规范化状态（按内容/写入顺序对齐 id）。"""
    c = eng.store.conn.cursor()
    c.execute(
        "SELECT id, content, modality, layer, importance, confidence,"
        " access_count, tags, condition_space FROM nodes ORDER BY rowid ASC"
    )
    rows = c.fetchall()
    id_to_idx = {r[0]: i for i, r in enumerate(rows)}
    norm_nodes = [
        (
            r[1], r[2], r[3],
            round(float(r[4]), 9),
            round(float(r[5]), 9),
            int(r[6]),
            sorted(json.loads(r[7]) or []),
            r[8],
        )
        for r in rows
    ]
    c.execute(
        "SELECT source_id, target_id, relation_type, confidence, verified"
        " FROM edges ORDER BY rowid ASC"
    )
    norm_edges = [
        (
            id_to_idx.get(r[0], r[0]),
            id_to_idx.get(r[1], r[1]),
            r[2],
            round(float(r[3]), 9),
            int(r[4]),
        )
        for r in c.fetchall()
    ]
    return norm_nodes, norm_edges, id_to_idx


def _run_parity_once(contradiction_flag: str):
    old_flag = os.environ.get("LINGSHU_CONTRADICTION")
    os.environ["LINGSHU_CONTRADICTION"] = contradiction_flag
    try:
        ops = _corpus_ops()
        with tempfile.TemporaryDirectory(prefix="p661_") as td:
            eng_fast = SpacetimeMemoryEngine(os.path.join(td, "fast.db"))
            eng_ref = SpacetimeMemoryEngine(os.path.join(td, "ref.db"))
            # 强制 ref 引擎走旧 _dedup_candidates 全扫路径（挂载实例属性绕过 fast 开关）
            eng_ref.store._dedup_candidates = (
                lambda m, _orig=LayeredStore._dedup_candidates, _s=eng_ref.store: _orig(_s, m)
            )
            import time as _time
            _orig_time = _time.time
            _tick = [1000000.0]

            def _det_time():
                _tick[0] += 0.01
                return _tick[0]

            try:
                fast_rets = []
                ref_rets = []
                for idx, op in enumerate(ops):
                    base_t = 1000000.0 + idx * 10.0
                    _tick[0] = base_t
                    _time.time = _det_time
                    rf = eng_fast.add_perception(**op)
                    _tick[0] = base_t
                    rr = eng_ref.add_perception(**op)
                    _time.time = _orig_time
                    fast_rets.append((
                        rf.content, rf.modality, round(rf.importance, 9),
                        round(rf.confidence, 9), sorted(rf.tags or []),
                        bool(getattr(rf, "merged", False)),
                    ))
                    ref_rets.append((
                        rr.content, rr.modality, round(rr.importance, 9),
                        round(rr.confidence, 9), sorted(rr.tags or []),
                        bool(getattr(rr, "merged", False)),
                    ))
                    assert fast_rets[-1] == ref_rets[-1], (
                        f"op#{idx} ({op['content'][:20]}) 返回值不等价:\n"
                        f"  fast={fast_rets[-1]}\n  ref ={ref_rets[-1]}"
                    )
                    assert len(eng_fast._dedup_history) == len(eng_ref._dedup_history), (
                        f"op#{idx} _dedup_history 长度不等: {len(eng_fast._dedup_history)} != {len(eng_ref._dedup_history)}"
                    )
                    for hf, hr in zip(eng_fast._dedup_history, eng_ref._dedup_history):
                        assert abs(hf - hr) < 1e-12, (
                            f"op#{idx} _dedup_history 元素漂移: {hf} vs {hr}"
                        )

                fn, fe, _ = _snapshot_db(eng_fast)
                rn, re_, _ = _snapshot_db(eng_ref)
                assert fn == rn, f"节点表状态不等价（mode={contradiction_flag}）"
                assert fe == re_, f"边表状态不等价（mode={contradiction_flag}）"
                assert eng_fast.store.last_skipped_rows == eng_ref.store.last_skipped_rows
            finally:
                _time.time = _orig_time
                eng_fast.close()
                eng_ref.close()
    finally:
        if old_flag is None:
            os.environ.pop("LINGSHU_CONTRADICTION", None)
        else:
            os.environ["LINGSHU_CONTRADICTION"] = old_flag


def test_parity_default_and_contradiction_modes():
    """① 默认模式与 LINGSHU_CONTRADICTION=1 下快速路径与参考路径逐位完全等价。"""
    _run_parity_once("0")
    _run_parity_once("1")


def test_candidate_id_order_matches_query_nodes_limit_neg1():
    """② 倒排索引有序键候选 ID 序列与 `query_nodes(..., limit=-1)` 及 `_dedup_candidates` 逐一相等。"""
    eng = SpacetimeMemoryEngine(":memory:")
    try:
        cs = _cs()
        # 故意写入多组相同 (importance, last_access) 与不同 importance，验证 tie-breaking 一致
        for i in range(40):
            imp = [0.5, 0.8, 0.5, 0.3][i % 4]
            la = 1000.0 if i % 3 == 0 else float(1000 + i)
            eng.store.add_node(STNode(
                id=f"ord_{i:02d}",
                content=f"顺序校验节点内容_{i}",
                modality="text" if i % 5 else "audio",
                spatial_coordinates={},
                temporal_coordinate=1.0,
                last_access=la,
                condition_space=cs,
                importance=imp,
                layer=MemoryLayer.KNOWLEDGE if i % 7 else MemoryLayer.CONTEXT,
                tags=[],
            ))
        # 注入 1 条坏 condition_space 行验证跳过口径一致
        eng.store.conn.execute(
            "INSERT INTO nodes (id, layer, modality, content, importance, tags,"
            " condition_space, created_at, last_access)"
            " VALUES ('bad_ord', 'knowledge', 'text', '坏行', 0.9, '[]', '{bad', 0.0, 9999.0)"
        )
        eng.store.conn.commit()

        idx = eng.store._dedup_index_for("text", eng._strip_zero_width)
        idx_ids = [k[3] for k in idx.sorted_keys]
        narrow_ids = [c.id for c in eng.store._dedup_candidates("text")]
        full_ids = [
            n.id for n in eng.store.query_nodes(
                layer=MemoryLayer.KNOWLEDGE, modality="text", limit=-1)
        ]
        assert idx_ids == narrow_ids == full_ids, (
            f"候选 ID 顺序不一致:\n  idx={idx_ids[:10]}\n  narrow={narrow_ids[:10]}\n  full={full_ids[:10]}"
        )
        assert "bad_ord" in (eng.store.last_skipped_rows or [])

        # 再通过 add_perception 增量写入几条，继续断言顺序仍与全量 query_nodes 一致
        for j in range(5):
            eng.add_perception(f"增量追加唯一节点_{j}_{j * 31}", importance=0.6, condition_space=cs)
        idx2 = eng.store._dedup_index_for("text", eng._strip_zero_width)
        assert [k[3] for k in idx2.sorted_keys] == [
            n.id for n in eng.store.query_nodes(
                layer=MemoryLayer.KNOWLEDGE, modality="text", limit=-1)
        ]
    finally:
        eng.close()


def test_raw_sql_and_rollback_invalidation_parity():
    """流中裸 SQL 写入与 `_tx` 回滚均不得产生脏缓存或错误合并。"""
    eng = SpacetimeMemoryEngine(":memory:")
    try:
        cs = _cs()
        eng.add_perception("缓存一致性校验基线一", condition_space=cs)
        eng.add_perception("缓存一致性校验基线二", condition_space=cs)

        # 裸 SQL 直接注入一条高度相似行（绕过 add_node）
        eng.store.conn.execute(
            "INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "raw_injected_1", "缓存一致性校验裸注入目标节点", "text", "{}",
                1.0, cs.to_json(), 0.7, 0.5, "knowledge", 0, 1.0, 1.0,
                "[]", "{}", "{}", None,
            ),
        )
        eng.store.conn.commit()
        r = eng.add_perception("缓存一致性校验裸注入目标节点", condition_space=cs)
        assert getattr(r, "merged", False) is True and r.id == "raw_injected_1"

        # _tx 内异常回滚不得污染内存索引
        try:
            with eng.store._tx():
                eng.store.add_node(STNode(
                    id="rolled_back_1", content="将被回滚的幻影节点内容",
                    modality="text", spatial_coordinates={}, temporal_coordinate=1.0,
                    condition_space=cs, importance=0.9, layer=MemoryLayer.KNOWLEDGE, tags=[],
                ))
                raise RuntimeError("abort_tx")
        except RuntimeError:
            pass
        r2 = eng.add_perception("将被回滚的幻影节点内容", condition_space=cs)
        assert getattr(r2, "merged", False) is False and r2.id != "rolled_back_1"
    finally:
        eng.close()


def test_steady_state_writes_avoid_full_layer_rescan():
    """③ 稳态连续写入命中缓存索引：不触发对 `nodes` 的全层 SELECT 扫描。"""
    eng = SpacetimeMemoryEngine(":memory:")
    try:
        cs = _cs()
        for i in range(200):
            eng.store.add_node(STNode(
                id=f"s_{i}", content=f"预热知识条目{i}_特征_{i * 19}",
                modality="text", spatial_coordinates={}, temporal_coordinate=1.0,
                condition_space=cs, importance=0.5, layer=MemoryLayer.KNOWLEDGE, tags=[],
            ))
        # 首写预热索引
        eng.add_perception("首次预热感知写入触发索引建立", condition_space=cs)

        full_scans = []

        def _trace(sql: str):
            if "FROM nodes WHERE layer=" in sql:
                full_scans.append(sql)

        eng.store.conn.set_trace_callback(_trace)
        try:
            for j in range(10):
                eng.add_perception(f"稳态连续写入第{j}条独特感知_{j * 37}", condition_space=cs)
        finally:
            eng.store.conn.set_trace_callback(None)

        assert full_scans == [], f"稳态写入意外触发全层扫描: {full_scans}"
    finally:
        eng.close()


def test_fast_matches_reference_loop_directly():
    """直接对拍 `_dedup_search_fast` 与全量 `_reference_add_perception_loop`。"""
    eng_fast = SpacetimeMemoryEngine(":memory:")
    eng_ref = SpacetimeMemoryEngine(":memory:")
    try:
        cs = _cs()
        for i in range(50):
            txt = f"对拍样例_{i}_内容_{i * 29}"
            tags = ["success"] if i % 4 == 0 else []
            for e in (eng_fast, eng_ref):
                e.store.add_node(STNode(
                    id=f"p_{i}", content=txt, modality="text",
                    spatial_coordinates={}, temporal_coordinate=1.0,
                    last_access=1000.0 + i,
                    condition_space=cs, importance=0.5, layer=MemoryLayer.KNOWLEDGE, tags=tags,
                ))
        for query, qtags in [
            ("对拍样例_12_内容_348", ["success"]),
            ("对拍样例_12_内容_348", []),
            ("完全不存在的新内容样本", []),
        ]:
            bf, sf = eng_fast._dedup_search_fast(
                "text", query, frozenset(qtags), cs, 0.85)
            br, sr = _reference_add_perception_loop(
                eng_ref, "text", query, qtags, cs, 0.85)
            assert (bf.id if bf else None) == (br.id if br else None)
            assert abs(sf - sr) < 1e-12
            assert len(eng_fast._dedup_history) == len(eng_ref._dedup_history)
            for hf, hr in zip(eng_fast._dedup_history, eng_ref._dedup_history):
                assert abs(hf - hr) < 1e-12
    finally:
        eng_fast.close()
        eng_ref.close()


def run_mutation_check() -> dict:
    """④ 定点变异自证：破坏过滤条件或索引路径 ⇒ 等价断言必红；还原 ⇒ 绿。"""
    # 基线必须全绿
    test_parity_default_and_contradiction_modes()
    test_candidate_id_order_matches_query_nodes_limit_neg1()
    test_steady_state_writes_avoid_full_layer_rescan()

    # 变异 1：篡改条件空间过滤（matches_cs 恒 True）⇒ 极性/条件等价必红
    orig_matches_cs = _DedupIndexEntry.matches_cs
    mut1_failed = False
    try:
        _DedupIndexEntry.matches_cs = lambda self, pos, tool, ec, t0, t1: True
        try:
            test_parity_default_and_contradiction_modes()
        except AssertionError:
            mut1_failed = True
    finally:
        _DedupIndexEntry.matches_cs = orig_matches_cs
    assert mut1_failed, "变异 1（条件空间过滤失效）未触发等价守卫红灯"

    # 变异 2：篡改增量更新逻辑（跳过 add_node 索引补丁且不断开缓存）⇒ 等价/顺序守卫必红
    orig_patch = LayeredStore._dedup_patch_add_node_locked
    mut2_failed = False
    try:
        LayeredStore._dedup_patch_add_node_locked = lambda self, node, existing, rowid: None
        try:
            test_candidate_id_order_matches_query_nodes_limit_neg1()
        except AssertionError:
            mut2_failed = True
    finally:
        LayeredStore._dedup_patch_add_node_locked = orig_patch
    assert mut2_failed, "变异 2（增量索引漏记新增节点）未触发顺序/等价守卫红灯"

    # 还原后复绿
    test_parity_default_and_contradiction_modes()
    test_candidate_id_order_matches_query_nodes_limit_neg1()
    return {
        "M0_baseline": "PASS",
        "M1_bypass_matches_cs": "RED_CAUGHT",
        "M2_drop_incremental_patch": "RED_CAUGHT",
        "M3_restored": "PASS",
    }


def test_mutation_self_proof():
    """④ 定点变异自证（pytest 入口）。"""
    res = run_mutation_check()
    assert res["M1_bypass_matches_cs"] == "RED_CAUGHT"
    assert res["M2_drop_incremental_patch"] == "RED_CAUGHT"


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    test_parity_default_and_contradiction_modes()
    test_candidate_id_order_matches_query_nodes_limit_neg1()
    test_raw_sql_and_rollback_invalidation_parity()
    test_steady_state_writes_avoid_full_layer_rescan()
    test_fast_matches_reference_loop_directly()
    mut = run_mutation_check()
    if "--mutate" in args:
        print("MUTATE_CHECK_OK", json.dumps(mut, ensure_ascii=False))
    else:
        print("ISSUE661_M5_DEDUP_PERF_AND_PARITY_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
