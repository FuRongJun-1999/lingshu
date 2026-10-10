#!/usr/bin/env bash
# 一键复现：先 fetch_data.py，再按「改前 / 各开关 / 全开」跑全部基准，结果写入 results/，汇总成 results/RESULTS.md。
# 用法：bash bench/repro/run_all.sh          （约 10 分钟）
#       FULL=1 bash bench/repro/run_all.sh   （另含 5 万条干扰与 1.5 万节点 TMS，约 25 分钟）
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore
python fetch_data.py
mkdir -p results; : > results/raw.jsonl
OFF="LINGSHU_FTS=0 LINGSHU_FTS_CONTEXT=0 LINGSHU_RETRIEVAL_NO_TOUCH=0 LINGSHU_CONTRADICTION=0 LINGSHU_TMS=0"
run() { local cfg="$1"; shift; echo ">> [$cfg] $*" >&2
        env $OFF $(cfg_env "$cfg") python "$@" 2>/dev/null | sed "s/^{/{\"cfg\": \"$cfg\", /" >> results/raw.jsonl; }
cfg_env() { case "$1" in
  base) echo "";;
  fts) echo "LINGSHU_FTS=1";;
  fts_ctx) echo "LINGSHU_FTS=1 LINGSHU_FTS_CONTEXT=1";;
  no_touch) echo "LINGSHU_RETRIEVAL_NO_TOUCH=1";;
  fts_no_touch) echo "LINGSHU_FTS=1 LINGSHU_RETRIEVAL_NO_TOUCH=1";;
  contra) echo "LINGSHU_CONTRADICTION=1";;
  fts_contra) echo "LINGSHU_FTS=1 LINGSHU_CONTRADICTION=1";;
  all) echo "LINGSHU_FTS=1 LINGSHU_FTS_CONTEXT=1 LINGSHU_RETRIEVAL_NO_TOUCH=1 LINGSHU_CONTRADICTION=1 LINGSHU_TMS=1";;
esac; }
SCALE="10000"; [ "${FULL:-0}" = 1 ] && SCALE="10000 50000"
# 1) 检索层 search_content
for c in base fts fts_ctx; do
  for a in "hmb A" "hmb B" "hmb B_full"; do run $c bench_search.py $a; done
  for n in $SCALE; do run $c bench_search.py hmbscale $n; ORDER=shuffle run $c bench_search.py hmbscale $n; run $c bench_search.py locomo $n; done
done
# 2) recall 端到端（三轮连续查询）
for c in base no_touch all; do for cut in A B; do run $c bench_recall_drift.py $cut 3; done; done
# 3) 知识更新：旧值被问过 1 次后写入新值
for c in base no_touch fts fts_no_touch; do run $c bench_supersede.py 200 1 0; done
# 4) 更正 / 矛盾（Refs #142）
for c in base contra fts fts_contra; do run $c bench_correction.py 30 5; done
python bench_contradiction_fp.py 2>/dev/null > results/contradiction_fp.txt
# 5) 真值维护（tms=0 为仅证据账本）
SIZES="300:600 2000:4000"; [ "${FULL:-0}" = 1 ] && SIZES="$SIZES 5000:10000"
for s in $SIZES; do for t in 0 1; do run base bench_tms.py ${s%:*} ${s#*:} 0.1 $t; done; done
python summarize.py results/raw.jsonl > results/RESULTS.md
echo "完成：results/RESULTS.md"
