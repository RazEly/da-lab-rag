#!/usr/bin/env bash
# Reproduce the whole analysis loop: build variant indexes -> gated sweeps -> plots.
# Run from part-2/analysis/. Uses the project venv (../.venv) which carries the
# production deps plus matplotlib/pandas (kept OUT of part-2/requirements.txt).
#
#   bash run_all.sh [--subset N | --full]   (default: --full)
set -euo pipefail
cd "$(dirname "$0")"

PY=../.venv/bin/python
SCOPE="${1:---full}"

echo "### A. chunk MAX_WORDS (index-affecting)"
$PY scripts/s2_chunk_maxwords.py "$SCOPE"
echo "### B. text-mode x MAX_WORDS (index-affecting)"
$PY scripts/s2_chunk_textmode.py "$SCOPE"

# C..G operate on the committed prod artifacts (no rebuild). If A/B changed
# production, rebuild artifacts (scripts/build_index.py) + invalidate caches FIRST.
echo "### C. hybrid vs solo";        $PY scripts/s5_retriever_grouped.py "$SCOPE"
echo "### D. per-retriever agg";     $PY scripts/s6_aggregation_heatmap.py "$SCOPE"
echo "### E. RRF vs blend / RRF_K";  $PY scripts/s7_rrf_ablation.py "$SCOPE"
echo "### F. BM25 IDF threshold";    $PY scripts/s7_idf_threshold.py "$SCOPE"
echo "### G. number filter mode";    $PY scripts/s7_number_filter.py "$SCOPE"

echo "### H-K. supporting stats (final config)"
$PY scripts/extra_stats.py "$SCOPE"

echo "### gate decisions:"; cat results/CHANGELOG.md
