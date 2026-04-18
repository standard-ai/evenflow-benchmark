#!/usr/bin/env bash
set -euo pipefail

BENCHMARK_ROOT="/Volumes/EvenFlow/benchmark"
ROBOT_JSON="./examples/robots/simple_disk.json"
OUTPUT_ROOT="./data/plans/benchmark"

# default to "none" for the no-prior batch
PRIOR_MODE="${1:-none}"
JOBS="${JOBS:-5}"

mkdir -p "$OUTPUT_ROOT"

run_one() {
  local task_json="$1"

  local family
  family="$(basename "$(dirname "$(dirname "$task_json")")")"

  local task_base
  task_base="$(basename "$task_json" .json)"

  local task_stem
  task_stem="${task_base%.task}"

  local out_dir
  out_dir="$OUTPUT_ROOT/$family"
  mkdir -p "$out_dir"

  local out_plan
  out_plan="$out_dir/${task_stem}.prior_astar_${PRIOR_MODE}.plan.json"

  if [[ -f "$out_plan" ]]; then
    echo "[skip] $out_plan already exists"
    return 0
  fi

  echo "[run ] mode=$PRIOR_MODE family=$family task=$task_stem"

  PYTHONPATH="src:.." python -m planners.prior_astar.run_prior_astar \
    "$task_json" \
    "$ROBOT_JSON" \
    "$out_plan" \
    --prior-mode "$PRIOR_MODE"

  echo "[done] $out_plan"
}

export BENCHMARK_ROOT ROBOT_JSON OUTPUT_ROOT PRIOR_MODE
export -f run_one

TASK_LIST="$(mktemp)"
find "$BENCHMARK_ROOT" -type f -path "*/tasks/*.json" | sort > "$TASK_LIST"

echo "Benchmark root : $BENCHMARK_ROOT"
echo "Robot          : $ROBOT_JSON"
echo "Output root    : $OUTPUT_ROOT"
echo "Prior mode     : $PRIOR_MODE"
echo "Parallel jobs  : $JOBS"
echo "Task count     : $(wc -l < "$TASK_LIST")"
echo

if command -v parallel >/dev/null 2>&1; then
  echo "Using GNU parallel with caffeinate..."
  caffeinate -dimsu parallel -j "$JOBS" run_one :::: "$TASK_LIST"
else
  echo "GNU parallel not found; using xargs -P $JOBS with caffeinate..."
  caffeinate -dimsu xargs -I{} -P "$JOBS" bash -lc 'run_one "$@"' _ {} < "$TASK_LIST"
fi

rm -f "$TASK_LIST"

echo
echo "Done."
