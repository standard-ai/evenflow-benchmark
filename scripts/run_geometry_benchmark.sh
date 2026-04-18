#!/usr/bin/env bash
set -euo pipefail

BENCHMARK_ROOT="/Volumes/EvenFlow/benchmark"
ROBOT_JSON="./examples/robots/simple_disk.json"
OUTPUT_ROOT="./data/plans/benchmark"

mkdir -p "$OUTPUT_ROOT"

find "$BENCHMARK_ROOT" -type f -path "*/tasks/*.json" | sort | while read -r task_json; do
  scenario_type="$(basename "$(dirname "$(dirname "$task_json")")")"
  task_base="$(basename "$task_json" .json)"

  out_dir="$OUTPUT_ROOT/$scenario_type"
  out_plan="$out_dir/${task_base}.geometry.plan.json"

  mkdir -p "$out_dir"

  echo "Running geometry planner:"
  echo "  task:   $task_json"
  echo "  output: $out_plan"

  PYTHONPATH=src python -m evenflow.cli run-geometry \
    "$task_json" \
    "$ROBOT_JSON" \
    "$out_plan"
done

echo "Done. Plans written to $OUTPUT_ROOT"
