#!/usr/bin/env bash
set -euo pipefail

BENCHMARK_ROOT="/Volumes/EvenFlow/benchmark/interaction_constrained"
ROBOT_JSON="./examples/robots/simple_disk.json"
OUTPUT_ROOT="./data/plans/benchmark/interaction_constrained"

mkdir -p "$OUTPUT_ROOT"

find "$BENCHMARK_ROOT" -type f -path "*/tasks/*.json" | sort | while read -r task_json; do
  task_base="$(basename "$task_json" .json)"
  task_stem="${task_base%.task}"

  out_plan="$OUTPUT_ROOT/${task_stem}.geometry.plan.json"

  echo "Running geometry planner:"
  echo "  task:   $task_json"
  echo "  output: $out_plan"

  PYTHONPATH=src python -m evenflow.cli run-geometry \
    "$task_json" \
    "$ROBOT_JSON" \
    "$out_plan"
done

echo "Done. ICN plans written to $OUTPUT_ROOT"
