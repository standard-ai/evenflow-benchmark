# EvenFlow

EvenFlow is an evaluation suite for shared-space navigation, built from real human trajectory data.

Most benchmarks evaluate whether an agent can navigate *around* people.  
**EvenFlow evaluates whether an agent can navigate *with* them.**

It converts real-world human trajectories into executable navigation tasks, enabling trajectory-level evaluation of planner behavior in realistic environments.

---

## ⚡ Getting Started in 2 Minutes

### 1. Install

```bash
git clone https://github.com/<your-org>/evenflow.git
cd evenflow
pip install -e .
```

---

### 2. Download the dataset

```bash
huggingface-cli download standard-cognition/EvenFlow \
  --repo-type dataset \
  --local-dir ./data
```

---

### 3. Visualize a scene

```bash
evenflow render-scene \
  data/benchmark/aligned_flow/scenes/aligned_flow.af_0001.scene.json \
  --output outputs/scene.png
```

---

### 4. Run a planner (geometric baseline)

```bash
evenflow run-geometry \
  data/benchmark/aligned_flow/tasks/aligned_flow.af_0001.task.json \
  examples/robots/simple_disk.json \
  --output outputs/plan.json
```

---

### 5. Evaluate

```bash
evenflow evaluate \
  outputs/plan.json \
  --task data/benchmark/aligned_flow/tasks/aligned_flow.af_0001.task.json
```

---

### 6. Visualize the plan

```bash
evenflow render \
  outputs/plan.json \
  --task data/benchmark/aligned_flow/tasks/aligned_flow.af_0001.task.json \
  --output outputs/render.png
```

---

## 🧪 Quickstart to Writing a Custom Planner

EvenFlow is designed to evaluate arbitrary planners. You only need to implement a minimal interface.

### Minimal interface

```python
def plan(task, robot):
    # Your planner logic here
    return PlanResult(...)
```

### Required output

Your planner must return a **PlanResult** containing:

- `success` (bool)
- `track` (TrackSimple)
- `runtime_s` (float)

### Key requirement

The output trajectory must be **time-parameterized**.

EvenFlow evaluates behavior over time—not just geometric feasibility.

### Running your planner

Add a simple wrapper and call:

```bash
evenflow run \
  <task.json> \
  <robot.json> \
  --planner your_planner
```

### Reference example

See the geometric baseline:

```
evenflow/planners/geometry/
```

---

## Dataset

👉 https://huggingface.co/datasets/standard-cognition/EvenFlow

---

## License

Custom research license. See LICENSE file.
