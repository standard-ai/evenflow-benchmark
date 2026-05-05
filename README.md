# EvenFlow

EvenFlow is an evaluation suite for shared-space navigation, built from real human trajectory data.

Most benchmarks evaluate whether an agent can navigate *around* people.  
**EvenFlow evaluates whether an agent can navigate *with* them.**

It converts real-world human trajectories into executable navigation tasks, enabling trajectory-level evaluation of planner behavior in realistic environments.

---

## 📦 Version

v1.0 (NeurIPS 2026 release)

---

## ⚡ Getting Started in 2 Minutes

### 1. Install

```bash
git clone https://github.com/standard-ai/evenflow-benchmark.git
cd evenflow-benchmark
pip install .
```

---

### 2. Download the dataset

```bash
pip install huggingface_hub

hf download standard-cognition/EvenFlow \
  --repo-type dataset \
  --local-dir data
```

---

### 3. Visualize a scene

```bash
evenflow render-scene \
  data/benchmark/aligned_flow/scenes/aligned_flow.af_0001.scene.json \
  outputs/scene.png \
  --show-tracks \
  --max-tracks 50
```

---

### 4. Run a planner (geometric baseline)

```bash
evenflow run-geometry \
  data/benchmark/aligned_flow/tasks/aligned_flow.af_0001.task.json \
  examples/robots/simple_disk.json \
  outputs/plan.json
```

---

### 5. Validate the plan

```bash
evenflow validate-plan outputs/plan.json
```

---

### 6. Evaluate the plan

```bash
evenflow evaluate-plan \
  data/benchmark/aligned_flow/tasks/aligned_flow.af_0001.task.json \
  examples/robots/simple_disk.json \
  outputs/plan.json
```

---

### 7. Visualize the result

```bash
evenflow render-plan \
  data/benchmark/aligned_flow/tasks/aligned_flow.af_0001.task.json \
  outputs/plan.json \
  outputs/render.png \
  --show-tracks
```

---

### ✅ Expected Result

![Rendered plan](assets/render_example.png)

This shows the planner trajectory (orange) over real human movement.

---

## 🧠 Core Concepts

- **Task**: Defines a navigation problem (start, goal, timing).
- **Scene**: Provides human trajectory context over a time window.
- **Layout**: Static environment geometry (walls, obstacles).
- **Tracks**: Real human motion trajectories within the scene.

---

## 📂 Dataset Structure

```
data/benchmark/
  aligned_flow/
    tasks/
    scenes/
    layouts/
  cross_flow/
  interaction_constrained/
```

---

## 🧭 Tracks

Human trajectories are provided in two forms:

- **TrackSimple**: (x, y, vx, vy) — canonical format used for planning  
- **Full track (pose)**: richer representation for future extensions  

Most planners should use `TrackSimple` via:

```python
store = load_track_store(scene, scene_json_path=scene_json)
tracks = list(store.iter_simple_tracks())
```

---

## 🧪 Quickstart to Writing a Custom Planner

EvenFlow evaluates planners by asking them to produce a **time-parameterized trajectory**.

### Minimal interface

```python
def plan(task, robot):
    return PlanResult(...)
```

---

### Plan requirements

A valid plan must:

- Be time-parameterized  
- Start at the task start  
- Reach the goal within the time horizon  

---

### Validate

```bash
evenflow validate-plan outputs/plan.json
```

---

### Evaluate

```bash
evenflow evaluate-plan \
  data/.../task.json \
  examples/robots/simple_disk.json \
  outputs/plan.json
```

---

## Dataset

Full dataset and documentation:

👉 https://huggingface.co/datasets/standard-cognition/EvenFlow

---

## ⚠️ Limitations

- Single environment (v1 release)  
- Offline evaluation (no closed-loop interaction)  

---

## License

Free for research and academic use.  
Commercial use requires a separate license.

See the [LICENSE](LICENSE) file for details.
