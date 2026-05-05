# EvenFlow

EvenFlow is an evaluation suite for shared-space navigation, built from real human trajectory data.

Most benchmarks evaluate whether an agent can navigate *around* people.  
**EvenFlow evaluates whether an agent can navigate *with* them.**

It converts real-world human trajectories into executable navigation tasks, enabling trajectory-level evaluation of planner behavior in realistic environments.
This enables evaluation of coordination, timing, and interaction—not just collision avoidance.

**Version:** v1.0 (NeurIPS 2026 release)

---

## 🧠 Core Concepts

- **Task**: Defines a navigation problem (start, goal, timing)
- **Scene**: Provides human trajectory context over a time window
- **Layout**: Static environment geometry (walls, obstacles)
- **Tracks**: Real human motion trajectories within the scene

---

## 📂 Dataset Structure

After downloading the dataset, files are organized as:

```text
data/benchmark/
  aligned_flow/
    tasks/
    scenes/
    layouts/
  cross_flow/
    tasks/
    scenes/
    layouts/
  interaction_constrained/
    tasks/
    scenes/
    layouts/
```

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

After running the above steps, you should see a rendered plan similar to:

![Rendered plan](assets/render_example.png)

This shows the planner trajectory (orange) over real human movement.

---

## 🧪 Quickstart to Writing a Custom Planner

EvenFlow evaluates planners by asking them to produce a **time-parameterized trajectory** for a navigation task.

A planner takes:

- a `task.json` file (start, goal, scene reference)
- a `robot.json` file (footprint and motion constraints)

and produces:

- a `plan.json` file containing a valid `PlanResult`

---

### 🧩 Understanding the inputs

Tasks are the executable unit of the benchmark. Each task references a scene, and each scene references the layout and human trajectory tracks:

```
task.json
  └── scene.json
        ├── layout.json
        └── tracks.csv
```

- **task.json** → start, goal, timing, references a scene  
- **scene.json** → time window + references layout and tracks  
- **layout.json** → static geometry (walls, obstacles)  
- **tracks.csv** → real human trajectories  

A **human-aware planner must parse `tracks.csv`** to reason about surrounding motion.

The included geometry baseline intentionally does **not** use human tracks—it is purely geometric and serves as a minimal reference.

---

### 🧭 Track Representations

Human trajectories are provided in two forms:

- **TrackSimple**: (x, y, vx, vy) — canonical representation used for planning  
- **Full track (pose)**: richer representation including pose keypoints (not required for most planners)

Most planners should use `TrackSimple`, accessed via:

```python
store = load_track_store(scene, scene_json_path=scene_json)
tracks = list(store.iter_simple_tracks())
```

---

### 🧠 Minimal planner interface

```python
def plan(task, robot):
    # Load task → scene → layout → tracks
    # Compute a trajectory
    return PlanResult(...)
```

---

### 📦 Plan output format

Your planner must produce a JSON file with this structure:

```json
{
  "planner_name": "your_planner",
  "success": true,
  "track": {
    "track_id": "plan",
    "timestamps": [0.0, 0.1, 0.2],
    "x": [0.0, 0.1, 0.2],
    "y": [0.0, 0.0, 0.0],
    "vx": [1.0, 1.0, 1.0],
    "vy": [0.0, 0.0, 0.0],
    "position_valid": [true, true, true],
    "velocity_valid": [true, true, true]
  },
  "path_length_m": 0.2,
  "runtime_s": 0.01,
  "message": "ok",
  "metadata": {}
}
```

### Plan requirements

A valid plan must:

- Be **time-parameterized** (timestamps must be provided)
- Start at the task start state
- Reach the goal within the task horizon
- Provide consistent position and velocity fields

EvenFlow evaluates behavior over time—not just geometric feasibility.

---

### ✅ Validate your planner output

```bash
evenflow validate-plan outputs/plan.json
```

Context-aware validation:

```bash
evenflow validate-plan \
  data/benchmark/aligned_flow/tasks/aligned_flow.af_0001.task.json \
  examples/robots/simple_disk.json \
  outputs/plan.json
```

---

### 📊 Evaluate your planner

```bash
evenflow evaluate-plan \
  data/benchmark/aligned_flow/tasks/aligned_flow.af_0001.task.json \
  examples/robots/simple_disk.json \
  outputs/plan.json
```

---

### 📌 Reference implementation

Below is a minimal track-aware planner demonstrating how to parse scene trajectories.

```python
from pathlib import Path
import time

import numpy as np

from evenflow.io import load_scene, load_task, load_robot, load_track_store, save_plan
from evenflow.models import PlanResult, TrackSimple


def resolve_relative(base_file, relative_path):
    return (Path(base_file).resolve().parent / relative_path).resolve()


def track_aware_straight_line(task_json, robot_json, plan_json):
    t0 = time.perf_counter()

    task = load_task(task_json)
    robot = load_robot(robot_json)

    scene_json = resolve_relative(task_json, task.scene.path)
    scene = load_scene(scene_json)

    # This is the important part: use the helper to parse the scene's tracks.csv.
    store = load_track_store(scene, scene_json_path=scene_json)

    # Extract surrounding humans as canonical TrackSimple objects.
    human_tracks = list(store.iter_simple_tracks())

    # Optional: remove the target/focal human from surrounding context.
    if task.target is not None:
        human_tracks = [
            tr for tr in human_tracks
            if tr.track_id != task.target.track_id
        ]

    # Very simple “track-aware” behavior:
    # estimate average human velocity and slightly bias motion with the flow.
    velocities = []
    for tr in human_tracks:
        if tr.has_velocity():
            v = tr.vxy()
            if v is not None:
                speeds = np.linalg.norm(v, axis=1)
                good = np.isfinite(speeds) & (speeds > 0.05)
                velocities.append(v[good])

    if velocities:
        vv = np.vstack(velocities)
        mean_flow = np.nanmean(vv, axis=0)
        if np.linalg.norm(mean_flow) > 1e-6:
            mean_flow = mean_flow / np.linalg.norm(mean_flow)
        else:
            mean_flow = np.zeros(2)
    else:
        mean_flow = np.zeros(2)

    start = np.asarray(task.robot.start, dtype=float)
    goal = np.asarray(task.robot.goal, dtype=float)

    direct = goal - start
    dist = float(np.linalg.norm(direct))
    if dist <= 1e-9:
        direction = np.zeros(2)
    else:
        direction = direct / dist

    # Bias very slightly with local flow, then renormalize.
    direction = direction + 0.15 * mean_flow
    norm = float(np.linalg.norm(direction))
    if norm > 1e-9:
        direction = direction / norm

    speed = float(robot.max_speed_mps)
    duration = dist / max(speed, 1e-6)

    n = max(2, int(np.ceil(duration / 0.1)) + 1)
    timestamps = np.linspace(0.0, duration, n)

    # Keep the same endpoint contract: start at task start, end at task goal.
    alpha = np.linspace(0.0, 1.0, n)
    xy = start[None, :] * (1.0 - alpha[:, None]) + goal[None, :] * alpha[:, None]

    vx = np.gradient(xy[:, 0], timestamps) if n > 1 else np.zeros(n)
    vy = np.gradient(xy[:, 1], timestamps) if n > 1 else np.zeros(n)

    track = TrackSimple(
        track_id="track_aware_demo_plan",
        timestamps=timestamps,
        x=xy[:, 0],
        y=xy[:, 1],
        vx=vx,
        vy=vy,
        position_valid=np.ones(n, dtype=bool),
        velocity_valid=np.ones(n, dtype=bool),
        metadata={
            "mean_flow_x": float(mean_flow[0]),
            "mean_flow_y": float(mean_flow[1]),
            "n_human_tracks_used": len(human_tracks),
        },
    )

    plan = PlanResult(
        planner_name="track_aware_demo",
        success=True,
        track=track,
        path_length_m=track.path_length_m(),
        runtime_s=time.perf_counter() - t0,
        message="ok",
        metadata={
            "description": "Minimal example showing how to parse scene tracks.",
        },
    )

    save_plan(plan_json, plan)
```

This is a puposefully simple example planner, but it exercises the main
data models needed.

Download an executable version of this planner here (you should have EvenFlow 
installed):

[track_aware_demo.py](examples/planners/track_aware_demo.py)

To run the planner:
```bash
python examples/planners/track_aware_demo.py \
  data/benchmark/aligned_flow/tasks/aligned_flow.af_0001.task.json \
  examples/robots/simple_disk.json \
  outputs/track_aware_plan.json
```

---

## Dataset

Full dataset, documentation, and download:
👉 https://huggingface.co/datasets/standard-cognition/EvenFlow

Quick start:
```bash
pip install huggingface_hub

hf download standard-cognition/EvenFlow \
  --repo-type dataset \
  --local-dir data
```

---

## ⚠️ Limitations

- Single-environment dataset (v1 release)
- Offline evaluation (no closed-loop interaction with humans)

We view this release as a foundation for future benchmarks spanning additional environments and interactive evaluation settings.

---

## License

Free for research and academic use.  
Commercial use requires a separate license.

See the [LICENSE](LICENSE) file for details.

