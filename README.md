# EvenFlow

EvenFlow is an evaluation suite for shared-space navigation, built from real human trajectory data.

Most benchmarks evaluate whether an agent can navigate *around* people.  
EvenFlow evaluates whether an agent can navigate *with* them.

It provides a lightweight specification and tooling for converting real-world trajectories into executable navigation tasks, enabling trajectory-level evaluation of planner behavior in realistic human environments.

---

## What EvenFlow Provides

- A small, explicit JSON specification for **layouts**, **scenes**, **tasks**, and **robots**
- A canonical motion representation based on **time-indexed trajectories**
- Validation utilities for catching spec errors early
- Rendering tools for visual debugging
- Reference baseline planners
- A foundation for **trajectory-based evaluation**
- CLI tools for validation, rendering, and evaluation

---

## Installation

```bash
git clone https://github.com/<your-org>/evenflow.git
cd evenflow
pip install -e .
```

---

## Quickstart

Run a planner on a task:

```bash
evenflow run \
  benchmark/aligned_flow/tasks/aligned_flow.af_0001.task.json \
  examples/robots/simple_disk.json \
  --planner astar \
  --output outputs/plan.json
```

Evaluate the result:

```bash
evenflow evaluate \
  outputs/plan.json \
  --task benchmark/aligned_flow/tasks/aligned_flow.af_0001.task.json
```

Render the result:

```bash
evenflow render \
  outputs/plan.json \
  --task benchmark/aligned_flow/tasks/aligned_flow.af_0001.task.json \
  --output outputs/render.png
```

---

## Concepts

EvenFlow represents navigation as a hierarchy from static geometry to executable tasks constructed from human trajectories.

### 1. Layout

The **layout** defines static geometry.

- boundary polygon  
- obstacles  
- exits (optional)  
- metadata  

### 2. Scene

A **scene** defines a time-localized slice of human behavior.

- time window  
- tracking CSV reference  
- dominant flow direction  
- interaction center  

### 3. Task

A **task** defines a robot navigation problem inside a scene.

- robot start  
- robot goal  
- task type  
- target human trajectory  

### 4. Robot

A **robot** defines the embodiment used by planners.

- kinematics  
- footprint  
- dynamics  

---

## Motion Representation

### TrackStore
Dense representation of all trajectories in a scene.

### TrackSimple
Canonical trajectory:

- timestamps  
- x, y  
- optional vx, vy  

### TrackPose
Extends TrackSimple with full pose data.

---

## Planner Output

Planners produce a `PlanResult`:

- planner_name  
- success  
- track (TrackSimple)  
- path_length_m  
- runtime_s  
- message  

Successful plans **must** include a time-parameterized trajectory.

This enforces a trajectory-first evaluation, where behavior—not just feasibility—is assessed.

---

## Evaluation

Evaluation compares the robot trajectory to the target human trajectory within the same dynamic scene context.

Metrics capture:

- task completion  
- trajectory efficiency  
- interaction with surrounding humans  
- deviation from human-like behavior  

EvenFlow is designed as a diagnostic benchmark: rather than a single scalar score, evaluation exposes how planners behave across different interaction regimes.

---

## CLI

Validate:

```bash
evenflow validate-layout layout.json
evenflow validate-scene scene.json
evenflow validate-task task.json
evenflow validate-robot robot.json
evenflow validate-plan plan.json
```

---

## Data

Benchmark data is hosted separately due to size and licensing.

Download instructions:  
👉 <link-to-dataset>

Expected directory structure:

```
benchmark/
  aligned_flow/
  cross_flow/
  icn/
```

---

## Summary

EvenFlow is a trajectory-first evaluation suite:

- plans are trajectories  
- humans define the task  
- evaluation is behavior, not just success  

It provides a clean path from real-world data to executable navigation benchmarks.

---

## Citation

```bibtex
@article{evenflow2026,
  title={EvenFlow: Evaluating Navigation with Humans},
  author={...},
  year={2026}
}
```

---

## License

This dataset is released under a custom license.

- Free for research use  
- Commercial use requires a separate agreement  

See LICENSE for details.
