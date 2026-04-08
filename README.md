# EvenFlow

EvenFlow is a lightweight specification and tooling library for constructing **data-grounded navigation benchmarks** from real human trajectory data. It provides:

- A small, explicit JSON spec for **layouts**, **scenes**, **tasks**, and **robots**
- Validation utilities for catching spec errors early
- Rendering tools for visual debugging
- Optional overlay of real trajectory CSV data
- A reference **baseline geometry planner**
- A clean foundation for planners and evaluation

The goal is to make it easy to go from:

```
real trajectories → scene extraction → task definition → planner evaluation
```

---

# Concepts

EvenFlow separates the world into four layers:

## 1. Layout
The **layout** defines static geometry.

It contains:

- boundary polygon
- obstacles
- exits (optional)
- metadata

Example:

```json
{
  "layout_id": "example.simple_10x10",
  "boundary": [[0,0],[10,0],[10,10],[0,10]],
  "obstacles": [...],
  "exits": [...]
}
```

Layouts are reusable across many scenes and tasks.

---

## 2. Scene
A **scene** defines a time-localized slice of human behavior.

It contains:

- time window
- tracking CSV reference
- dominant flow direction (û)
- interaction center (p*)
- metadata

Example:

```json
{
  "scene_id": "example.scene_001",
  "layout_id": "example.simple_10x10",
  "window": {...},
  "tracking": {...},
  "flow": {
    "p_star": [5.0, 5.0],
    "u_hat": [0.0, -1.0]
  }
}
```

Scenes are derived from real trajectory data.

They represent **where and how humans are moving**.

---

## 3. Task
A **task** defines a robot navigation problem inside a scene.

It contains:

- robot start
- robot goal
- task type (cross-flow, aligned-flow, etc)

Example:

```json
{
  "task_id": "example.cross_001",
  "scene_id": "example.scene_001",
  "robot": {
    "start": [1.0, 8.5],
    "goal": [8.5, 2.0]
  }
}
```

Tasks define **what the robot must do**.

---

## 4. Robot
A **robot** defines the embodiment used by planners.

It contains:

- kinematics type
- footprint geometry
- size
- dynamic limits (optional)

Example:

```json
{
  "robot_id": "example.disk",
  "kinematics": "holonomic",
  "footprint": {
    "type": "disk",
    "radius_m": 0.25
  },
  "dynamics": {
    "max_speed_mps": 1.2
  }
}
```

Robots are defined separately so the **same task can be evaluated across different embodiments**.

---

# Baseline Planner

EvenFlow includes a reference planner:

### GeometryPlanner

This planner:

- respects layout boundary
- respects obstacle polygons
- respects robot footprint radius
- plans start → goal
- ignores human motion

It provides a **minimal geometry baseline** for:

- feasibility checking
- baseline comparisons
- regression testing
- evaluation sanity checks

Example:

```python
from evenflow import (
    load_layout,
    load_scene,
    load_task,
    load_robot,
    GeometryPlanner,
)

layout = load_layout("layout.json")
scene = load_scene("scene.json")
task = load_task("task.json")
robot = load_robot("robot.json")

planner = GeometryPlanner()
result = planner.plan(layout, scene, task, robot)

print(result.success)
print(result.path_length_m)
```

Planner output:

```
PlanResult
    success
    waypoints
    path_length_m
    runtime_s
    message
```

---

# Rendering

EvenFlow includes built-in rendering for visual debugging.

## Layout

```
evenflow render-layout layout.json out.png
```

## Scene

```
evenflow render-scene layout.json scene.json out.png
```

## Scene + Tracks

```
evenflow render-scene   layout.json   scene.json   out.png   --show-tracks
```

## Scene + Task

```
evenflow render-scene-task   layout.json   scene.json   task.json   out.png
```

## Scene + Task + Tracks

```
evenflow render-scene-task   layout.json   scene.json   task.json   out.png   --show-tracks
```

---

# Validation

EvenFlow includes validation tools:

```
evenflow validate-layout layout.json
```

```
evenflow validate-scene scene.json
```

```
evenflow validate-task task.json
```

```
evenflow validate-robot robot.json
```

---

# Typical Workflow

1. Create layout
2. Extract trajectories
3. Define scene
4. Define task
5. Define robot
6. Render
7. Run planner
8. Evaluate

---

# Repository Structure

```
evenflow/
    models.py
    geometry.py
    planners.py
    io.py
    validation.py
    render.py
    cli.py
```

Examples:

```
examples/
    layouts/
    scenes/
    tasks/
    robots/
    tracks/
```

---

# Status

Baseline planner implemented.

Next steps:

- human-aware planners
- flow-aware cost functions
- evaluation metrics
- leaderboard support
