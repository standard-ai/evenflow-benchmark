# EvenFlow

EvenFlow is a lightweight specification and tooling library for constructing **data-grounded navigation benchmarks** from real human trajectory data. It provides:

- A small, explicit JSON spec for **layouts**, **scenes**, **tasks**, and **robots**
- Validation utilities for catching spec errors early
- Rendering tools for visual debugging
- Optional overlay of real trajectory CSV data
- A reference **baseline geometry planner**
- A clean foundation for planners and **evaluation**
- CLI tools for **planning, rendering, and evaluation**

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
  "tracking": {
    "path": "scene_001.tracks.csv"
  },
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
planner = GeometryPlanner()
result = planner.plan(layout, scene, task, robot)
```

---

# Planner Output Specification

Planners must return a `PlanResult`.

```python
PlanResult(
    planner_name: str
    success: bool
    waypoints: tuple[PlanWaypoint, ...]
    path_length_m: float | None
    runtime_s: float | None
    message: str
    metadata: dict
)
```

Waypoints:

```python
PlanWaypoint(
    x: float
    y: float
    t: float | None = None
)
```

Notes:

- timestamps optional
- must contain ≥ 2 waypoints if success=True
- timestamps must be monotonic if provided

---

# Evaluation

EvenFlow includes a **first-pass evaluation framework** for comparing planners.

Current metrics:

```
EvalResult
    success
    path_length_m
    runtime_s
    num_waypoints
    min_human_distance_m
    message
```

### min_human_distance_m

Minimum Euclidean distance between:

- robot path (polyline)
- all human track points in scene window

This provides a **first scene-aware safety metric**.

---

# Evaluate a Plan

```
evenflow evaluate-plan     layout.json     scene.json     task.json     robot.json     plan.json
```

Example output:

```
Evaluation OK
  success: True
  path_length_m: 14.0
  runtime_s: 0.002
  num_waypoints: 3
  min_human_distance_m: 3.36
  message: ok
```

This command:

1. loads layout
2. loads scene
3. loads task
4. loads robot
5. loads plan
6. loads scene track CSV
7. computes evaluation metrics

---

# Rendering Planner Outputs

Render plan:

```
evenflow render-plan layout.json plan.json out.png
```

Render full context:

```
evenflow render-scene-task-plan     layout.json     scene.json     task.json     plan.json     out.png
```

---

# Rendering

Layout

```
evenflow render-layout layout.json out.png
```

Scene

```
evenflow render-scene layout.json scene.json out.png
```

Scene + tracks

```
evenflow render-scene layout.json scene.json out.png --show-tracks
```

Scene + task

```
evenflow render-scene-task layout.json scene.json task.json out.png
```

Scene + task + plan

```
evenflow render-scene-task-plan     layout.json     scene.json     task.json     plan.json     out.png
```

---

# Validation

```
evenflow validate-layout layout.json
evenflow validate-scene scene.json
evenflow validate-task task.json
evenflow validate-robot robot.json
```

---

# Typical Workflow

1. Create layout  
2. Extract trajectories  
3. Define scene  
4. Define task  
5. Define robot  
6. Run planner  
7. Evaluate planner  
8. Compare planners  

---

# Repository Structure

```
evenflow/
    models.py
    geometry.py
    planners.py
    io.py
    evaluation.py
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
    plans/
```
