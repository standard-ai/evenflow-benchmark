# EvenFlow

EvenFlow is a lightweight specification and tooling library for constructing **data‑grounded navigation benchmarks** from real human trajectory data. It provides:

- A small, explicit JSON spec for **layouts**, **scenes**, and **tasks**
- Validation utilities for catching spec errors early
- Rendering tools for visual debugging
- Optional overlay of real trajectory CSV data
- A clean foundation for planners and evaluation

The goal is to make it easy to go from:

```
real trajectories → scene extraction → task definition → planner evaluation
```

---

# Concepts

EvenFlow separates the world into three layers:

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
A **scene** defines a time‑localized slice of human behavior.

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
- task type (cross‑flow, aligned‑flow, etc)

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

# Rendering
EvenFlow includes built‑in rendering for visual debugging.

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
evenflow render-scene \
  layout.json \
  scene.json \
  out.png \
  --show-tracks
```

## Scene + Task

```
evenflow render-scene-task \
  layout.json \
  scene.json \
  task.json \
  out.png
```

## Scene + Task + Tracks

```
evenflow render-scene-task \
  layout.json \
  scene.json \
  task.json \
  out.png \
  --show-tracks
```

---

# Track CSV Format

The scene references a CSV file:

```
tracking.path
```

The renderer expects:

- track id column
- timestamp column
- x/y position columns

Example:

```
timestamp,person_track_id,bkg_x,bkg_y,vx,vy
1971...,1001,4.2,6.4,0.1,-0.8
...
```

Default coordinate fields:

```
bkg_x
bkg_y
```

Override with:

```
--tracks-x-field
--tracks-y-field
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

Validation checks:

- schema correctness
- layout references
- geometry consistency
- task validity

---

# Design Principles

EvenFlow is intentionally:

### Minimal
Small schema, few assumptions.

### Data‑grounded
Scenes reference real trajectories.

### Planner‑agnostic
Works with any planner.

### Reproducible
JSON spec defines the benchmark exactly.

### Visualizable
Everything can be rendered.

---

# Typical Workflow

1. Create layout

```
layout.json
```

2. Extract human trajectories

```
tracks.csv
```

3. Define scene

```
scene.json
```

4. Define task

```
task.json
```

5. Render

```
evenflow render-scene-task ...
```

6. Run planner

```
planner(layout, scene, task)
```

7. Evaluate

```
metrics(path, humans)
```

---

# Repository Structure

```
evenflow/
    models.py
    geometry.py
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
    tracks/
```

---

# Roadmap

Planned additions:

- planner interface
- baseline A*
- human‑aware cost functions
- evaluation metrics
- leaderboard support
- batch runner

---

# Motivation

Most social navigation benchmarks rely on:

- synthetic simulations
- small datasets
- hand‑crafted scenarios

EvenFlow instead builds benchmarks from:

- real human trajectories
- measured flow structure
- data‑derived interaction regions

This allows evaluation of planners that:

- move with flow
- avoid disruption
- exhibit legible behavior
- respect human conventions

---

# License

TBD

---

# Status

Early prototype — rendering and validation complete.

Next steps:

- planner API
- baseline planner
- evaluation framework

---

# Author

David Woollard
Standard Labs / Standard AI

