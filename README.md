# EvenFlow

EvenFlow is a lightweight specification and tooling library for constructing **data-grounded navigation benchmarks** from real human trajectory data.

It provides:

- A small, explicit JSON spec for **layouts**, **scenes**, **tasks**, and **robots**
- A canonical motion representation based on **time-indexed trajectories**
- Validation utilities for catching spec errors early
- Rendering tools for visual debugging
- A reference **baseline geometry planner**
- A clean foundation for planners and **trajectory-based evaluation**
- CLI tools for **validation, rendering, and evaluation**

The goal is to make it easy to go from:

real trajectories → scene extraction → task definition → planner evaluation

---

# Concepts

EvenFlow separates the world into four layers:

## 1. Layout

The **layout** defines static geometry.

- boundary polygon
- obstacles
- exits (optional)
- metadata

## 2. Scene

A **scene** defines a time-localized slice of human behavior.

- time window
- tracking CSV reference
- dominant flow direction
- interaction center

## 3. Task

A **task** defines a robot navigation problem inside a scene.

- robot start
- robot goal
- task type
- target human trajectory

## 4. Robot

A **robot** defines the embodiment used by planners.

- kinematics
- footprint
- dynamics

---

# Motion Representation

## TrackStore
Dense representation of all trajectories in a scene.

## TrackSimple
Canonical trajectory:

- timestamps
- x, y
- optional vx, vy

## TrackPose
Extends TrackSimple with full pose data.

---

# Planner Output

PlanResult:

- planner_name
- success
- track (TrackSimple)
- path_length_m
- runtime_s
- message

Successful plans MUST include a trajectory.

---

# Evaluation

Evaluation compares:

robot trajectory vs target human trajectory

Metric example:
- min_human_distance_m

---

# CLI

Validate:
evenflow validate-layout layout.json
evenflow validate-scene scene.json
evenflow validate-task task.json
evenflow validate-robot robot.json
evenflow validate-plan plan.json

---

# Summary

EvenFlow is a trajectory-first benchmark:

- plans are trajectories
- humans are ground truth
- evaluation is trajectory-based
