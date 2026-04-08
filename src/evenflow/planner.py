from __future__ import annotations

from abc import ABC, abstractmethod

from .models import Layout, PlanResult, Robot, Scene, Task


class BasePlanner(ABC):
    """
    Public planner interface for EvenFlow benchmark planners.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def plan(
        self,
        layout: Layout,
        scene: Scene,
        task: Task,
        robot: Robot,
    ) -> PlanResult:
        """
        Compute a plan for the given layout / scene / task / robot tuple.
        """
        raise NotImplementedError
