"""Procedural, dock-safe animation support for Open Duck Mini v2."""

from .dock import DockMode
from .gaze import GazeSolver, GazeTracker
from .idle_engine import IdleEngine, IdleEngineConfig
from .scheduler import BehaviorScheduler

__all__ = ["BehaviorScheduler", "DockMode", "GazeSolver", "GazeTracker", "IdleEngine", "IdleEngineConfig"]
