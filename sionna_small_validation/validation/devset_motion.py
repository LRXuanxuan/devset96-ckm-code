"""Shared physical trajectory for tracing and fixed-clock device reports."""
from __future__ import annotations

from .devset_routes import trajectory_at


def motion_at(route, time_sec):
    position, velocity, pose, _ = trajectory_at(route, time_sec)
    return position, velocity, pose
