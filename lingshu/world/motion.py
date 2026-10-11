"""Motion statistics from real, timestamped observations only."""
from __future__ import annotations

import math
from typing import Optional, Tuple


def observed_positions(history, eid, window=None, nested=False):
    """Return actual observations; include one context point before a tick window.

    Missing frames are not zero-velocity measurements. The preceding real
    observation supplies dt for the first displacement in the window.
    """
    start = max(0, len(history) - window) if window is not None else 0
    points = []
    for rec in history[start:]:
        cur = rec["entities"].get(eid)
        if cur is not None:
            points.append((rec["tick"], cur["pos"] if nested else cur))
    if points and start and len(points) < len(history) - start:
        for index in range(start - 1, -1, -1):
            rec = history[index]
            cur = rec["entities"].get(eid)
            if cur is not None:
                points.insert(0, (rec["tick"], cur["pos"] if nested else cur))
                break
    return points


def estimate_motion(history, eid, window, nested=False):
    """Return (persistence, speed, samples, last moving direction), or None."""
    points = observed_positions(history, eid, window, nested)
    moves = []
    for (t0, p0), (t1, p1) in zip(points, points[1:]):
        dt = t1 - t0
        if dt > 0:
            moves.append(((p1[0] - p0[0]) / dt, (p1[2] - p0[2]) / dt))
    if not moves:
        return None
    lengths = [math.hypot(*m) for m in moves]
    units = [(m[0] / length, m[1] / length)
             for m, length in zip(moves, lengths) if length > 1e-6]
    speed = sum(lengths) / len(lengths)
    persistence = (math.hypot(sum(u[0] for u in units), sum(u[1] for u in units))
                   / len(units)) if units else 0.0
    direction = (units[-1][0], 0.0, units[-1][1]) if units else None
    return round(persistence, 3), round(speed, 3), len(moves), direction


def recent_direction(history, eid, nested=False) -> Optional[Tuple[float, float, float]]:
    """Latest nonzero displacement, without fabricating missing observations."""
    newer = None
    for rec in reversed(history):
        cur = rec["entities"].get(eid)
        if cur is None:
            continue
        pos = cur["pos"] if nested else cur
        if newer is not None:
            dx, dz = newer[0] - pos[0], newer[2] - pos[2]
            length = math.hypot(dx, dz)
            if length > 1e-6:
                return dx / length, 0.0, dz / length
        newer = pos
    return None
