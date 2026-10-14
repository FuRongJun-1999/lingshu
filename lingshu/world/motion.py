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
    return _estimate_points(observed_positions(history, eid, window, nested))


def _estimate_points(points):
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


class ObservationWindow:
    """A shared, one-pass view of the real observations used by inference.

    Normalize only the active window. Observation masks reject pairs without
    joint evidence before consulting older history. Each remaining pair
    shares one context lookup and traversal for both directions. This view
    lives for one learn/infer_patterns call; there is no cross-tick cache.
    """

    def __init__(self, history, window, nested=False):
        self.history = history
        self.start = max(0, len(history) - window)
        self.nested = nested
        self.frames = []
        self.points = {}
        self.masks = {}
        self._motions = {}
        self._pairs = {}
        for index, rec in enumerate(history[self.start:]):
            frame = {eid: cur['pos'] if nested else cur
                     for eid, cur in rec['entities'].items()}
            self.frames.append(frame)
            for eid, pos in frame.items():
                self.points.setdefault(eid, []).append((rec['tick'], pos))
                self.masks[eid] = self.masks.get(eid, 0) | (1 << index)

    def motion(self, eid):
        if eid not in self._motions:
            points = list(self.points.get(eid, []))
            if points and self.start and len(points) < len(self.frames):
                for index in range(self.start - 1, -1, -1):
                    rec = self.history[index]
                    cur = rec['entities'].get(eid)
                    if cur is not None:
                        points.insert(0, (rec['tick'], cur['pos'] if self.nested else cur))
                        break
            self._motions[eid] = _estimate_points(points)
        return self._motions[eid]

    def pair(self, a, b):
        """Return (mean cosine, directional samples, observed intervals).

        Stationary/coincident observations are measured intervals, but do not
        support a direction. Missing observations contribute neither count.
        """
        common = self.masks.get(a, 0) & self.masks.get(b, 0)
        if not common:
            return 0.0, 0, 0
        if (a, b) not in self._pairs:
            previous = None
            if self.start and common != (1 << len(self.frames)) - 1:
                for index in range(self.start - 1, -1, -1):
                    entities = self.history[index]['entities']
                    if a in entities and b in entities:
                        previous = ((entities[a]['pos'], entities[b]['pos']) if self.nested
                                    else (entities[a], entities[b]))
                        break
            scores_a, scores_b, intervals = [], [], 0
            for index, frame in enumerate(self.frames):
                if not (common & (1 << index)):
                    continue
                pa, pb = frame[a], frame[b]
                if previous is not None:
                    intervals += 1
                    tx, tz = pb[0] - pa[0], pb[2] - pa[2]
                    length = math.hypot(tx, tz)
                    if length > 1e-6:
                        for pos, old, dx_target, dz_target, scores in (
                                (pa, previous[0], tx, tz, scores_a),
                                (pb, previous[1], -tx, -tz, scores_b)):
                            dx, dz = pos[0] - old[0], pos[2] - old[2]
                            movement = math.hypot(dx, dz)
                            if movement > 1e-6:
                                scores.append((dx * dx_target + dz * dz_target) / (movement * length))
                previous = pa, pb
            for key, scores in (((a, b), scores_a), ((b, a), scores_b)):
                mean = round(sum(scores) / len(scores), 3) if scores else 0.0
                self._pairs[key] = mean, len(scores), intervals
        return self._pairs[a, b]


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
