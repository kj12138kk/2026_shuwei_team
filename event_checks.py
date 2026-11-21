"""Audit event histories and continuous vehicle separation without changing the model."""
from collections import defaultdict
from itertools import combinations

import numpy as np

from disturbances import DISTURBANCE, _clip, locked_at_command
from geometry import EPS, RADIUS, STARTS


def verify_history(reference, result, scenario):
    """All policies must agree with the original instructions up to the command time."""
    if scenario.kind == "nominal":
        return
    cutoff = scenario.cutoff

    def prefix(events):
        clipped = [_clip(e, cutoff) for e in events if e.end > e.begin + EPS]
        return sorted((e for e in clipped if e is not None), key=lambda e: (e.vehicle, e.begin, e.phase))

    former, current = prefix(reference.events), prefix(result.events)
    if len(former) != len(current):
        raise ValueError("Actions before command time differ from the original instructions.")
    for a, b in zip(former, current):
        if ((a.vehicle, a.target, a.phase) != (b.vehicle, b.target, b.phase) or
                max(abs(a.begin - b.begin), abs(a.end - b.end)) > 1e-7 or
                a.path.shape != b.path.shape or not np.allclose(a.path, b.path, rtol=0., atol=1e-6) or
                max(abs(a.pan0 - b.pan0), abs(a.pan1 - b.pan1),
                    abs(a.tilt0 - b.tilt0), abs(a.tilt1 - b.tilt1)) > 1e-6):
            raise ValueError("Travel or pointing before command time was changed.")

    for name, time in locked_at_command(reference.starts, scenario).items():
        if abs(result.starts.get(name, np.inf) - time) > EPS:
            raise ValueError("An original observation started before the command was not locked.")
    if any(e.phase == "abort" and e.begin < cutoff - EPS for e in result.events):
        raise ValueError("An original assignment was canceled before command time.")
    if scenario.new is not None and any(e.target == scenario.new.name and e.begin < cutoff - EPS for e in result.events):
        raise ValueError("A new alert received a command before communication was available.")
    failures = [e for e in result.events if e.phase == "failed"]
    if scenario.failure is None:
        if failures:
            raise ValueError("Vehicle failed in a scenario without a failure.")
    elif (len(failures) != 1 or failures[0].vehicle != scenario.failure or
          abs(failures[0].begin - DISTURBANCE) > EPS or
          any(e.vehicle == scenario.failure and e.end > DISTURBANCE + EPS for e in result.events)):
        raise ValueError("Failure was not applied at the declared time.")


def _segments(events, horizon):
    """Return (begin, end, position at begin, velocity, phase) per vehicle."""
    by_vehicle = defaultdict(list)
    failed = {}
    for event in events:
        if event.phase == "failed":
            failed[event.vehicle] = event.begin
        elif event.end > event.begin + EPS:
            by_vehicle[event.vehicle].append(event)
    zero = np.zeros(2)
    result = []
    for v, initial in enumerate(STARTS):
        stop = min(horizon, failed.get(v, np.inf))
        t, p, segments = 0., initial, []
        for e in sorted(by_vehicle[v], key=lambda item: item.begin):
            if e.begin >= stop:
                break
            if e.begin > t:
                segments.append((t, e.begin, p, zero, "idle"))
            if e.phase != "travel":
                segments.append((e.begin, min(stop, e.end), e.path[0], zero, e.phase))
            else:
                displacements = np.diff(e.path, axis=0)
                distances = np.linalg.norm(displacements, axis=1)
                total = distances.sum()
                if total <= EPS:
                    raise ValueError("A positive-duration drive has no path length.")
                times = e.begin + np.r_[0., distances.cumsum()] / total * (e.end - e.begin)
                for a, b, x, delta in zip(times[:-1], times[1:], e.path, displacements):
                    if b > a and a < stop:
                        segments.append((a, min(stop, b), x, delta / (b - a), "travel"))
            t, p = e.end, e.path[-1]
        if t < stop:
            segments.append((t, stop, p, zero, "idle"))
        result.append(segments)
    return result


def separation_audit(events, horizon):
    """Minimize relative quadratic distance on every overlapping linear leg."""
    closest = (np.inf, None, None, None, None)
    conflicts = defaultdict(set)
    all_conflicts = set()
    for (v, first), (w, second) in combinations(enumerate(_segments(events, horizon)), 2):
        i = j = 0
        while i < len(first) and j < len(second):
            a, b, p, u, phase = first[i]
            c, d, q, z, other = second[j]
            begin, end = max(a, c), min(b, d)
            if begin < end:
                r = p + (begin - a) * u - q - (begin - c) * z
                velocity = u - z
                speed_sq = velocity @ velocity
                step = float(np.clip(-(r @ velocity) / speed_sq, 0., end - begin)) if speed_sq > 0. else 0.
                distance = float(np.linalg.norm(r + step * velocity))
                if distance < closest[0]:
                    closest = distance, (v, w), begin + step, phase, other
                if distance < 2. * RADIUS - 1e-6:
                    pair = (v, w)
                    all_conflicts.add(pair)
                    if phase == other == "travel":
                        conflicts["travel_travel"].add(pair)
                    elif {phase, other} == {"travel", "observe"}:
                        conflicts["travel_observe"].add(pair)
                    elif phase == other == "observe":
                        conflicts["observe_observe"].add(pair)
            if b <= d:
                i += 1
            if d <= b:
                j += 1
    distance, pair, time, phase, other = closest
    return dict(min_separation_m=distance if pair is not None else None,
                closest_vehicles="+".join(f"V{v + 1}" for v in pair) if pair is not None else "",
                closest_time_s=time, closest_phases=f"{phase}|{other}" if pair is not None else "",
                conflicting_pairs=len(all_conflicts), travel_travel_pairs=len(conflicts["travel_travel"]),
                travel_observe_pairs=len(conflicts["travel_observe"]), observe_observe_pairs=len(conflicts["observe_observe"]))
