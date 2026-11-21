"""Discrete-event replay, failure/communication scenarios and replanning."""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product

import numpy as np

from geometry import ANGULAR_SPEED, EPS, OBSERVE, SPEED, STARTS, Target, path_length, split_path
from scheduler import State, build_plan, make_leg, solve

DISTURBANCE = 5.
WINDOW = 60.


@dataclass(frozen=True)
class Event:
    vehicle: int
    target: str
    phase: str
    begin: float
    end: float
    path: np.ndarray
    pan0: float = 0.
    tilt0: float = 0.
    pan1: float = 0.
    tilt1: float = 0.


@dataclass(frozen=True)
class Scenario:
    key: str
    kind: str
    new: Target | None = None
    failure: int | None = None
    recovery: float | None = None

    @property
    def cutoff(self):
        return DISTURBANCE if self.recovery is None else self.recovery


def scenarios():
    discovery = tuple(Target(f"N{a:03d}_{r}", a, r, 50.)
                      for a, r in [*product(range(0, 360, 90), (30, 50)),
                                   *((n, 40) for n in range(45, 360, 90))])
    crossed = (discovery[1], discovery[5])  # East and west limits of the square.
    out = [Scenario("nominal", "nominal")]
    for new_group, fail_group, out_group in product((False, True), repeat=3):
        if not any((new_group, fail_group, out_group)):
            continue
        kind = "+".join(name for name, flag in zip(("new", "failure", "outage"), (new_group, fail_group, out_group)) if flag)
        alerts = (crossed if fail_group or out_group else discovery) if new_group else (None,)
        failures = range(5) if fail_group else (None,)
        recoveries = (10., 20., 30.) if out_group else (None,)
        for alert, failure, recovery in product(alerts, failures, recoveries):
            key = "_".join(str(value) for value in (alert.name if alert else None,
                                                   f"V{failure + 1}" if failure is not None else None,
                                                   f"R{int(recovery)}" if recovery else None) if value)
            out.append(Scenario(key, kind, alert, failure, recovery))
    return discovery, out


def pressure_scenarios():
    """Cross all failure identities and eight edge/far-field bearings at late recovery."""
    targets = tuple(Target(f"N{theta:03d}_50", theta, 50., 50.) for theta in range(0, 360, 45))
    design = [Scenario(f"{target.name}_V{failure + 1}_R30", "new+failure+outage",
                       target, failure, 30.)
              for target, failure in product(targets, range(len(STARTS)))]
    return targets, design


def _point_event(vehicle, target, phase, begin, end, point, a=(0., 0.), b=(0., 0.)):
    return Event(vehicle, target, phase, begin, end, np.asarray([point]), *a, *b)


def _approach(a, b, elapsed):
    delta = np.asarray(b) - a
    return np.asarray(a) + np.sign(delta) * np.minimum(np.abs(delta), ANGULAR_SPEED * max(0., elapsed))


def _leg_events(leg, target, start=None, continued=None):
    v, p, q = leg.vehicle, leg.path[0], leg.path[-1]
    a, b = (leg.initial_pan, leg.initial_tilt), (leg.pan, leg.tilt)
    if continued is None:
        phase = "depart" if leg.arrive > leg.depart + EPS else "at_station"
        yield _point_event(v, target, phase, leg.depart, leg.depart, p)
    if leg.arrive > leg.depart + EPS:
        yield Event(v, target, "travel", leg.depart, leg.arrive, leg.path)
        yield _point_event(v, target, "arrive", leg.arrive, leg.arrive, q)
    if leg.ready > leg.arrive + EPS:
        yield _point_event(v, target, "slew", leg.arrive, leg.ready, q, a, b)
    if start is None:
        return
    if start > leg.ready + EPS:
        yield _point_event(v, target, "wait_sync", leg.ready, start, q, b, b)
    yield _point_event(v, target, "observe", start, start + OBSERVE, q, b, b)
    if leg.slew > EPS:
        yield _point_event(v, target, "reset", start + OBSERVE, start + OBSERVE + leg.slew, q, b)


def plan_events(plan):
    return [event for m in plan for leg in m.legs for event in _leg_events(leg, m.target, m.start)]


def _clip(event, cutoff):
    if event.begin >= cutoff - EPS:
        return None
    if event.end <= cutoff + EPS:
        return event
    if event.phase == "travel":
        trail, _ = split_path(event.path, SPEED * (cutoff - event.begin))
        return replace(event, end=cutoff, path=trail)
    angles = _approach((event.pan0, event.tilt0), (event.pan1, event.tilt1), cutoff - event.begin)
    return replace(event, end=cutoff, pan1=float(angles[0]), tilt1=float(angles[1]))


def _continue_leg(leg, cutoff):
    if cutoff <= leg.arrive - EPS:
        _, remaining = split_path(leg.path, SPEED * (cutoff - leg.depart))
        return replace(leg, path=remaining, depart=cutoff), "travel"
    pan, tilt = _approach((0., 0.), (leg.pan, leg.tilt), cutoff - leg.arrive)
    mode = "slew" if cutoff < leg.ready - EPS else "wait"
    return (replace(leg, path=leg.path[-1:], depart=cutoff, arrive=cutoff, ready=max(cutoff, leg.ready),
                    initial_pan=float(pan), initial_tilt=float(tilt)), mode)


def replay(nominal, scenario):
    """Replay only precommitted commands. Locked observations finish on survivors."""
    cutoff, failure = scenario.cutoff, scenario.failure
    prefix, successes = [], {}
    positions = STARTS.copy()
    release = [cutoff] * len(positions)
    occupied = set()
    engaged = {}
    blocked = set()
    for mission in nominal:
        possible = all(leg.vehicle not in blocked and (leg.vehicle != failure or mission.start < DISTURBANCE - EPS)
                       for leg in mission.legs)
        locked = possible and mission.start < cutoff - EPS
        if locked:
            successes[mission.target] = mission.start
        for leg in mission.legs:
            v = leg.vehicle
            if v in blocked:
                continue
            end_at = DISTURBANCE if v == failure else cutoff
            original = list(_leg_events(leg, mission.target, mission.start if locked else None))
            if locked and v != failure:
                prefix.extend(original)
                positions[v] = leg.path[-1]
                release[v] = max(cutoff, mission.start + OBSERVE + leg.slew)
                if release[v] > cutoff + EPS:
                    occupied.add(v)
            else:
                prefix.extend(piece for item in original if (piece := _clip(item, end_at)) is not None)
                if not locked and leg.depart < end_at - EPS:
                    current, mode = _continue_leg(leg, end_at)
                    positions[v] = current.path[0]
                    if v != failure:
                        engaged[v] = (mission.target, current, mode)
                    if mode == "wait" and end_at > leg.ready + EPS:
                        angles = (leg.pan, leg.tilt)
                        prefix.append(_point_event(v, mission.target, "wait_sync", leg.ready, end_at,
                                                   leg.path[-1], angles, angles))
                elif locked and v == failure:
                    positions[v] = leg.path[-1]
        if not possible:
            blocked.update(leg.vehicle for leg in mission.legs)
    if failure is not None:
        release[failure] = np.inf
        prefix.append(_point_event(failure, "", "failed", DISTURBANCE, DISTURBANCE, positions[failure]))
    for v in occupied:
        engaged.pop(v, None)
    return prefix, successes, positions, release, engaged


def _abort(v, engaged, position, cutoff):
    if v not in engaged:
        return cutoff, []
    target, remaining, mode = engaged[v]
    if mode == "travel":
        angles = (0., 0.)
    else:
        angles = (remaining.initial_pan, remaining.initial_tilt)
    reset_time = max(map(abs, angles)) / ANGULAR_SPEED
    events = [_point_event(v, target, "abort", cutoff, cutoff, position, angles, angles)]
    if reset_time > EPS:
        events.append(_point_event(v, target, "abort_reset", cutoff, cutoff + reset_time, position, angles))
    return cutoff + reset_time, events


def retain(space, nominal, scenario, successes, positions, release, engaged):
    """Keep feasible original assignments; cancel failed pairs after recovery."""
    fixed = [m for m in nominal if m.target not in successes and all(leg.vehicle != scenario.failure for leg in m.legs)]
    first = {}
    for mission in fixed:
        for leg in mission.legs:
            first.setdefault(leg.vehicle, mission.target)
    use_continuation = {v for v, (name, _, _) in engaged.items() if first.get(v) == name}
    x, available = positions.copy(), release.copy()
    events = []
    starts = dict(successes)
    for v in engaged.keys() - use_continuation:
        available[v], aborted = _abort(v, engaged, x[v], scenario.cutoff)
        events.extend(aborted)
    for m in fixed:
        legs, continuation = [], {}
        for old in m.legs:
            v = old.vehicle
            if v in use_continuation:
                _, leg, continuation[v] = engaged[v]
                use_continuation.remove(v)
            else:
                leg = make_leg(space.router, v, x[v], old.path[-1], available[v], space.targets[m.target])
            legs.append(leg)
        start = max(scenario.cutoff, *(leg.ready for leg in legs))
        starts[m.target] = start
        for leg in legs:
            v = leg.vehicle
            events.extend(_leg_events(leg, m.target, start, continuation.get(v)))
            x[v] = leg.path[-1]
            available[v] = start + OBSERVE + leg.slew
    return events, starts, x, available


def _schedule(space, targets, x, available, cutoff, width):
    initial = State(space.origins(x), tuple(available), latest=cutoff)
    solution = solve(space, initial, targets, width=width, limit=18, branch=3, deadline=WINDOW)
    plan, _, _ = build_plan(space, initial, solution)
    return plan_events(plan), {m.target: m.start for m in plan}


def score(starts, total):
    in_window = [time for time in starts.values() if time <= WINDOW + EPS]
    return len(in_window) / total, max(in_window, default=0.) if len(in_window) == total else WINDOW


def run_scenario(space, nominal, scenario, width=8, *, repair=False):
    if scenario.kind == "nominal":
        original = {m.target: m.start for m in nominal}
        names = ("retain", "retain_plus", "replan_raw", "replan")
        if repair:
            names = (*names, "repair")
        return {name: (original, plan_events(nominal)) for name in names}
    prefix, locked, positions, release, engaged = replay(nominal, scenario)
    fixed_events, fixed, x, available = retain(space, nominal, scenario, locked, positions, release, engaged)
    pending = [m.target for m in nominal if m.target not in locked]
    extended, extension = [], {}
    if scenario.new:
        pending.append(scenario.new.name)
        extended, extension = _schedule(space, [scenario.new.name], x, available, scenario.cutoff, width)

    free, cancelled = release.copy(), []
    for v in engaged:
        free[v], actions = _abort(v, engaged, positions[v], scenario.cutoff)
        cancelled.extend(actions)
    updates, replanned = _schedule(space, pending, positions, free, scenario.cutoff, width)
    adapt = ({**locked, **replanned}, prefix + cancelled + updates)
    strict = (fixed, prefix + fixed_events)
    append = ({**fixed, **extension}, strict[1] + extended)
    total = len(nominal) + int(scenario.new is not None)
    key = lambda s: (-score(s[0], total)[0], score(s[0], total)[1])
    alternatives = {"retain": strict, "retain_plus": append, "replan_raw": adapt}
    compared = [append]
    if repair:
        # Retain healthy pairs and original orders. Only unfixed jobs can be rebid.
        disrupted = [name for name in pending if name not in fixed]
        repairs, restored = _schedule(space, disrupted, x, available, scenario.cutoff, width)
        repaired = ({**fixed, **restored}, strict[1] + repairs)
        alternatives["repair"] = repaired
        compared.append(repaired)
    alternatives["replan"] = min([*compared, adapt], key=key)
    return alternatives
