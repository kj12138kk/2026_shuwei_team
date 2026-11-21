"""Discrete-event replay, failure/communication scenarios and replanning."""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations, product
from time import perf_counter

import numpy as np

from exact_schedule import seed_from_layout, solve_exact
from geometry import ANGULAR_SPEED, EPS, OBSERVE, SPEED, STARTS, Target, split_path
from scheduler import Continuation, State, build_plan, make_leg, solve

DISTURBANCE = 5.
WINDOW = 60.
POLICIES = ("retain", "retain_plus", "repair", "replan_raw", "replan")


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


@dataclass(frozen=True)
class PolicyResult:
    starts: dict
    events: list
    source: str
    kept: tuple = ()
    continued: tuple = ()


def _started_at_cutoff(time, cutoff, inclusive):
    return time <= cutoff + EPS if inclusive else time < cutoff - EPS


def locked_at_command(starts, scenario):
    if scenario.kind == "nominal":
        return {}
    return {name: time for name, time in starts.items()
            if _started_at_cutoff(time, scenario.cutoff, scenario.recovery is not None)}


def _new_alerts(ranges):
    return tuple(Target(f"N{theta:03d}_{r}", theta, r, 50.) for theta, r in product(range(0, 360, 45), ranges))


def scenarios():
    discovery = _new_alerts((30, 40, 50))
    out = [Scenario("nominal", "nominal")]
    for new_group, fail_group, out_group in product((False, True), repeat=3):
        if not any((new_group, fail_group, out_group)):
            continue
        kind = "+".join(name for name, flag in zip(("new", "failure", "outage"), (new_group, fail_group, out_group)) if flag)
        triple = new_group and fail_group and out_group
        alerts = (tuple(t for t in discovery if t.distance == 50.) if triple else discovery) if new_group else (None,)
        failures = range(len(STARTS)) if fail_group else (None,)
        recoveries = ((10., 20., 30.) if triple else (10., 15., 20., 25., 30.)) if out_group else (None,)
        for alert, failure, recovery in product(alerts, failures, recoveries):
            key = "_".join(str(value) for value in (alert.name if alert else None,
                                                   f"V{failure + 1}" if failure is not None else None,
                                                   f"R{int(recovery)}" if recovery else None) if value)
            out.append(Scenario(key, kind, alert, failure, recovery))
    return discovery, out


def pressure_scenarios():
    """Cross all failure identities and eight edge/far-field bearings at late recovery."""
    targets = _new_alerts((50,))
    design = [Scenario(f"{target.name}_V{failure + 1}_R30", "new+failure+outage",
                       target, failure, 30.)
              for target, failure in product(targets, range(len(STARTS)))]
    return targets, design


def _point_event(vehicle, target, phase, begin, end, point, a=(0., 0.), b=(0., 0.)):
    return Event(vehicle, target, phase, begin, end, np.asarray([point]), *a, *b)


def _approach(a, b, elapsed):
    delta = np.asarray(b) - a
    return np.asarray(a) + np.sign(delta) * np.minimum(np.abs(delta), ANGULAR_SPEED * max(0., elapsed))


def _leg_events(leg, target, start=None):
    v, p, q = leg.vehicle, leg.path[0], leg.path[-1]
    a, b = (leg.initial_pan, leg.initial_tilt), (leg.pan, leg.tilt)
    if not leg.continued:
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
    if event.phase == "arrive" and abs(event.begin - cutoff) <= EPS:
        return event
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
        return replace(leg, path=remaining, depart=cutoff, continued=True)
    pan, tilt = _approach((0., 0.), (leg.pan, leg.tilt), cutoff - leg.arrive)
    return replace(leg, path=leg.path[-1:], depart=cutoff, arrive=cutoff, ready=max(cutoff, leg.ready),
                   initial_pan=float(pan), initial_tilt=float(tilt), continued=True)


def replay(nominal, scenario, *, through_window=False):
    """At recovery, ready old pairs start before commands; at t=5, failure acts first."""
    cutoff = WINDOW if through_window else scenario.cutoff
    inclusive = through_window or scenario.recovery is not None
    failure = scenario.failure
    prefix, successes = [], {}
    positions = STARTS.copy()
    release = [cutoff] * len(positions)
    engaged = {}
    blocked = set()
    failed_angles = (0., 0.)
    for mission in nominal:
        possible = all(leg.vehicle not in blocked and (leg.vehicle != failure or mission.start < DISTURBANCE - EPS)
                       for leg in mission.legs)
        locked = possible and _started_at_cutoff(mission.start, cutoff, inclusive)
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
            else:
                pieces = [piece for item in original if (piece := _clip(item, end_at)) is not None]
                prefix.extend(pieces)
                if v == failure and pieces:
                    last = pieces[-1]
                    positions[v] = last.path[-1]
                    failed_angles = (last.pan1, last.tilt1)
                if not locked and leg.depart < end_at - EPS:
                    current = _continue_leg(leg, end_at)
                    positions[v] = current.path[0]
                    if v != failure:
                        engaged[v] = (mission.target, current)
                    if end_at > leg.ready + EPS:
                        angles = (leg.pan, leg.tilt)
                        prefix.append(_point_event(v, mission.target, "wait_sync", leg.ready, end_at,
                                                    leg.path[-1], angles, angles))
        if not possible:
            blocked.update(leg.vehicle for leg in mission.legs)
    if failure is not None:
        release[failure] = np.inf
        prefix.append(_point_event(failure, "", "failed", DISTURBANCE, DISTURBANCE,
                                   positions[failure], failed_angles, failed_angles))
    return prefix, successes, positions, release, engaged


def _abort(v, ongoing, cutoff):
    target, remaining = ongoing
    position = remaining.path[0]
    angles = (remaining.initial_pan, remaining.initial_tilt)
    reset_time = max(map(abs, angles)) / ANGULAR_SPEED
    events = [_point_event(v, target, "abort", cutoff, cutoff, position, angles, angles)]
    if reset_time > EPS:
        events.append(_point_event(v, target, "abort_reset", cutoff, cutoff + reset_time, position, angles))
    return cutoff + reset_time, events


def _keep(space, nominal, keep, cutoff, positions, release, engaged, *, allow_continue=True):
    """Reserve the retained jobs at their original stations and in each original queue."""
    x, available = positions.copy(), release.copy()
    remaining = engaged.copy()
    events, starts, continued = [], {}, []
    for m in nominal:
        if m.target not in keep:
            continue
        legs = []
        for old in m.legs:
            v = old.vehicle
            ongoing = remaining.pop(v, None)
            if ongoing is not None and ongoing[0] == m.target and allow_continue:
                leg = ongoing[1]
                continued.append((v, m.target))
            else:
                if ongoing is not None:
                    available[v], aborted = _abort(v, ongoing, cutoff)
                    events.extend(aborted)
                leg = make_leg(space.router, v, x[v], old.path[-1], available[v], space.targets[m.target])
            legs.append(leg)
        start = max(cutoff, *(leg.ready for leg in legs))
        starts[m.target] = start
        for leg in legs:
            v = leg.vehicle
            events.extend(_leg_events(leg, m.target, start))
            x[v] = leg.path[-1]
            available[v] = start + OBSERVE + leg.slew
    return events, starts, x, available, remaining, continued


def _schedule(space, targets, x, release, engaged, cutoff, width, *, allow_continue, limit=18, branch=3,
              stats=None, exact_budget=None, fixed=None, retained_first=False, seed_layout=None):
    available = release.copy()
    aborts, continuations = {}, {}
    for v, ongoing in engaged.items():
        available[v], aborts[v] = _abort(v, ongoing, cutoff)
        name, leg = ongoing
        if allow_continue and name in targets:
            station, = space.origins([leg.path[-1]])
            if station not in space.station_ids[name]:
                raise ValueError("A continuation station is missing from the candidate set.")
            continuations[v] = Continuation(name, station, leg)
    plan = []
    if targets:
        initial = State(space.origins(x), tuple(available), latest=cutoff)
        if exact_budget is None:
            solution = solve(space, initial, targets, width=width, limit=limit, branch=branch, deadline=WINDOW,
                             continuations=continuations, stats=stats)
        else:
            seed = None
            if not fixed:
                if seed_layout:
                    seed = seed_from_layout(space, initial, seed_layout, continuations, WINDOW)
                if seed is None:
                    seed = solve(space, initial, targets, width=width, limit=limit, branch=branch,
                                 deadline=WINDOW, continuations=continuations)
            solution, certificate = solve_exact(space, initial, targets, continuations=continuations, fixed=fixed,
                                                retained_first=retained_first, incumbent=seed,
                                                deadline=WINDOW, state_budget=exact_budget)
            if stats is not None:
                stats.update(certificate)
            if solution is None:
                return None
        plan, _, _ = build_plan(space, initial, solution, continuations=continuations)
    continued = [(leg.vehicle, m.target) for m in plan for leg in m.legs if leg.continued]
    holding = {v for v, _ in continued}
    events = [e for v, actions in aborts.items() if v not in holding for e in actions] + plan_events(plan)
    return events, {m.target: m.start for m in plan}, continued


def score(starts, total):
    in_window = [time for time in starts.values() if time <= WINDOW + EPS]
    return len(in_window) / total, max(in_window, default=0.) if len(in_window) == total else WINDOW


def _rank(result, original, total):
    eta, last = score(result.starts, total)
    delays = [max(0., result.starts[name] - start) for name, start in original.items()
              if result.starts.get(name, np.inf) <= WINDOW + EPS]
    return -eta, round(last, 9), -len(delays), round(sum(delays), 9)


def run_scenario(space, nominal, scenario, width=12, *, limit=18, branch=3, diagnostics=None, search_stats=None):
    if scenario.kind == "nominal":
        original = PolicyResult({m.target: m.start for m in nominal}, plan_events(nominal), "nominal")
        return dict.fromkeys(POLICIES, original)
    setup_start = perf_counter()
    prefix, locked, positions, release, engaged = replay(nominal, scenario)
    space = space.with_current_stations(positions[np.isfinite(release)])
    new = (scenario.new.name,) if scenario.new else ()
    pending = tuple(m.target for m in nominal if m.target not in locked) + new
    healthy = tuple(m.target for m in nominal if m.target not in locked and
                    all(leg.vehicle != scenario.failure for leg in m.legs))
    cutoff = scenario.cutoff
    strict_events, strict_starts, *_ = replay(nominal, scenario, through_window=True)
    retain = PolicyResult(strict_starts, strict_events, "retain")
    reserved = {}

    def variant(source, keep, targets, allow_continue=True, stats=None):
        key = keep, allow_continue
        if key not in reserved:
            reserved[key] = _keep(space, nominal, keep, cutoff, positions, release, engaged,
                                  allow_continue=allow_continue)
        events, starts, x, available, ongoing, continued = reserved[key]
        updates, assigned, preserved = _schedule(space, targets, x, available, ongoing, cutoff, width,
                                                 allow_continue=allow_continue, limit=limit, branch=branch, stats=stats)
        return PolicyResult({**locked, **starts, **assigned}, prefix + events + updates, source, keep,
                            tuple(sorted(continued + preserved)))

    def subsets():
        return (keep for n in range(len(healthy) + 1) for keep in combinations(healthy, n))

    total = len(nominal) + len(new)
    original = {m.target: m.start for m in nominal}

    def rank(result):
        return _rank(result, original, total)

    now = perf_counter()
    collect = diagnostics is not None or search_stats is not None
    stats = {} if collect else None
    append = variant("retain_plus", healthy, new, stats=stats)
    disrupted = tuple(name for name in pending if name not in healthy)
    repair = variant("repair", healthy, disrupted, stats=stats) if disrupted != new else replace(append, source="repair")
    raw_time, raw_stats = perf_counter(), {} if collect else None
    raw = variant("replan_raw", (), pending, allow_continue=False, stats=raw_stats)
    raw_time = perf_counter() - raw_time
    options = [retain, append, repair, raw]
    progress, progress_stats, progress_time = repair, None, 0.
    for keep in subsets():
        if keep == healthy and keep:
            continue
        if not keep:
            tick = perf_counter()
            progress_stats = {} if collect else None
            progress = variant("adaptive", (), pending, stats=progress_stats)
            progress_time = perf_counter() - tick
            options.append(progress)
        else:
            options.append(variant("partial", keep, tuple(name for name in pending if name not in keep), stats=stats))

    best = min(options, key=rank)
    if collect:
        searched = ("expanded_states", "generated_states")

        def combined_counts(*parts):
            return {key: sum((p or {}).get(key, 0) for p in parts) for key in searched}

        counts = combined_counts(stats, raw_stats, progress_stats)
        seconds = perf_counter() - now
        if search_stats is not None:
            search_stats.update(counts, setup_s=now - setup_start, seconds=seconds)
    if diagnostics is not None:
        # Compare the replanners before the strict-retain fallback. All three
        # enhancements can keep A's feasible restart result as an incumbent.
        def record(name, outcome, elapsed, counters):
            diagnostics[name] = (outcome, dict(seconds=elapsed, **(counters or {})))

        record("A_restart", raw, raw_time, raw_stats)
        record("B_progress", min((raw, progress), key=rank), raw_time + progress_time,
               combined_counts(raw_stats, progress_stats))
        record("D_both", min(options[1:], key=rank), seconds, counts)
        tick, no_progress_stats = perf_counter(), {}
        no_progress = [raw]
        if disrupted != new:
            no_progress.append(variant("append_without_progress", healthy, new, allow_continue=False,
                                       stats=no_progress_stats))
        for keep in subsets():
            if keep:
                no_progress.append(variant("retain_without_progress", keep,
                                           tuple(name for name in pending if name not in keep),
                                           allow_continue=False, stats=no_progress_stats))
        record("C_retention", min(no_progress, key=rank), perf_counter() - tick + raw_time,
               combined_counts(no_progress_stats, raw_stats))
    return dict(zip(POLICIES, (retain, append, repair, raw, best)))


def exact_scenario(space, nominal, scenario, *, state_budget=20_000, keep=(), retained_first=False, incumbent=None):
    """Solve a small scenario on the full station pool, optionally with fixed original visits."""
    tick = perf_counter()
    prefix, locked, positions, release, engaged = replay(nominal, scenario)
    space = space.with_current_stations(positions[np.isfinite(release)])
    pending = tuple(m.target for m in nominal if m.target not in locked)
    if scenario.new is not None:
        pending += (scenario.new.name,)
    fixed = {}
    for mission in nominal:
        if mission.target not in keep:
            continue
        vehicles = tuple(leg.vehicle for leg in mission.legs)
        if mission.target not in pending or scenario.failure in vehicles:
            raise ValueError("Only an outstanding healthy assignment can be retained.")
        fixed[mission.target] = (vehicles, space.origins([leg.path[-1] for leg in mission.legs]))
    seed_layout = []
    if incumbent is not None and not fixed:
        for name, start in sorted(incumbent.starts.items(), key=lambda item: (item[1], item[0])):
            if name in pending and start <= WINDOW + EPS:
                pair = [e for e in incumbent.events if e.phase == "observe" and e.target == name]
                if len(pair) != 2 or pair[0].vehicle == pair[1].vehicle:
                    raise ValueError("A seed observation must have two distinct vehicles.")
                seed_layout.append((name, tuple(e.vehicle for e in pair), space.origins([e.path[0] for e in pair])))
    setup = perf_counter() - tick
    stats = {}
    tick = perf_counter()
    updates = _schedule(space, pending, positions, release, engaged, scenario.cutoff, 12,
                        allow_continue=True, stats=stats, exact_budget=state_budget,
                        fixed=fixed, retained_first=retained_first, seed_layout=seed_layout)
    stats["seconds"] = perf_counter() - tick
    stats["setup_s"] = setup
    if not pending:
        stats.update(visited_states=0, expanded_states=0, dominance_prunes=0, bound_prunes=0,
                     complete=True, feasible=True, confirmed_on_pool=True, infeasible_on_pool=False,
                     termination="no_pending")
    if updates is None:
        return None, stats
    events, starts, continued = updates
    return PolicyResult({**locked, **starts}, prefix + events, "exact", tuple(fixed), tuple(continued)), stats
