"""Finite-candidate schedule search and full schedule reconstruction."""
from copy import copy
from dataclasses import dataclass, replace
from itertools import combinations, permutations

import numpy as np

from geometry import EPS, OBSERVE, RADIUS, SPEED, los_clearance, observable, pair_geometry, path_length, pointing

MIN_ANGLE = 0.01  # Exclude non-identifiable pairs, with a small numerical guard.
AZIMUTH_SD = np.deg2rad(1.)


def _visible_with_margin(points, target, los_margin):
    visible = observable(points, target)
    if los_margin > 0. and np.any(visible):
        visible[visible] = los_clearance(points[visible], target) >= los_margin - EPS
    return visible


@dataclass(frozen=True)
class Choice:
    target: str
    vehicles: tuple
    stations: tuple
    start: float


@dataclass(frozen=True)
class State:
    positions: tuple
    releases: tuple
    choices: tuple = ()
    latest: float = 0.
    used: int = 0


@dataclass
class Leg:
    vehicle: int
    path: np.ndarray
    depart: float
    arrive: float
    ready: float
    pan: float
    tilt: float
    slew: float
    initial_pan: float = 0.
    initial_tilt: float = 0.
    continued: bool = False


@dataclass(frozen=True)
class Continuation:
    target: str
    station: int
    leg: Leg


@dataclass
class Mission:
    target: str
    start: float
    legs: tuple


class Space:
    def __init__(self, router, targets, grid, anchors, *, max_rmse=None, min_angle=MIN_ANGLE, los_margin=0.):
        if (max_rmse is not None and not max_rmse > 0.) or not 0. < min_angle <= 90. or not los_margin >= 0.:
            raise ValueError("Invalid RMSE ceiling, minimum pair angle or line-of-sight margin.")
        self.router = router
        self.targets = {target.name: target for target in targets}
        self.max_rmse, self.min_angle, self.los_margin = max_rmse, min_angle, los_margin
        if grid is None:
            points = np.unique(anchors, axis=0)
        else:
            axis = np.unique(np.r_[RADIUS, np.arange(grid, 100., grid), 100. - RADIUS])
            coordinates = np.column_stack([a.ravel() for a in np.meshgrid(axis, axis)])
            phi = np.deg2rad(np.arange(0., 360., 15.))
            direction = np.column_stack((np.cos(phi), np.sin(phi)))
            rings = [t.xy + factor * t.min_range * direction for t in targets for factor in (1., 1.5)]
            points = np.unique(np.vstack([coordinates, anchors, *rings]), axis=0)
        free = router.free_points(points)
        visible = {name: self._station_mask(points, target, free) for name, target in self.targets.items()}
        keep = free & np.any(tuple(visible.values()), axis=0)
        self.points = points[keep]
        self.station_ids = {name: np.flatnonzero(good[keep]) for name, good in visible.items()}
        self.slews = {}
        self.compatible = {}
        for name, ids in self.station_ids.items():
            stations = self.points[ids]
            target = self.targets[name]
            self.slews[name] = pointing(stations, target)[2]
            self.compatible[name] = self._compatible(target, stations, stations)
        self.travel = self._travel(self.points, self.points)
        self.min_ready = {}

    def _station_mask(self, points, target, free):
        close = free.copy()
        if self.max_rmse is not None:
            distance = np.linalg.norm(target.xy - points, axis=1)
            close &= AZIMUTH_SD * np.hypot(distance, target.min_range) <= self.max_rmse + EPS
        close[close] = _visible_with_margin(points[close], target, self.los_margin)
        return close

    def _compatible(self, target, p, q):
        u, v = target.xy - p, target.xy - q
        r, s = np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1)
        u, v = u / r[:, None], v / s[:, None]
        sine = np.abs(u[:, [0]] * v[:, 1] - u[:, [1]] * v[:, 0])
        dx, dy = p[:, [0]] - q[:, 0], p[:, [1]] - q[:, 1]
        sine_floor = np.sin(np.deg2rad(self.min_angle)) - (EPS if self.min_angle != MIN_ANGLE else 0.)
        compatible = (sine >= sine_floor) & (dx * dx + dy * dy >= (2 * RADIUS - EPS) ** 2)
        if self.max_rmse is not None:
            compatible &= AZIMUTH_SD * np.hypot(r[:, None], s) <= (self.max_rmse + EPS) * sine
        return compatible

    def _travel(self, starts, ends):
        if not len(starts) or not len(ends):
            return np.empty((len(starts), len(ends)))
        return np.concatenate([self.router.distances(starts[i:i + 64], ends) / SPEED
                               for i in range(0, len(starts), 64)])

    def origins(self, points):
        ids = []
        new = []
        for point in points:
            all_points = np.vstack([self.points, *new])
            match = np.flatnonzero(np.max(np.abs(all_points - point), axis=1) < EPS)
            if len(match):
                ids.append(int(match[0]))
            else:
                ids.append(len(all_points))
                new.append(point)
        if new:
            self.points = np.vstack([self.points, *new])
            old = len(self.travel)
            lower = self._travel(self.points[old:], self.points)
            self.travel = np.block([[self.travel, lower[:, :old].T], [lower]])
            self.min_ready.clear()
        return tuple(ids)

    def with_current_stations(self, positions):
        """Give one scenario its own additional stops at the current vehicle positions."""
        result = copy(self)
        result.min_ready = {}
        result.station_ids = self.station_ids.copy()
        result.slews = self.slews.copy()
        result.compatible = self.compatible.copy()
        ids = np.unique(result.origins(positions))
        points = result.points[ids]
        free = self.router.free_points(points)
        for name, target in self.targets.items():
            old = self.station_ids[name]
            new = np.setdiff1d(ids[self._station_mask(points, target, free)], old)
            if not len(new):
                continue
            combined = np.union1d(old, new)
            a, b = np.searchsorted(combined, old), np.searchsorted(combined, new)
            compatible = np.empty((len(combined), len(combined)), dtype=bool)
            cross = self._compatible(target, result.points[new], result.points[combined])
            compatible[b] = cross
            compatible[:, b] = cross.T
            compatible[np.ix_(a, a)] = self.compatible[name]
            result.station_ids[name] = combined
            result.slews[name] = pointing(result.points[combined], target)[2]
            result.compatible[name] = compatible
        return result

    def lower_ready(self, target, state, continuations):
        if target not in self.min_ready:
            ids = self.station_ids[target]
            self.min_ready[target] = (self.travel[:, ids] + self.slews[target]).min(axis=1, initial=np.inf)
        ready = self.min_ready[target][list(state.positions)] + state.releases
        for v, ongoing in continuations.items():
            if not (state.used & (1 << v)) and ongoing.target == target:
                ready[v] = min(ready[v], ongoing.leg.ready)
        return float(np.partition(ready, 1)[1])


def _ready_times(space, state, target, vehicles, continuations):
    ids = space.station_ids[target]
    drive = space.travel[np.ix_([state.positions[v] for v in vehicles], ids)]
    ready = drive + space.slews[target] + np.asarray([state.releases[v] for v in vehicles])[:, None]
    saved = {}
    for row, v in enumerate(vehicles):
        ongoing = continuations.get(v)
        if ongoing is not None and ongoing.target == target and not (state.used & (1 << v)):
            ix = np.searchsorted(ids, ongoing.station)
            if ix < len(ids) and ids[ix] == ongoing.station:
                ready[row, ix] = ongoing.leg.ready
                saved[row] = ix
    return drive, ready, saved


def _options(space, state, target, pair, limit, branch, not_before, deadline, continuations,
             *, greedy=None, fixed_stations=None):
    ids = space.station_ids[target]
    if not len(ids) or not_before > deadline + EPS:
        return
    slew = space.slews[target]
    drive, ready, saved = _ready_times(space, state, target, pair, continuations)
    shortlist = []
    for row in range(2):
        prior = saved.get(row)
        if fixed_stations is not None:
            ix = np.searchsorted(ids, fixed_stations[row])
            if ix == len(ids) or ids[ix] != fixed_stations[row] or ready[row, ix] > deadline + EPS:
                return
            shortlist.append(np.array([ix]))
            continue
        available = np.flatnonzero(ready[row] <= deadline + EPS) if np.isfinite(deadline) else np.arange(len(ids))
        if not len(available):
            return
        if limit is not None and len(available) > limit:
            available = available[np.argsort(ready[row, available], kind="stable")[:limit]]
            if prior is not None and prior not in available and ready[row, prior] <= deadline + EPS:
                available = np.r_[available, prior]
        shortlist.append(available)
    a, b = shortlist
    feasible = space.compatible[target][np.ix_(a, b)]
    start = np.maximum(np.maximum(ready[0, a, None], ready[1, b]), not_before)
    if greedy:
        distance = drive[0, a, None] + drive[1, b]
        primary, secondary = (distance, start) if greedy == "nearest" else (start, distance)
        ranked = np.where(feasible, primary, np.inf)
        best = ranked.min()
        if not np.isfinite(best):
            return
        equal = np.flatnonzero(ranked.ravel() <= best + (EPS if greedy == "nearest" else 0.))
        index = equal[np.argmin(secondary.ravel()[equal])]
        x, y = np.unravel_index(index, ranked.shape)
        yield (float(distance[x, y]), Choice(target, pair, (int(ids[a[x]]), int(ids[b[y]])), float(start[x, y])),
               (float(slew[a[x]]), float(slew[b[y]])))
        return
    if branch is None:
        # Complete enumeration for a finite-pool certificate; no shortlist ranking.
        rows, cols = np.nonzero(feasible & np.isfinite(start))
        for index in np.argsort(start[rows, cols], kind="stable"):
            x, y = rows[index], cols[index]
            yield (float(start[x, y]), Choice(target, pair, (int(ids[a[x]]), int(ids[b[y]])), float(start[x, y])),
                   (float(slew[a[x]]), float(slew[b[y]])))
        return
    rank = np.where(feasible, start + 0.04 * (slew[a, None] + slew[b]) +
                    0.02 * (ready[0, a, None] + ready[1, b]), np.inf)
    flat = rank.ravel()
    valid = np.flatnonzero(np.isfinite(flat))
    if not len(valid):
        return
    if len(valid) > branch:
        valid = valid[np.argpartition(flat[valid], branch - 1)[:branch]]
    for n in sorted(valid, key=lambda k: (flat[k], k)):
        x, y = np.unravel_index(n, rank.shape)
        yield (float(rank[x, y]), Choice(target, pair, (int(ids[a[x]]), int(ids[b[y]])), float(start[x, y])),
               (float(space.slews[target][a[x]]), float(space.slews[target][b[y]])))


def _advance(state, choice, slews):
    positions, releases = list(state.positions), list(state.releases)
    used = state.used
    for vehicle, station, slew in zip(choice.vehicles, choice.stations, slews):
        positions[vehicle] = station
        releases[vehicle] = choice.start + OBSERVE + slew
        used |= 1 << vehicle
    return State(tuple(positions), tuple(releases), (*state.choices, choice), max(state.latest, choice.start), used)


def solve(space, initial, targets, width=12, limit=12, branch=2, deadline=60., nearest=False, *,
          earliest=False, continuations=None, stats=None):
    """Beam search by assigned-alert subset; only shared vehicles impose precedence."""
    if nearest and earliest:
        raise ValueError("Select one greedy baseline.")
    continuations = continuations or {}
    hold_mask = sum(1 << v for v in continuations)
    targets = tuple(target for target in targets if space.compatible[target].any())
    pairs = tuple(combinations([i for i, time in enumerate(initial.releases) if np.isfinite(time)], 2))
    if nearest or earliest:
        criterion = "nearest" if nearest else "earliest"
        state, pending = initial, list(targets)
        while pending:
            moves = [move for target in pending for pair in pairs
                     for move in _options(space, state, target, pair, None, 1, initial.latest, deadline,
                                          continuations, greedy=criterion)]
            if not moves:
                break
            def key(move):
                distance, choice, _ = move
                score = (distance, choice.start) if nearest else (choice.start, distance)
                return (*score, choice.target, choice.vehicles, choice.stations)

            _, choice, slews = min(moves, key=key)
            state = _advance(state, choice, slews)
            pending.remove(choice.target)
        return replace(state, choices=tuple(sorted(state.choices, key=lambda c: (c.start, c.target))))

    beams = {0: [initial]}
    best = initial
    for mask in range(1 << len(targets)):
        current = beams.pop(mask, [])
        if not current:
            continue
        pending = [(n, target) for n, target in enumerate(targets) if not (mask & (1 << n))]
        states = current if mask == 0 else [state for _, state in current]
        for state in states:
            if stats is not None:
                stats["expanded_states"] = stats.get("expanded_states", 0) + 1
            if (-len(state.choices), state.latest) < (-len(best.choices), best.latest):
                best = state
            for n, target in pending:
                new_mask = mask | (1 << n)
                remaining = [name for m, name in pending if m != n]
                heap = beams.setdefault(new_mask, [])
                for pair in pairs:
                    args = (branch, initial.latest, deadline, continuations)
                    options = tuple(_options(space, state, target, pair, limit, *args))
                    if not options and limit is not None:
                        options = tuple(_options(space, state, target, pair, max(3 * limit, 40), *args))
                    if not options and limit is not None:
                        options = tuple(_options(space, state, target, pair, None, *args))
                    for _, choice, slews in options:
                        if stats is not None:
                            stats["generated_states"] = stats.get("generated_states", 0) + 1
                        nxt = _advance(state, choice, slews)
                        optimistic = max([nxt.latest, *(space.lower_ready(t, nxt, continuations) for t in remaining)])
                        score = (optimistic, nxt.latest, sum(x for x in nxt.releases if np.isfinite(x)))
                        heap.append((score, nxt))
                if len(heap) > width:
                    heap.sort(key=lambda item: item[0])
                    signatures = set()
                    selected = []
                    for score, nxt in heap:
                        signature = (nxt.positions, tuple(round(r, 8) for r in nxt.releases), nxt.used & hold_mask)
                        if signature in signatures:
                            continue
                        signatures.add(signature)
                        selected.append((score, nxt))
                        if len(selected) >= width:
                            break
                    heap[:] = selected
    return best


def make_leg(router, vehicle, origin, station, depart, target):
    path = router.path(origin, station)
    arrive = depart + path_length(path) / SPEED
    pan, tilt, slew = map(float, pointing(station, target))
    return Leg(vehicle, path, depart, arrive, arrive + slew, pan, tilt, slew)


def build_plan(space, initial, solution, *, continuations=None):
    positions = space.points[list(initial.positions)].copy()
    releases = list(initial.releases)
    pending = dict(continuations or {})
    plan = []
    for choice in solution.choices:
        target = space.targets[choice.target]
        legs = []
        for v, p in zip(choice.vehicles, choice.stations):
            ongoing = pending.pop(v, None)
            if ongoing is not None and ongoing.target == choice.target and ongoing.station == p:
                legs.append(ongoing.leg)
            else:
                legs.append(make_leg(space.router, v, positions[v], space.points[p], releases[v], target))
        start = max(initial.latest, *(leg.ready for leg in legs))
        if abs(start - choice.start) > 1e-7:
            raise ValueError("Reconstructed start time differs from the search.")
        plan.append(Mission(choice.target, start, tuple(legs)))
        for leg in legs:
            positions[leg.vehicle] = leg.path[-1]
            releases[leg.vehicle] = start + OBSERVE + leg.slew
    plan.sort(key=lambda m: (m.start, m.target))
    return plan, positions, tuple(releases)


def evaluate_fixed(router, targets, layout, origins, availability, *,
                   max_rmse=None, min_angle=MIN_ANGLE, los_margin=0.):
    """Recompute earliest times for fixed vehicle pairs and continuous stations."""
    positions = np.asarray(origins).copy()
    releases = list(availability)
    plan = []
    for name, vehicles, stations in layout:
        target = targets[name]
        stations = np.asarray(stations, dtype=float)
        if not np.all(_visible_with_margin(stations, target, los_margin)) or not np.all(router.free_points(stations)):
            return None
        geom = pair_geometry(*stations, target)
        angle_slack = 0. if min_angle == MIN_ANGLE else EPS
        if (geom["baseline_m"] < 2. * RADIUS - EPS or
                not min_angle - angle_slack <= geom["angle_deg"] <= 180. - min_angle + angle_slack):
            return None
        if max_rmse is not None and geom["linear_rmse_m"] > max_rmse + EPS:
            return None
        legs = tuple(make_leg(router, v, positions[v], station, releases[v], target) for v, station in zip(vehicles, stations))
        start = max(leg.ready for leg in legs)
        plan.append(Mission(name, start, legs))
        for leg in legs:
            positions[leg.vehicle] = leg.path[-1]
            releases[leg.vehicle] = start + OBSERVE + leg.slew
    return sorted(plan, key=lambda m: (m.start, m.target))


def refine(router, targets, plan, origins, availability, *, max_rmse=None, min_angle=MIN_ANGLE, los_margin=0.):
    """Feasibility-preserving coordinate pattern search on the fixed task sequence."""
    if not plan:
        return plan
    layout = [(m.target, tuple(leg.vehicle for leg in m.legs), np.array([leg.path[-1] for leg in m.legs])) for m in plan]
    constraints = dict(max_rmse=max_rmse, min_angle=min_angle, los_margin=los_margin)
    best = evaluate_fixed(router, targets, layout, origins, availability, **constraints)
    if best is None:
        return None

    def score(result):
        return (round(max(m.start for m in result), 9), round(sum(m.start for m in result), 9),
                sum(leg.arrive - leg.depart for m in result for leg in m.legs))

    directions = np.column_stack((np.cos(np.arange(8) * np.pi / 4.), np.sin(np.arange(8) * np.pi / 4.)))
    for step in (2., 1., .5, .25, .10, .05, .02, .01):
        for _ in range(3):
            improved = False
            for name, vehicles, stations in layout:
                for i in range(2):
                    original = stations[i].copy()
                    winner = original
                    for direction in directions:
                        stations[i] = original + step * direction
                        proposal = evaluate_fixed(router, targets, layout, origins, availability, **constraints)
                        if proposal is not None and score(proposal) < score(best):
                            best, winner, improved = proposal, stations[i].copy(), True
                    stations[i] = winner
            if not improved:
                break
    return best


def _first_ready(space, initial, name):
    ids = space.station_ids[name]
    return (space.travel[np.ix_(initial.positions, ids)] + space.slews[name] +
            np.asarray(initial.releases)[:, None])


def _earliest_partner(compatible, times):
    order = np.argsort(times, kind="stable")
    possible = compatible[:, order]
    first = order[np.argmax(possible, axis=1)]
    return first, np.where(possible.any(axis=1), times[first], np.inf)


def reuse_chain_lower_bound(space, initial):
    """Lower bound over all reuse patterns on the same three-alert candidate set."""
    names = tuple(space.targets)
    if len(names) != 3 or len(initial.positions) != 5 or any(not space.compatible[j].any() for j in names):
        return None
    ready = {j: _first_ready(space, initial, j) for j in names}
    other = {j: np.array([_earliest_partner(space.compatible[j], np.delete(ready[j], v, axis=0).min(axis=0))[1]
                          for v in range(5)]) for j in names}
    synchronized = {j: np.maximum(np.maximum(ready[j], other[j]), initial.latest) for j in names}
    chain = {}
    for j, k, third in permutations(names):
        transfer = space.travel[np.ix_(space.station_ids[j], space.station_ids[k])]
        release = synchronized[j] + OBSERVE + space.slews[j]
        independent = synchronized[third].min()
        chain[j + "->" + k] = [
            float(max(independent, np.maximum((first[:, None] + transfer).min(axis=0) + space.slews[k], mate).min()))
            for first, mate in zip(release, other[k])]
    return dict(lower_bound_s=min(min(values) for values in chain.values()), by_chain_vehicle_s=chain)


def single_reuse(space, initial):
    """Exactly solve the finite-grid subproblem with one reused vehicle."""

    names = tuple(space.targets)
    vehicles = tuple(v for v, time in enumerate(initial.releases) if np.isfinite(time))
    if len(names) != 3 or len(vehicles) != 5 or any(not space.compatible[name].any() for name in names):
        return None
    ready = {}
    partner_time = {}
    partner_station = {}
    independent = {}
    for name in names:
        ids = space.station_ids[name]
        times = _first_ready(space, initial, name)
        ready[name] = times
        for v in vehicles:
            partner_station[name, v], partner_time[name, v] = _earliest_partner(space.compatible[name], times[v])
        for v, w in combinations(vehicles, 2):
            simultaneous = np.maximum(np.maximum(times[v], partner_time[name, w]), initial.latest)
            p = int(np.argmin(simultaneous))
            if np.isfinite(simultaneous[p]):
                q = partner_station[name, w][p]
                independent[name, v, w] = Choice(name, (v, w), (int(ids[p]), int(ids[q])), float(simultaneous[p]))

    best = None
    upper = np.inf
    for j, k, unused in permutations(names):
        j_ids, k_ids = space.station_ids[j], space.station_ids[k]
        between = space.travel[np.ix_(j_ids, k_ids)]
        for i, u in permutations(vehicles, 2):
            first = np.maximum(np.maximum(ready[j][i], partner_time[j, u]), initial.latest)
            release = first + OBSERVE + space.slews[j]
            eligible = np.flatnonzero(np.isfinite(release) & (release + space.slews[k].min() <= upper + EPS))
            if not len(eligible):
                continue
            second = release[eligible, None] + between[eligible]
            best_p = eligible[np.argmin(second, axis=0)]
            i_second = np.min(second, axis=0) + space.slews[k]
            for v in set(vehicles) - {i, u}:
                w, z = sorted(set(vehicles) - {i, u, v})
                other = independent.get((unused, w, z))
                if other is None or other.start > upper + EPS:
                    continue
                k_start = np.maximum(i_second, partner_time[k, v])
                total = np.maximum(k_start, other.start)
                near = np.flatnonzero(np.isfinite(total) & (total <= min(upper, total.min()) + EPS))
                if not len(near):
                    continue
                q = min(near, key=lambda index: (total[index], k_start[index] + first[best_p[index]]))
                p = best_p[q]
                choices = (
                    Choice(j, (i, u), (int(j_ids[p]), int(j_ids[partner_station[j, u][p]])), float(first[p])),
                    Choice(k, (i, v), (int(k_ids[q]), int(k_ids[partner_station[k, v][q]])), float(k_start[q])),
                    other,
                )
                ordered = tuple(sorted(choices, key=lambda c: (c.start, c.target)))
                candidate = State(initial.positions, initial.releases, ordered, float(total[q]))
                if best is None or (round(candidate.latest, 9), sum(c.start for c in ordered)) < (
                        round(best.latest, 9), sum(c.start for c in best.choices)):
                    best = candidate
                    upper = candidate.latest
    return best
