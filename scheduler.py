"""Finite-candidate schedule search and full schedule reconstruction."""
from dataclasses import dataclass
from itertools import combinations

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
        self.exhaustive_fallback = max_rmse is not None or min_angle != MIN_ANGLE or los_margin > 0.
        axis = np.unique(np.r_[RADIUS, np.arange(grid, 100., grid), 100. - RADIUS])
        coordinates = np.column_stack([a.ravel() for a in np.meshgrid(axis, axis)])
        phi = np.deg2rad(np.arange(0., 360., 15.))
        direction = np.column_stack((np.cos(phi), np.sin(phi)))
        rings = [t.xy + factor * t.min_range * direction for t in targets for factor in (1., 1.5)]
        points = np.unique(np.vstack([coordinates, anchors, *rings]), axis=0)
        free = router.free_points(points)
        visible = {}
        for name, target in self.targets.items():
            if max_rmse is None:
                visible[name] = _visible_with_margin(points, target, los_margin)
            else:
                distance = np.linalg.norm(target.xy - points, axis=1)
                close = (AZIMUTH_SD * np.hypot(distance, target.min_range) <= max_rmse + EPS) & free
                mask = np.zeros(len(points), dtype=bool)
                mask[close] = _visible_with_margin(points[close], target, los_margin)
                visible[name] = mask
        keep = free & np.any(tuple(visible.values()), axis=0)
        self.points = points[keep]
        self.station_ids = {name: np.flatnonzero(good[keep]) for name, good in visible.items()}
        self.slews = {}
        self.compatible = {}
        for name, ids in self.station_ids.items():
            stations = self.points[ids]
            target = self.targets[name]
            self.slews[name] = pointing(stations, target)[2]
            u = target.xy - stations
            distance = np.linalg.norm(u, axis=1)
            u /= distance[:, None]
            sine = np.abs(u[:, [0]] * u[:, 1] - u[:, [1]] * u[:, 0])
            dx = stations[:, [0]] - stations[:, 0]
            dy = stations[:, [1]] - stations[:, 1]
            sine_floor = np.sin(np.deg2rad(min_angle))
            if min_angle != MIN_ANGLE:
                sine_floor -= EPS
            compatible = (sine >= sine_floor) & (dx * dx + dy * dy >= (2 * RADIUS - EPS) ** 2)
            if max_rmse is not None:
                rmse_numerator = AZIMUTH_SD * np.hypot(distance[:, None], distance)
                compatible &= rmse_numerator <= (max_rmse + EPS) * sine
            self.compatible[name] = compatible
        self.travel = self._travel(self.points, self.points)
        self.min_ready = {}

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

    def lower_ready(self, target, positions, releases):
        if target not in self.min_ready:
            ids = self.station_ids[target]
            self.min_ready[target] = (self.travel[:, ids] + self.slews[target]).min(axis=1, initial=np.inf)
        return float(np.partition(self.min_ready[target][list(positions)] + releases, 1)[1])


def _options(space, state, target, pair, limit, branch, nearest):
    ids = space.station_ids[target]
    if not len(ids):
        return
    i, j = pair
    drive = space.travel[np.ix_([state.positions[i], state.positions[j]], ids)]
    ready = drive + space.slews[target] + np.array([state.releases[i], state.releases[j]])[:, None]
    if nearest or limit is None:
        a = b = np.arange(len(ids))
    else:
        a, b = (np.argsort(row, kind="stable")[:limit] for row in ready)
    feasible = space.compatible[target][np.ix_(a, b)]
    start = np.maximum(np.maximum(ready[0, a, None], ready[1, b]), state.latest)
    if nearest:
        rank = drive[0, a, None] + drive[1, b]
    else:
        slew = space.slews[target]
        rank = start + 0.04 * (slew[a, None] + slew[b]) + 0.02 * (ready[0, a, None] + ready[1, b])
    rank = np.where(feasible, rank, np.inf)
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
    for vehicle, station, slew in zip(choice.vehicles, choice.stations, slews):
        positions[vehicle] = station
        releases[vehicle] = choice.start + OBSERVE + slew
    return State(tuple(positions), tuple(releases), (*state.choices, choice), choice.start)


def solve(space, initial, targets, width=12, limit=12, branch=2, deadline=60., nearest=False):
    """Beam search by completed-alert subset; earliest start for each chosen pair."""
    targets = tuple(target for target in targets if space.compatible[target].any())
    pairs = tuple(combinations([i for i, time in enumerate(initial.releases) if np.isfinite(time)], 2))
    if nearest:
        state, pending = initial, list(targets)
        while pending:
            moves = [move for target in pending for pair in pairs
                     for move in _options(space, state, target, pair, limit, 1, True)
                     if move[1].start <= deadline + EPS]
            if not moves:
                break
            _, choice, slews = min(moves, key=lambda move: (move[0], move[1].start, move[1].target))
            state = _advance(state, choice, slews)
            pending.remove(choice.target)
        return state

    beams = {0: [initial]}
    best = initial
    for mask in range(1 << len(targets)):
        current = beams.pop(mask, [])
        if not current:
            continue
        pending = [(n, target) for n, target in enumerate(targets) if not (mask & (1 << n))]
        states = current if mask == 0 else [state for _, state in current]
        for state in states:
            if (-len(state.choices), state.latest) < (-len(best.choices), best.latest):
                best = state
            for n, target in pending:
                new_mask = mask | (1 << n)
                remaining = [name for m, name in pending if m != n]
                heap = beams.setdefault(new_mask, [])
                for pair in pairs:
                    options = tuple(_options(space, state, target, pair, limit, branch, False))
                    if not options:
                        options = tuple(_options(space, state, target, pair, max(3 * limit, 40), branch, False))
                    if not options and space.exhaustive_fallback:
                        options = tuple(_options(space, state, target, pair, None, branch, False))
                    for _, choice, slews in options:
                        if choice.start > deadline + EPS:
                            continue
                        nxt = _advance(state, choice, slews)
                        optimistic = max([nxt.latest, *(space.lower_ready(t, nxt.positions, nxt.releases) for t in remaining)])
                        score = (optimistic, nxt.latest, sum(x for x in nxt.releases if np.isfinite(x)))
                        heap.append((score, nxt))
                if len(heap) > width:
                    heap.sort(key=lambda item: item[0])
                    signatures = set()
                    selected = []
                    for score, nxt in heap:
                        signature = (nxt.positions, tuple(round(r, 8) for r in nxt.releases))
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


def build_plan(space, initial, solution):
    positions = space.points[list(initial.positions)].copy()
    releases = list(initial.releases)
    plan = []
    clock = initial.latest
    for choice in solution.choices:
        target = space.targets[choice.target]
        legs = tuple(make_leg(space.router, v, positions[v], space.points[p], releases[v], target)
                     for v, p in zip(choice.vehicles, choice.stations))
        clock = max(clock, *(leg.ready for leg in legs))
        plan.append(Mission(choice.target, clock, legs))
        for leg in legs:
            positions[leg.vehicle] = leg.path[-1]
            releases[leg.vehicle] = clock + OBSERVE + leg.slew
    return plan, positions, tuple(releases)


def evaluate_fixed(router, targets, layout, origins, availability, *,
                   max_rmse=None, min_angle=MIN_ANGLE, los_margin=0.):
    """Recompute earliest times for fixed vehicle pairs and continuous stations."""
    positions = np.asarray(origins).copy()
    releases = list(availability)
    plan = []
    clock = 0.
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
        clock = max(clock, *(leg.ready for leg in legs))
        plan.append(Mission(name, clock, legs))
        for leg in legs:
            positions[leg.vehicle] = leg.path[-1]
            releases[leg.vehicle] = clock + OBSERVE + leg.slew
    return plan


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


def single_reuse(space, initial):
    """Exactly solve the finite-grid subproblem with one reused vehicle."""
    from itertools import permutations

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
        times = space.travel[np.ix_(initial.positions, ids)] + space.slews[name]
        times = times + np.asarray(initial.releases)[:, None]
        ready[name] = times
        for v in vehicles:
            order = np.argsort(times[v], kind="stable")
            possible = space.compatible[name][:, order]
            first = order[np.argmax(possible, axis=1)]
            partner_station[name, v] = first
            partner_time[name, v] = np.where(possible.any(axis=1), times[v, first], np.inf)
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
