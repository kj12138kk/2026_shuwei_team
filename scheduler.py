"""Finite-candidate schedule search and full schedule reconstruction."""
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from geometry import EPS, OBSERVE, RADIUS, SPEED, observable, pair_geometry, path_length, pointing

MIN_ANGLE = 0.01  # Exclude non-identifiable pairs, with a small numerical guard.


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
    def __init__(self, router, targets, grid, anchors):
        self.router = router
        self.targets = {target.name: target for target in targets}
        axis = np.unique(np.r_[RADIUS, np.arange(grid, 100., grid), 100. - RADIUS])
        coordinates = np.column_stack([a.ravel() for a in np.meshgrid(axis, axis)])
        phi = np.deg2rad(np.arange(0., 360., 15.))
        direction = np.column_stack((np.cos(phi), np.sin(phi)))
        rings = [t.xy + factor * t.min_range * direction for t in targets for factor in (1., 1.5)]
        points = np.unique(np.vstack([coordinates, anchors, *rings]), axis=0)
        free = router.free_points(points)
        visible = {name: observable(points, t) for name, t in self.targets.items()}
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
            u /= np.linalg.norm(u, axis=1)[:, None]
            cross = u[:, [0]] * u[:, 1] - u[:, [1]] * u[:, 0]
            dx = stations[:, [0]] - stations[:, 0]
            dy = stations[:, [1]] - stations[:, 1]
            self.compatible[name] = (np.abs(cross) >= np.sin(np.deg2rad(MIN_ANGLE))) & (dx * dx + dy * dy >= (2 * RADIUS - EPS) ** 2)
        self.travel = self._travel(self.points, self.points)
        self.min_ready = {}

    def _travel(self, starts, ends):
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
            self.min_ready[target] = (self.travel[:, ids] + self.slews[target]).min(axis=1)
        return float(np.partition(self.min_ready[target][list(positions)] + releases, 1)[1])


def _options(space, state, target, pair, limit, branch, nearest):
    ids = space.station_ids[target]
    i, j = pair
    drive = space.travel[np.ix_([state.positions[i], state.positions[j]], ids)]
    ready = drive + space.slews[target] + np.array([state.releases[i], state.releases[j]])[:, None]
    if nearest:
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
    targets = tuple(targets)
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


def evaluate_fixed(router, targets, layout, origins, availability):
    """Recompute earliest times for fixed vehicle pairs and continuous stations."""
    positions = np.asarray(origins).copy()
    releases = list(availability)
    plan = []
    clock = 0.
    for name, vehicles, stations in layout:
        target = targets[name]
        if not np.all(observable(stations, target)) or not np.all(router.free_points(stations)):
            return None
        geom = pair_geometry(*stations, target)
        if geom["baseline_m"] < 2. * RADIUS - EPS or not MIN_ANGLE <= geom["angle_deg"] <= 180. - MIN_ANGLE:
            return None
        legs = tuple(make_leg(router, v, positions[v], station, releases[v], target) for v, station in zip(vehicles, stations))
        clock = max(clock, *(leg.ready for leg in legs))
        plan.append(Mission(name, clock, legs))
        for leg in legs:
            positions[leg.vehicle] = leg.path[-1]
            releases[leg.vehicle] = clock + OBSERVE + leg.slew
    return plan


def refine(router, targets, plan, origins, availability):
    """Feasibility-preserving coordinate pattern search on the fixed task sequence."""
    layout = [(m.target, tuple(leg.vehicle for leg in m.legs), np.array([leg.path[-1] for leg in m.legs])) for m in plan]
    best = plan

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
                        proposal = evaluate_fixed(router, targets, layout, origins, availability)
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
    vehicles = tuple(range(len(initial.positions)))
    ready = {}
    partner_time = {}
    partner_station = {}
    independent = {}
    for name in names:
        ids = space.station_ids[name]
        times = space.travel[np.ix_(initial.positions, ids)] + space.slews[name]
        ready[name] = times
        for v in vehicles:
            order = np.argsort(times[v], kind="stable")
            possible = space.compatible[name][:, order]
            first = order[np.argmax(possible, axis=1)]
            partner_station[name, v] = first
            partner_time[name, v] = np.where(possible.any(axis=1), times[v, first], np.inf)
        for v, w in combinations(vehicles, 2):
            simultaneous = np.maximum(times[v], partner_time[name, w])
            p = int(np.argmin(simultaneous))
            q = partner_station[name, w][p]
            independent[name, v, w] = Choice(name, (v, w), (int(ids[p]), int(ids[q])), float(simultaneous[p]))

    best = None
    upper = np.inf
    for j, k, unused in permutations(names):
        j_ids, k_ids = space.station_ids[j], space.station_ids[k]
        between = space.travel[np.ix_(j_ids, k_ids)]
        for i, u in permutations(vehicles, 2):
            first = np.maximum(ready[j][i], partner_time[j, u])
            release = first + OBSERVE + space.slews[j]
            eligible = np.flatnonzero(release + space.slews[k].min() <= upper + EPS)
            if not len(eligible):
                continue
            second = release[eligible, None] + between[eligible]
            best_p = eligible[np.argmin(second, axis=0)]
            i_second = np.min(second, axis=0) + space.slews[k]
            for v in set(vehicles) - {i, u}:
                w, z = sorted(set(vehicles) - {i, u, v})
                other = independent[unused, w, z]
                if other.start > upper + EPS:
                    continue
                k_start = np.maximum(i_second, partner_time[k, v])
                total = np.maximum(k_start, other.start)
                near = np.flatnonzero(total <= min(upper, total.min()) + EPS)
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
