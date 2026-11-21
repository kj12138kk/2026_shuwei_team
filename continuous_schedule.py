"""Feasible joint refinement and continuous bounds for the nominal three alerts."""
from functools import lru_cache
from heapq import heappop, heappush
from itertools import count, product

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from geometry import ANGULAR_SPEED, BUILDINGS, CAMERA, EPS, OBSERVE, RADIUS, SPEED, STARTS, path_length, pointing
from problem1 import _shadow_polygons
from scheduler import MIN_ANGLE, evaluate_fixed


def _shadows(target):
    edges = []
    for polygon in _shadow_polygons(target):
        edge = np.roll(polygon, -1, axis=0) - polygon
        normal = np.column_stack((edge[:, 1], -edge[:, 0]))
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        edges.append((normal, np.sum(normal * polygon, axis=1)))
    return edges


def refine_joint(router, targets, plan):
    """Vary the two stations on the critical reuse chain and their partners together."""
    if not plan:
        return plan, []
    layout = [(m.target, tuple(leg.vehicle for leg in m.legs)) for m in plan]
    previous = {leg.vehicle: i for i, m in enumerate(plan[:-1]) for leg in m.legs}
    reused = [leg for leg in plan[-1].legs if leg.vehicle in previous]
    if not reused:
        return plan, []
    critical = max(reused, key=lambda leg: leg.ready).vehicle
    j, k = previous[critical], len(plan) - 1
    selected = (2 * j, 2 * j + 1, 2 * k + layout[k][1].index(critical))
    best = plan
    history = []
    shadows = {name: _shadows(targets[name]) for name in (layout[j][0], layout[k][0])}

    def rank(candidate):
        times = [m.start for m in candidate]
        return max(times), sum(times)

    for stations in (selected, (2 * j, 2 * j + 1, 2 * k, 2 * k + 1)):
        # Preserve the original per-vehicle precedence, even if independent alerts swap in time.
        by_name = {m.target: m for m in best}
        base = np.array([leg.path[-1] for name, _ in layout for leg in by_name[name].legs])
        slots = list(stations)
        initial = base[slots].ravel()
        upper = best[-1].start
        lower_coords = np.maximum(RADIUS, initial - 1.5)
        upper_coords = np.minimum(100. - RADIUS, initial + 1.5)

        def coordinates(values):
            points = base.copy()
            points[slots] = values[:-1].reshape(-1, 2)
            return points.reshape(-1, 2, 2)

        def constraints(values):
            groups = coordinates(values)
            bounds = []
            for m in (j, k):
                name, _ = layout[m]
                p = groups[m]
                target = targets[name]
                d = target.xy - p
                r = np.linalg.norm(d, axis=1)
                bounds.extend(r - target.min_range)
                bounds.append(np.linalg.norm(p[0] - p[1]) - 2. * RADIUS)
                sine = abs(d[0, 0] * d[1, 1] - d[0, 1] * d[1, 0]) / max(EPS, r.prod())
                bounds.append(sine - np.sin(np.deg2rad(MIN_ANGLE)))
                for normal, intercept in shadows[name]:
                    # Strict LOS exclusion, with a small numerical interior tolerance.
                    bounds.extend(np.max(p @ normal.T - intercept, axis=1) - 2e-7)
                for normal, intercept, _, _ in router.halfspaces:
                    bounds.extend(np.max(intercept - p @ normal.T, axis=1))
            positions, release = STARTS.copy(), np.zeros(len(STARTS))
            for (name, vehicles), p in zip(layout, groups):
                ready = []
                slew = pointing(p, targets[name])[2]
                for v, station, g in zip(vehicles, p, slew):
                    try:
                        distance = path_length(router.path(positions[v], station))
                    except ValueError:
                        # Keep the trial objective finite inside an excluded obstacle.
                        distance = np.linalg.norm(positions[v] - station)
                    ready.append(release[v] + distance / SPEED + g)
                start = max(ready)
                bounds.append(values[-1] - start)
                for v, station, g in zip(vehicles, p, slew):
                    positions[v] = station
                    release[v] = start + OBSERVE + g
            return np.asarray(bounds)

        # The epigraph variable avoids differentiating the top-level maximum.
        perturb = .04 * np.resize(np.array([1., -1., -1., 1.]), len(initial))
        accepted = 0
        for seed in (initial, initial + perturb, initial - perturb):
            solved = minimize(lambda z: z[-1], np.r_[np.clip(seed, lower_coords, upper_coords), upper],
                              bounds=[*zip(lower_coords, upper_coords), (0., upper + 2.)],
                              constraints={"type": "ineq", "fun": constraints}, method="SLSQP",
                              options={"maxiter": 100, "ftol": 1e-10})
            if not np.all(np.isfinite(solved.x)):
                continue
            for fraction in (1., .99, .9, .5):
                trial = np.r_[initial + fraction * (solved.x[:-1] - initial), solved.x[-1]]
                candidate_layout = [(name, vehicles, p) for (name, vehicles), p in zip(layout, coordinates(trial))]
                candidate = evaluate_fixed(router, targets, candidate_layout, STARTS, (0.,) * len(STARTS))
                if candidate is not None and rank(candidate) < rank(best):
                    best = candidate
                    accepted += 1
                    break
        history.append(dict(vehicles=[critical + 1], chain=[layout[j][0], layout[k][0]], coordinates=len(initial),
                            starts_tried=3, improvements=accepted, before_s=upper, after_s=best[-1].start))
    return best, history


def _bearing_spread(centers, half, target):
    delta = target.xy - centers
    distance = np.linalg.norm(delta, axis=-1)
    radius = np.sqrt(2.) * half
    spread = np.arcsin(np.minimum(1., radius / np.maximum(distance, EPS)))
    return np.arctan2(delta[..., 1], delta[..., 0]), np.where(distance <= radius, np.pi, spread)


def _cell_slew(centers, half, target, shadows):
    """Optimistic pointing time; reject a cell only if it is wholly infeasible."""
    delta = target.xy - centers
    max_range = np.linalg.norm(np.abs(delta) + half, axis=1)
    x_max = delta[:, 0] + half
    y_abs = np.abs(delta[:, 1])
    y = np.where(x_max >= 0., np.maximum(y_abs - half, 0.), y_abs + half)
    pan_lower = np.rad2deg(np.arctan2(y, x_max))
    tilt_lower = np.rad2deg(np.arctan2(target.height - CAMERA, max_range))
    allowed = max_range >= target.min_range - EPS
    for x0, x1, y0, y1, _ in BUILDINGS:
        offset = np.maximum(np.maximum(np.array([x0, y0]) - (centers - half),
                                        centers + half - np.array([x1, y1])), 0.)
        allowed &= np.linalg.norm(offset, axis=1) >= RADIUS - EPS
    for normal, intercept in shadows:
        outside = centers @ normal.T - intercept + half * np.sum(np.abs(normal), axis=1)
        allowed &= np.any(outside >= -EPS, axis=1)
    return np.where(allowed, np.maximum(pan_lower, tilt_lower) / ANGULAR_SPEED, np.inf)


def _partition(step):
    edges = np.linspace(RADIUS, 100. - RADIUS, int(np.ceil((100. - 2. * RADIUS) / step)) + 1)
    axis = (edges[1:] + edges[:-1]) / 2.
    return np.column_stack([a.ravel() for a in np.meshgrid(axis, axis)]), float((edges[1] - edges[0]) / 2.)


def _start_drive(centers, half):
    return np.linalg.norm(np.maximum(np.abs(STARTS[:, None] - centers) - half, 0.), axis=-1) / SPEED


def adaptive_chain_bound(targets, upper, legacy, *, relative_gap=.005, node_budget=60_000):
    """Refine only the two-station cells still capable of beating a feasible plan."""
    # The first and second mate are independently available at t=0, and the
    # third mission may ignore all other resource use. Both relaxations are safe.
    shadows = {name: _shadows(target) for name, target in targets.items()}
    mates_xy, mate_half = _partition(.5)
    drive = _start_drive(mates_xy, mate_half)
    mate_data, independent = {}, {}
    # Refine the quickest mate boxes separately. The first omitted time from
    # each partition remains a valid bound even when its geometry is unknown.
    mate_parts = 8
    sub_half = mate_half / mate_parts
    axis = -mate_half + (np.arange(mate_parts) + .5) * (2. * sub_half)
    offsets = np.column_stack([a.ravel() for a in np.meshgrid(axis, axis)])
    for name, target in targets.items():
        g = _cell_slew(mates_xy, mate_half, target, shadows[name])
        ready = drive + g
        independent[name] = float(np.partition(ready.min(axis=1), 1)[1])
        for v in range(len(STARTS)):
            mate = np.delete(ready, v, axis=0).min(axis=0)
            order = np.argsort(mate, kind="stable")
            roots = order[:32]
            escape = float(mate[order[32]]) if len(order) > 32 else np.inf
            xy = (mates_xy[roots, None] + offsets).reshape(-1, 2)
            g_sub = _cell_slew(xy, sub_half, target, shadows[name])
            t_sub = np.delete(_start_drive(xy, sub_half), v, axis=0).min(axis=0) + g_sub
            t_sub = np.maximum(t_sub, np.repeat(mate[roots], len(offsets)))
            order = np.argsort(t_sub, kind="stable")[:129]
            if len(order) > 128:
                escape = min(escape, float(t_sub[order[128]]))
            leading = order[:128]
            angles, spread = _bearing_spread(xy[leading], sub_half, target)
            mate_data[name, v] = (xy[leading], angles, spread, t_sub[leading], escape)

    def mate_ready(name, v, xy, half):
        p, angles, spread, ready, escape = mate_data[name, v]
        angle, error = _bearing_spread(xy, half, targets[name])
        separation = np.linalg.norm(np.abs(p - xy) + half + sub_half, axis=1)
        diff = np.abs((angle - angles + np.pi / 2.) % np.pi - np.pi / 2.)
        maybe = ((separation >= 2. * RADIUS - EPS) &
                 (diff + spread + error >= np.deg2rad(MIN_ANGLE) - EPS))
        return float(min(escape, ready[maybe].min(initial=np.inf)))

    @lru_cache(maxsize=65_536)
    def cell(name, v, coord):
        x, y, half = coord
        xy = np.array([x, y])
        g = float(_cell_slew(xy[None], half, targets[name], shadows[name])[0])
        if not np.isfinite(g):
            return np.inf, np.inf, np.inf
        t = np.linalg.norm(np.maximum(np.abs(STARTS[v] - xy) - half, 0.)) / SPEED + g
        return g, float(t), mate_ready(name, v, xy, half)

    def chain_bound(chain, p, q, parent):
        j, k, v, third = chain
        g0, own, mate0 = cell(j, v, p)
        g1, _, mate1 = cell(k, v, q)
        gap = np.maximum(np.abs(np.subtract(p[:2], q[:2])) - p[2] - q[2], 0.)
        chained = max(own, mate0) + OBSERVE + g0 + np.linalg.norm(gap) / SPEED + g1
        return max(parent, independent[third], mate1, float(chained) - EPS)

    centers, half = _partition(1.)
    slews = {name: _cell_slew(centers, half, target, shadows[name]) for name, target in targets.items()}
    drive = _start_drive(centers, half)
    ready = {name: drive + g for name, g in slews.items()}
    counter = count()
    heap = []
    chains = []
    for link, vehicles in legacy["by_target_pair_vehicle_s"].items():
        j, k = link.split("->")
        third, = (name for name in targets if name not in (j, k))
        if independent[third] >= upper:
            continue
        for v, old_bound in enumerate(vehicles):
            if old_bound >= upper:
                continue
            chain = (j, k, v, third)
            chains.append(f"V{v + 1}:{link}")
            g_min = slews[k].min()
            partner = np.delete(ready[j], v, axis=0).min()
            prefix = np.maximum(ready[j][v], partner) + OBSERVE + slews[j]
            first = np.flatnonzero(prefix + g_min < upper + EPS)
            second = np.flatnonzero(np.isfinite(slews[k]))
            if not len(first) or not len(second):
                continue
            tree = cKDTree(centers[second])
            radius = SPEED * (upper - prefix[first] - g_min + EPS) + 2. * np.sqrt(2.) * half
            for i, candidates in zip(first, tree.query_ball_point(centers[first], radius)):
                p = (*centers[i], half)
                for h in second[candidates]:
                    q = (*centers[h], half)
                    lower = chain_bound(chain, p, q, old_bound)
                    if lower < upper:
                        heappush(heap, (lower, next(counter), chain, p, q))

    root_pairs, expanded, pruned = len(heap), 0, 0
    goal = upper / (1. + relative_gap)
    min_half = 1e-6
    while heap and heap[0][0] < goal and expanded < node_budget:
        lower, order, chain, p, q = heappop(heap)
        first = p[2] >= q[2]
        x, y, size = p if first else q
        if size <= min_half:
            heappush(heap, (lower, order, chain, p, q))
            break
        expanded += 1
        step = size / 2.
        for dx, dy in product((-step, step), repeat=2):
            child = (x + dx, y + dy, step)
            a, b = (child, q) if first else (p, child)
            bound = chain_bound(chain, a, b, lower)
            if bound < upper:
                heappush(heap, (bound, next(counter), chain, a, b))
            else:
                pruned += 1

    lower = heap[0][0] if heap else upper
    return dict(lower_bound_s=float(lower), upper_bound_s=float(upper), absolute_gap_s=float(upper - lower),
                relative_gap=float((upper - lower) / lower), target_relative_gap=relative_gap,
                termination=("closed" if not heap else "gap_target" if lower >= goal else
                             "node_budget" if expanded >= node_budget else "cell_size_limit"),
                active_reuse_chains=chains, root_pairs=root_pairs, expanded_pairs=expanded,
                pruned_children=pruned, remaining_pairs=len(heap))
