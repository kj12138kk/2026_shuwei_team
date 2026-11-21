"""Problem 2(1): the minimum-DRMS family and its earliest synchronized fix."""
from itertools import permutations

import numpy as np

from errors import validate_intersection
from geometry import (
    ANGULAR_SPEED, EPS, SPEED, STARTS, TARGETS, observable, pair_geometry,
    path_length, pointing,
)
from problem1 import _optimal_intervals, _shadow_polygons, _station


TRAVEL_LIMIT_S = 30.
SCAN_STEP_DEG = .05


def _pair_arc_closures(target):
    station_arcs = _optimal_intervals(target, _shadow_polygons(target))
    pair_arcs = []
    for a, b, _, _ in station_arcs:
        for c, d, _, _ in station_arcs:
            for turn in (0., 2. * np.pi):
                low = max(a, c - np.pi / 2. + turn)
                high = min(b, d - np.pi / 2. + turn)
                if low <= high:
                    pair_arcs.append((low, high))
    return pair_arcs


def _euclidean_ready_breaks(origin, target):
    """Candidate minima of straight-line travel plus pointing on the circle."""
    vector = origin - target.xy
    azimuth = np.arctan2(vector[1], vector[0])
    # Pan dominance switches at 100 and 260 degrees. At 0 it changes slope.
    angles = [*np.deg2rad([0., 100., 260.]), azimuth, azimuth + np.pi]
    radius, distance = target.min_range, np.linalg.norm(vector)
    if distance > EPS:
        # In the pan-dominated pieces, equate the magnitude of the
        # Euclidean travel-time derivative to the pan-time slope (radians).
        k = (SPEED * np.rad2deg(1.) / ANGULAR_SPEED) ** 2
        product = radius * distance
        coefficients = [product ** 2, -2. * k * product,
                        k * (radius ** 2 + distance ** 2) - product ** 2]
        for root in np.roots(coefficients):
            if abs(root.imag) <= EPS and abs(root.real) <= 1. + EPS:
                shift = np.arccos(np.clip(root.real, -1., 1.))
                angles.extend((azimuth - shift, azimuth + shift))
    return np.asarray(angles) % (2. * np.pi)


def _euclidean_ready_minimum(origin, target, critical, offset, low, high):
    angles = (critical - offset) % (2. * np.pi)
    sample = np.r_[low, high, angles[(low <= angles) & (angles <= high)]]
    points = _station(target, sample + offset)
    travel = np.linalg.norm(points - origin, axis=1) / SPEED
    return float(np.min(travel + pointing(points, target)[2])), sample


def _describe(router, target, vehicles, stations, polar_angles):
    paths = [router.path(STARTS[v], station) for v, station in zip(vehicles, stations)]
    travel = np.asarray([path_length(path) / SPEED for path in paths])
    pan, elevation, slew = pointing(stations, target)
    ready = travel + slew
    if np.any(travel > TRAVEL_LIMIT_S + EPS) or not np.all(observable(stations, target)):
        raise RuntimeError("An accuracy-optimal T2 station pair is not feasible.")
    return dict(
        vehicles=[f"V{vehicle + 1}" for vehicle in vehicles],
        stations_m=stations.tolist(),
        station_polar_angles_deg=(np.rad2deg(polar_angles) % 360.).tolist(),
        travel_times_s=travel.tolist(),
        path_xy_m=[path.tolist() for path in paths],
        azimuth_deg=pan.tolist(),
        elevation_deg=elevation.tolist(),
        pointing_times_s=slew.tolist(),
        ready_times_s=ready.tolist(),
        synchronized_ready_s=float(ready.max()),
        **pair_geometry(*stations, target),
    )


def solve_t2_accuracy(router):
    target = TARGETS[1]
    assignments = tuple(permutations(range(len(STARTS)), 2))
    lower_bounds = {pair: np.inf for pair in assignments}
    arc_closures = _pair_arc_closures(target)
    critical = [_euclidean_ready_breaks(origin, target) for origin in STARTS]
    candidates = []
    for low, high in arc_closures:
        arc_minima = np.empty((2, len(STARTS)))
        candidates.extend(np.arange(low, high, np.deg2rad(SCAN_STEP_DEG)))
        for station, offset in enumerate((0., np.pi / 2.)):
            for vehicle, origin in enumerate(STARTS):
                arc_minima[station, vehicle], sample = _euclidean_ready_minimum(
                    origin, target, critical[vehicle], offset, low, high)
                candidates.extend(sample)
        for i, j in assignments:
            relaxed = float(max(arc_minima[0, i], arc_minima[1, j]))
            lower_bounds[i, j] = min(lower_bounds[i, j], relaxed)

    phi = np.unique(candidates)
    first = _station(target, phi)
    second = _station(target, phi + np.pi / 2.)
    allowed = (observable(first, target) & observable(second, target)
               & router.free_points(first) & router.free_points(second))
    phi, first, second = phi[allowed], first[allowed], second[allowed]
    if not len(phi):
        raise RuntimeError("The T2 minimum-range circle has no feasible perpendicular stations.")
    first_travel = router.distances(STARTS, first) / SPEED
    second_travel = router.distances(STARTS, second) / SPEED
    first_slew = pointing(first, target)[2]
    second_slew = pointing(second, target)[2]
    options = []
    for i, j in assignments:
        feasible = (first_travel[i] <= TRAVEL_LIMIT_S + EPS) & (second_travel[j] <= TRAVEL_LIMIT_S + EPS)
        objective = np.maximum(first_travel[i] + first_slew, second_travel[j] + second_slew)
        objective = np.where(feasible, objective, np.inf)
        n = int(np.argmin(objective))
        if np.isfinite(objective[n]):
            options.append((float(objective[n]), i, j, n))
    if not options:
        raise RuntimeError("No accuracy-optimal T2 pair satisfies the 30 s travel limit.")
    _, i, j, best = min(options)
    stations = np.array([first[best], second[best]])
    selected = _describe(router, target, (i, j), stations, phi[best] + np.array([0., np.pi / 2.]))

    proof_angles = np.deg2rad([180., 90.])
    certificate = _describe(router, target, (0, 2), _station(target, proof_angles), proof_angles)
    geometry_bound = float(np.deg2rad(1.) * np.sqrt(2.) * target.min_range)
    if not all(abs(pair["linear_rmse_m"] - geometry_bound) <= EPS for pair in (selected, certificate)):
        raise RuntimeError("The T2 accuracy lower bound was not attained.")

    # The Euclidean relaxation is valid for actual disk paths as well as the
    # polygonal visibility graph. A feasible upper bound matching it proves
    # continuous optimality of the ready-time tiebreak.
    ready_bound = min(lower_bounds.values())
    gap = selected["synchronized_ready_s"] - ready_bound
    if gap < -EPS:
        raise RuntimeError("The T2 readiness lower bound exceeds the feasible solution.")
    return dict(
        target_xy_m=target.xy.tolist(),
        minimum_range_m=float(target.min_range),
        minimum_drms_m=geometry_bound,
        travel_limit_s=TRAVEL_LIMIT_S,
        first_angle_interval_closures_deg=np.rad2deg(arc_closures).tolist(),
        angular_scan_step_deg=SCAN_STEP_DEG,
        feasible_angle_samples=len(phi),
        ready_time_lower_bound_s=ready_bound,
        certified_ready_gap_s=max(0., gap),
        continuous_ready_optimal=gap <= EPS,
        assignment_ready_lower_bounds_s={
            f"V{a + 1}+V{b + 1}": bound for (a, b), bound in lower_bounds.items()
        },
        west_north_certificate=certificate,
        ready_time_improvement_s=certificate["synchronized_ready_s"] - selected["synchronized_ready_s"],
        **selected,
        **validate_intersection(*stations, target),
    )
