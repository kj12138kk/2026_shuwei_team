"""Problem 1: continuous region, complete optimal arc and ready-time tie break."""
from itertools import product

import numpy as np
from scipy.integrate import quad_vec
from scipy.spatial import ConvexHull

from geometry import (
    ANGULAR_SPEED, BUILDINGS, CAMERA, RADIUS, SPEED, STARTS, TARGETS,
    observable, pointing, station_constraints,
)

AREA_LABELS = ("body_forbidden", "elevation_loss", "occlusion_loss", "observable")


def _shadow_polygons(target):
    """Project each building from the target to the camera plane."""
    shadows = []
    for x0, x1, y0, y1, height in BUILDINGS:
        corners = np.array(list(product((x0, x1), (y0, y1))))
        scale = (target.height - CAMERA) / (target.height - height)
        vertices = np.vstack((corners, target.xy + scale * (corners - target.xy)))
        shadows.append(vertices[ConvexHull(vertices).vertices])
    return shadows


def _circle_line_angles(center, radius, normal, offset):
    size = np.linalg.norm(normal)
    ratio = (offset - normal @ center) / (radius * size)
    if abs(ratio) > 1. + 1e-12:
        return ()
    angle = np.arctan2(normal[1], normal[0])
    shift = np.arccos(np.clip(ratio, -1., 1.))
    return ((angle - shift) % (2. * np.pi), (angle + shift) % (2. * np.pi))


def _station(target, angles):
    return target.xy + target.min_range * np.stack((np.cos(angles), np.sin(angles)), axis=-1)


def _optimal_intervals(target, shadows):
    """Partition the minimum-range circle at every possible boundary hit."""
    center, radius = target.xy, target.min_range
    cuts = [0., 2. * np.pi]
    for j, normal in enumerate(np.eye(2)):
        offsets = [RADIUS, 100. - RADIUS]
        for rect in BUILDINGS:
            offsets.extend((rect[2 * j] - RADIUS, rect[2 * j + 1] + RADIUS))
        for offset in offsets:
            cuts.extend(_circle_line_angles(center, radius, normal, offset))
    for x0, x1, y0, y1, _ in BUILDINGS:
        for corner in map(np.asarray, product((x0, x1), (y0, y1))):
            normal = corner - center
            if np.linalg.norm(normal) > 1e-12:
                offset = (corner @ corner - center @ center + radius ** 2 - RADIUS ** 2) / 2.
                cuts.extend(_circle_line_angles(center, radius, normal, offset))
    for shadow in shadows:
        for a, b in zip(shadow, np.roll(shadow, -1, axis=0)):
            dx, dy = b - a
            normal = np.array([-dy, dx])
            cuts.extend(_circle_line_angles(center, radius, normal, normal @ a))

    cuts = np.sort(cuts)
    cuts = cuts[np.r_[True, np.diff(cuts) > 1e-10]]
    cuts[-1] = 2. * np.pi
    endpoints = observable(_station(target, cuts), target)
    visible = observable(_station(target, (cuts[:-1] + cuts[1:]) / 2.), target)
    intervals = []
    for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
        if visible[i]:
            if i > 0 and visible[i - 1] and endpoints[i]:
                intervals[-1] = (intervals[-1][0], b, intervals[-1][2], bool(endpoints[i + 1]))
            else:
                intervals.append((a, b, bool(endpoints[i]), bool(endpoints[i + 1])))
        elif endpoints[i] and (i == 0 or not visible[i - 1]):
            intervals.append((a, a, True, True))
    return intervals


def _union_length(intervals):
    """Length of the union after clipping to the vehicle-center field in y."""
    end, length = RADIUS, 0.
    for low, high in sorted(intervals):
        low, high = max(end, low), min(100. - RADIUS, high)
        if high > low:
            length += high - low
            end = high
    return length


def _continuous_areas(target, shadows):
    edges = []
    breaks = [target.xy[0] - target.min_range, target.xy[0] + target.min_range]
    for x0, x1, *_ in BUILDINGS:
        breaks.extend((x0 - RADIUS, x0, x1, x1 + RADIUS))
    for shadow in shadows:
        a = shadow
        b = np.roll(shadow, -1, axis=0)
        valid = abs(b[:, 0] - a[:, 0]) > 1e-12
        a, b = a[valid], b[valid]
        slope = (b[:, 1] - a[:, 1]) / (b[:, 0] - a[:, 0])
        edges.append((np.minimum(a[:, 0], b[:, 0]), np.maximum(a[:, 0], b[:, 0]),
                      slope, a[:, 1] - slope * a[:, 0]))
        breaks.extend(shadow[:, 0])
    breaks = sorted({x for x in breaks if RADIUS < x < 100. - RADIUS})
    tx, ty = target.xy
    minimum = target.min_range

    def slice_lengths(x):
        body = []
        for x0, x1, y0, y1, _ in BUILDINGS:
            dx = max(x0 - x, x - x1, 0.)
            if dx <= RADIUS:
                dy = np.sqrt(RADIUS ** 2 - dx ** 2)
                body.append((y0 - dy, y1 + dy))
        body_cut = _union_length(body)
        if abs(x - tx) < minimum:
            dy = np.sqrt(minimum ** 2 - (x - tx) ** 2)
            body.append((ty - dy, ty + dy))
        elevation_cut = _union_length(body)
        for lo, hi, slope, intercept in edges:
            hits = (lo <= x) & (x <= hi)
            ys = slope[hits] * x + intercept[hits]
            if len(ys) >= 2:
                body.append((ys.min(), ys.max()))
        total_cut = _union_length(body)
        return np.array([body_cut, elevation_cut - body_cut, total_cut - elevation_cut,
                         100. - 2. * RADIUS - total_cut])

    areas, estimate, info = quad_vec(slice_lengths, RADIUS, 100. - RADIUS, points=breaks,
                                     epsabs=1e-6, epsrel=1e-9, full_output=True)
    if not info.success:
        raise RuntimeError("Problem 1 area quadrature did not converge.")
    areas[0] += 100. ** 2 - (100. - 2. * RADIUS) ** 2
    return dict(zip(AREA_LABELS, areas.tolist())), float(estimate)


def classification_grid(target, step):
    """Four exclusive classes, in the order body, elevation, occlusion, visible."""
    n = round(100. / step)
    axis = (np.arange(n) + .5) * step
    classes = np.empty((n, n), dtype=np.uint8)
    for i in range(0, n, 96):
        x, y = np.meshgrid(axis, axis[i:i + 96])
        body, elevation, visible = station_constraints(np.stack((x, y), axis=-1), target)
        classes[i:i + len(x)] = np.where(body, np.where(elevation, np.where(visible, 3, 2), 1), 0)
    return classes


def _fastest_station(target, router):
    """Certify the unrestricted Euclidean winner and its feasible straight path."""
    minimum = target.min_range
    start_offsets = STARTS - target.xy
    radii = np.linalg.norm(start_offsets, axis=1)
    lower = abs(radii - minimum) / SPEED + 80. / ANGULAR_SPEED
    v = np.argmin(lower)
    if radii[v] / SPEED >= np.rad2deg(1.) / ANGULAR_SPEED:
        raise RuntimeError("Problem 1 monotonicity certificate needs a nearer initial vehicle.")

    # In the pan-limited sectors the angular slew slope dominates the drive-time
    # slope. In the level sector only the end points or the nearest direction can win.
    az0 = np.arctan2(start_offsets[v, 1], start_offsets[v, 0]) % (2. * np.pi)
    angles = np.array([np.deg2rad(100.), np.deg2rad(260.), az0])
    points = _station(target, angles)
    travel = np.linalg.norm(points - STARTS[v], axis=1) / SPEED
    pan, tilt, slew = pointing(points, target)
    i = np.argmin(travel + slew)
    ready = travel[i] + slew[i]
    if (not observable(points[i], target) or
            not np.isfinite(router.direct_matrix(STARTS[v:v + 1], points[i:i + 1])[0, 0]) or
            np.any(np.delete(lower, v) <= ready)):
        raise RuntimeError("The straight-path and competing-vehicle certificate failed.")
    return dict(selected_station_m=points[i].tolist(), selected_vehicle=f"V{v + 1}",
                station_polar_angle_deg=float(np.rad2deg(angles[i])),
                azimuth_deg=float(pan[i]), elevation_deg=float(tilt[i]),
                drive_distance_m=float(travel[i] * SPEED), travel_time_s=float(travel[i]),
                slew_time_s=float(slew[i]), ready_time_s=float(ready),
                other_vehicles_ready_lower_bound_s={f"V{j + 1}": float(t) for j, t in enumerate(lower) if j != v},
                path_xy_m=[STARTS[v].tolist(), points[i].tolist()])


def solve_t1(router):
    target = TARGETS[0]
    shadows = _shadow_polygons(target)
    intervals = _optimal_intervals(target, shadows)
    areas, error = _continuous_areas(target, shadows)
    grids = {}
    for step in (.125, .10, .05):
        grid = classification_grid(target, step)
        grid_areas = np.bincount(grid.ravel(), minlength=4) * step ** 2
        grids[str(step)] = dict(zip(AREA_LABELS, grid_areas.tolist()))
    widths = BUILDINGS[:, 1] - BUILDINGS[:, 0]
    heights = BUILDINGS[:, 3] - BUILDINGS[:, 2]
    body_allowed = (100. - 2. * RADIUS) ** 2 - np.sum(
        widths * heights + 2. * RADIUS * (widths + heights) + np.pi * RADIUS ** 2)
    assert abs(100. ** 2 - areas["body_forbidden"] - body_allowed) < 1e-4
    assert abs(sum(areas.values()) - 100. ** 2) < 1e-6
    length = sum(b - a for a, b, _, _ in intervals)
    return dict(target_xy_m=target.xy.tolist(), range_m=float(target.min_range),
                quality_per_m=float(1. / target.min_range), continuous_areas_m2=areas,
                body_allowed_closed_form_m2=float(body_allowed), quadrature_error_estimate_m2=error,
                observable_area_by_grid_m2={step: values["observable"] for step, values in grids.items()},
                grid_areas_m2=grids,
                observable_from_starts=[f"V{i + 1}" for i, seen in enumerate(observable(STARTS, target)) if seen],
                optimal_arc_intervals_deg=[
                    dict(start_deg=float(np.rad2deg(a)), end_deg=float(np.rad2deg(b)),
                         start_included=inc_a, end_included=inc_b) for a, b, inc_a, inc_b in intervals],
                optimal_arc_total_deg=float(np.rad2deg(length)),
                optimal_arc_length_m=float(target.min_range * length),
                **_fastest_station(target, router))
