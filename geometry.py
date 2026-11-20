"""Field geometry and polygonal shortest paths. Angles are in degrees."""
from dataclasses import dataclass

import numpy as np

CENTER = np.array([50.0, 50.0])
STARTS = np.array([[15., 14.], [88., 10.], [14., 86.], [88., 68.], [50., 93.]])
# Columns are xmin, xmax, ymin, ymax, height.
BUILDINGS = np.array([
    [22, 34, 16, 26, 20], [63, 79, 59, 71, 28],
    [10, 24, 65, 76, 12], [72, 86, 22, 34, 24], [41, 51, 38, 46, 32],
], dtype=float)
RADIUS, SPEED, CAMERA, ANGULAR_SPEED, OBSERVE = 1.5, 3., 2., 30., 5.
EPS = 1e-9


@dataclass(frozen=True)
class Target:
    name: str
    theta: float
    distance: float
    height: float

    @property
    def xy(self):
        angle = np.deg2rad(self.theta)
        return CENTER + self.distance * np.array([np.cos(angle), np.sin(angle)])

    @property
    def min_range(self):
        return (self.height - CAMERA) / np.tan(np.deg2rad(80.))


TARGETS = (Target("T1", 30, 40, 45), Target("T2", 150, 45, 35), Target("T3", 285, 38, 55))


def wrap(angle):
    return (angle + 180.) % 360. - 180.


def pointing(points, target):
    delta = target.xy - np.asarray(points)
    distance = np.linalg.norm(delta, axis=-1)
    pan = wrap(np.rad2deg(np.arctan2(delta[..., 1], delta[..., 0])))
    tilt = np.rad2deg(np.arctan2(target.height - CAMERA, distance))
    slew = np.maximum(np.abs(pan), np.abs(tilt)) / ANGULAR_SPEED
    return pan, tilt, slew


def _box_interval(start, end, low, high, cap=1.):
    """Clip a segment against a closed axis-aligned box."""
    delta = end - start
    entry = np.zeros(delta.shape[:-1])
    exit_ = np.broadcast_to(np.asarray(cap, dtype=float), entry.shape).copy()
    for j in range(delta.shape[-1]):
        static = np.abs(delta[..., j]) <= EPS
        divisor = np.where(static, 1., delta[..., j])
        a = (low[j] - start[..., j]) / divisor
        b = (high[j] - start[..., j]) / divisor
        outside = (start[..., j] < low[j] - EPS) | (start[..., j] > high[j] + EPS)
        entry = np.maximum(entry, np.where(static, np.where(outside, np.inf, -np.inf), np.minimum(a, b)))
        exit_ = np.minimum(exit_, np.where(static, np.inf, np.maximum(a, b)))
    return entry <= exit_ + EPS


def observable(points, target):
    points = np.asarray(points, dtype=float)
    good = np.all((points >= RADIUS - EPS) & (points <= 100. - RADIUS + EPS), axis=-1)
    _, tilt, _ = pointing(points, target)
    good &= (tilt >= -10. - EPS) & (tilt <= 80. + EPS)
    for x0, x1, y0, y1, height in BUILDINGS:
        low, high = np.array([x0, y0]), np.array([x1, y1])
        offset = np.maximum(np.maximum(low - points, points - high), 0.)
        good &= np.sum(offset ** 2, axis=-1) >= (RADIUS - EPS) ** 2
        cap = min(1., (height - CAMERA) / (target.height - CAMERA))
        good &= ~_box_interval(points, np.broadcast_to(target.xy, points.shape), low, high, cap)
    return good


def pair_geometry(p, q, target):
    u, v = target.xy - p, target.xy - q
    r1, r2 = np.linalg.norm(u), np.linalg.norm(v)
    cross = u[0] * v[1] - u[1] * v[0]
    gamma = np.rad2deg(np.arctan2(abs(cross), u @ v))
    baseline = np.linalg.norm(p - q)
    rmse = np.deg2rad(1.) * np.hypot(r1, r2) / abs(np.sin(np.deg2rad(gamma))) if abs(cross) > EPS else np.inf
    return dict(baseline_m=float(baseline), angle_deg=float(gamma), ranges_m=[float(r1), float(r2)],
                linear_rmse_m=float(rmse))


def _polygon(x0, x1, y0, y1, arc_parts):
    """Circumscribed tangent polygon of a radius-1.5 rounded rectangle."""
    corners = [(x1, y1), (x0, y1), (x0, y0), (x1, y0)]
    step = np.pi / (2 * arc_parts)
    theta = (np.arange(arc_parts) + 0.5) * step
    radius = RADIUS / np.cos(step / 2.)
    vertices = []
    for j, corner in enumerate(corners):
        angle = theta + j * np.pi / 2.
        vertices.extend(np.array(corner) + radius * np.column_stack((np.cos(angle), np.sin(angle))))
    return np.asarray(vertices)


class Router:
    """Visibility graph through all obstacle vertices, with open interiors excluded."""

    def __init__(self, arc_parts=4):
        polygons = [_polygon(*box[:4], arc_parts) for box in BUILDINGS]
        self.vertices = np.concatenate(polygons)
        self.halfspaces = []
        for polygon, box in zip(polygons, BUILDINGS):
            edge = np.roll(polygon, -1, axis=0) - polygon
            inward = np.column_stack((-edge[:, 1], edge[:, 0]))
            inward /= np.linalg.norm(inward, axis=1)[:, None]
            intercept = np.sum(inward * polygon, axis=1)
            self.halfspaces.append((inward, intercept, np.array([box[0], box[2]]) - RADIUS,
                                    np.array([box[1], box[3]]) + RADIUS))
        distance = self.direct_matrix(self.vertices, self.vertices)
        self.next_node = np.broadcast_to(np.arange(len(distance)), distance.shape).copy()
        for k in range(len(distance)):
            via = distance[:, [k]] + distance[[k], :]
            better = via < distance - EPS
            self.next_node = np.where(better, self.next_node[:, [k]], self.next_node)
            distance = np.minimum(distance, via)
        self.graph_distance = distance

    def free_points(self, points):
        points = np.asarray(points)
        result = np.ones(len(points), dtype=bool)
        for normals, intercept, _, _ in self.halfspaces:
            result &= ~np.all(points @ normals.T > intercept + EPS, axis=1)
        return result

    def direct_matrix(self, starts, ends):
        starts, ends = np.asarray(starts), np.asarray(ends)
        distance = np.linalg.norm(starts[:, None] - ends[None, :], axis=-1)
        for normals, intercept, low, high in self.halfspaces:
            start_in = starts @ normals.T - intercept - EPS
            end_in = ends @ normals.T - intercept - EPS
            overlap = np.all(np.maximum(starts[:, None], ends[None, :]) >= low, axis=-1)
            overlap &= np.all(np.minimum(starts[:, None], ends[None, :]) <= high, axis=-1)
            a, b = np.nonzero(overlap & np.isfinite(distance))
            if not len(a):
                continue
            p, q = start_in[a], end_in[b]
            delta = q - p
            static = np.abs(delta) <= EPS
            roots = -p / np.where(static, 1., delta)
            enter = np.max(np.where(delta > EPS, roots, -np.inf), axis=1)
            leave = np.min(np.where(delta < -EPS, roots, np.inf), axis=1)
            cross = np.minimum(1., leave) > np.maximum(0., enter) + EPS
            cross &= ~np.any(static & (p <= 0.), axis=1)
            distance[a[cross], b[cross]] = np.inf
        return distance

    def distances(self, starts, ends):
        direct = self.direct_matrix(starts, ends)
        if np.all(np.isfinite(direct)):
            return direct
        ingress = self.direct_matrix(starts, self.vertices)
        egress = self.direct_matrix(self.vertices, ends)
        best_to_corner = np.min(ingress[:, :, None] + self.graph_distance[None], axis=1)
        for i in range(len(self.vertices)):
            direct = np.minimum(direct, best_to_corner[:, [i]] + egress[[i], :])
        return direct

    def path(self, start, end):
        start, end = np.asarray(start), np.asarray(end)
        if np.isfinite(self.direct_matrix(start[None], end[None])[0, 0]):
            return np.asarray([start, end])
        ingress = self.direct_matrix(start[None], self.vertices)[0]
        egress = self.direct_matrix(self.vertices, end[None])[:, 0]
        distances = ingress[:, None] + self.graph_distance + egress[None, :]
        a, b = np.unravel_index(np.argmin(distances), distances.shape)
        if not np.isfinite(distances[a, b]):
            raise ValueError("No collision-free path.")
        route = [start, self.vertices[a]]
        while a != b:
            a = self.next_node[a, b]
            route.append(self.vertices[a])
        return np.asarray([*route, end])


def path_length(path):
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def split_path(path, distance):
    path = np.asarray(path)
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    accumulated = np.r_[0., lengths.cumsum()]
    if distance >= accumulated[-1] - EPS:
        return path, path[-1:]
    i = np.searchsorted(accumulated, max(distance, 0.), side="right") - 1
    point = path[i] + (path[i + 1] - path[i]) * ((distance - accumulated[i]) / lengths[i])
    return np.vstack((path[:i + 1], point)), np.vstack((point, path[i + 1:]))
