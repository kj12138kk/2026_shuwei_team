"""Orthographic view of the observation field and routes to the Q1 station."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon, Rectangle
import numpy as np

from geometry import BUILDINGS, CAMERA, RADIUS, STARTS, TARGETS, Router, observable, path_length


INK = "#27323A"
GREY = "#687280"
GRID = "#DCE3E8"
TARGET_COLORS = ("#D55E00", "#0072B2", "#008464")
ROUTE_COLORS = ("#0072B2", "#D55E00", "#008464", "#7D4492", "#8C6700")
LABEL_BOX = {"boxstyle": "square,pad=0.08", "facecolor": "white", "edgecolor": "none", "alpha": 0.86}

# East, north and height are drawn at the same scale in an orthographic view.
AZIMUTH, ELEVATION = np.deg2rad([-35, 36])
VIEW = np.array([
    [-np.sin(AZIMUTH), np.cos(AZIMUTH), 0],
    [-np.cos(AZIMUTH) * np.sin(ELEVATION), -np.sin(AZIMUTH) * np.sin(ELEVATION),
     np.cos(ELEVATION)],
    [np.cos(AZIMUTH) * np.cos(ELEVATION), np.sin(AZIMUTH) * np.cos(ELEVATION),
     np.sin(ELEVATION)],
])


def _screen(points):
    return np.asarray(points) @ VIEW[:2].T


def _faces():
    """The east, south and roof faces visible from the selected viewpoint."""
    faces = []
    for x0, x1, y0, y1, height in BUILDINGS:
        faces.extend([
            ([(x1, y0, 0), (x1, y1, 0), (x1, y1, height), (x1, y0, height)], "#AAB5BE"),
            ([(x0, y0, 0), (x1, y0, 0), (x1, y0, height), (x0, y0, height)], "#C5CDD4"),
            ([(x0, y0, height), (x1, y0, height), (x1, y1, height), (x0, y1, height)], "#E1E7EB"),
        ])
    return [(np.array(vertices) @ VIEW.T, color) for vertices, color in faces]


def _hidden_interval(start, delta, face):
    """Find the parameter interval where a segment passes behind a projected face."""
    corner = face[0]
    edge_x, edge_y = face[[1, 3]] - corner
    basis = np.column_stack((edge_x[:2], edge_y[:2]))
    face_at_start = np.linalg.solve(basis, start[:2] - corner[:2])
    face_change = np.linalg.solve(basis, delta[:2])
    depth_edges = np.array([edge_x[2], edge_y[2]])
    behind = corner[2] + face_at_start @ depth_edges - start[2]
    behind_change = face_change @ depth_edges - delta[2]
    values = np.r_[face_at_start, 1 - face_at_start, behind]
    slopes = np.r_[face_change, -face_change, behind_change]

    lower, upper = 0.0, 1.0
    for value, slope in zip(values, slopes):
        if slope > 1e-10:
            lower = max(lower, -value / slope)
        elif slope < -1e-10:
            upper = min(upper, -value / slope)
        elif value < -1e-10:
            return None
    return (lower, upper) if lower < upper else None


def _visible_spans(start, stop, faces):
    start, stop = np.array([start, stop]) @ VIEW.T
    delta = stop - start
    hidden = sorted(
        interval for face, _ in faces
        if (interval := _hidden_interval(start, delta, face)) is not None
    )
    previous = 0.0
    for lower, upper in hidden:
        if lower > previous:
            yield (start + previous * delta)[:2], (start + lower * delta)[:2]
        previous = max(previous, upper)
    if previous < 1.0:
        yield (start + previous * delta)[:2], stop[:2]


def _base(ax):
    floor = _screen([(0, 0, 0), (100, 0, 0), (100, 100, 0), (0, 100, 0)])
    ax.add_patch(Polygon(floor, closed=True, fc="#F3F6F8", ec=GREY, lw=0.7, zorder=0))
    for step in range(20, 100, 20):
        ax.plot(*_screen([(step, 0, 0), (step, 100, 0)]).T, color=GRID, lw=0.55, zorder=1)
        ax.plot(*_screen([(0, step, 0), (100, step, 0)]).T, color=GRID, lw=0.55, zorder=1)

    ax.plot(*_screen([(0, 0, 0), (0, 0, 60)]).T, color=GREY, lw=0.75)
    for value in range(0, 101, 20):
        east = _screen((value, 0, 0))
        ax.plot([east[0], east[0] - 1.4], [east[1], east[1] - 1.6], color=GREY, lw=0.65)
        ax.annotate(str(value), east, xytext=(-2, -5), textcoords="offset points",
                    ha="center", va="top", fontsize=8, color=INK)
        if value:
            north = _screen((100, value, 0))
            ax.plot([north[0], north[0] + 1.6], [north[1], north[1] - 0.5], color=GREY, lw=0.65)
            ax.annotate(str(value), north, xytext=(4, -1), textcoords="offset points",
                        ha="left", va="center", fontsize=8, color=INK)
    for value in range(20, 61, 20):
        _, height = _screen((0, 0, value))
        ax.plot([0, -1.6], [height, height], color=GREY, lw=0.65)
        ax.annotate(str(value), (0, height), xytext=(-4, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=8, color=INK)
    ax.annotate("East (m)", _screen((49, 0, 0)), xytext=(-8, -28),
                textcoords="offset points", ha="center", va="top", color=INK, fontsize=8.5)
    ax.annotate("North (m)", _screen((100, 57, 0)), xytext=(19, -7),
                textcoords="offset points", ha="left", va="top", color=INK, fontsize=8.5)
    ax.annotate("Height (m)", _screen((0, 0, 60)), xytext=(-9, 7),
                textcoords="offset points", ha="left", va="bottom", color=INK, fontsize=8.5)


def _scene(ax):
    _base(ax)
    faces = _faces()
    for face, color in sorted(faces, key=lambda item: item[0][:, 2].mean()):
        ax.add_patch(Polygon(face[:, :2], closed=True, fc=color, ec=GREY, lw=0.6, zorder=2))

    cameras = np.column_stack((STARTS, np.full(len(STARTS), CAMERA)))
    for target, color, offset in zip(TARGETS, TARGET_COLORS, [(6, 0), (-6, 8), (-6, 6)]):
        top = np.r_[target.xy, target.height]
        ground = np.r_[target.xy, 0]
        projection = list(_visible_spans(top, ground, faces))
        ax.add_collection(LineCollection(projection, color=color, linewidth=0.75,
                                         linestyle="--", alpha=0.67, zorder=3))
        ground_xy = _screen(ground)
        if projection and np.allclose(projection[-1][1], ground_xy):
            ax.scatter(*ground_xy, marker="x", color=color, s=12, linewidth=0.85, zorder=3)
        sight_lines = [
            span for camera in cameras[observable(STARTS, target)]
            for span in _visible_spans(camera, top, faces)
        ]
        ax.add_collection(LineCollection(sight_lines, color=color, linewidth=0.85,
                                         alpha=0.88, zorder=4))
        top_xy = _screen(top)
        ax.scatter(*top_xy, marker="^", s=43, color=color, ec="white", lw=0.65, zorder=6)
        ax.annotate(f"{target.name}  {target.height:g} m", top_xy, xytext=offset,
                    textcoords="offset points", color=color, fontsize=8.5,
                    ha="left" if offset[0] > 0 else "right", va="center",
                    bbox=LABEL_BOX, zorder=7)

    for index, (x0, x1, y0, y1, height) in enumerate(BUILDINGS, 1):
        top_center = _screen(((x0 + x1) / 2, (y0 + y1) / 2, height))
        ax.text(*top_center, f"B{index}", color=INK, fontsize=8, ha="center", va="center",
                bbox=LABEL_BOX, zorder=5)
    locations = _screen(cameras)
    ax.scatter(*locations.T, marker="s", s=23, color=INK, ec="white", lw=0.6, zorder=6)
    for index, (point, offset) in enumerate(zip(locations, [(-4, -6), (7, -3), (6, 5),
                                                            (5, -7), (5, 7)]), 1):
        ax.annotate(f"V{index}", point, xytext=offset, textcoords="offset points",
                    ha="left" if offset[0] > 0 else "right", va="center",
                    color=INK, fontsize=8, bbox=LABEL_BOX, zorder=7)
    ax.set(xlim=(-12, 148), ylim=(-54, 82), aspect="equal")
    ax.axis("off")


def _routes(ax, station, router):
    paths = [router.path(origin, station) for origin in STARTS]
    ax.add_patch(Rectangle((RADIUS, RADIUS), 100 - 2 * RADIUS, 100 - 2 * RADIUS,
                           fill=False, edgecolor="#A6B2BB", lw=0.6, ls=(0, (3, 2)), zorder=1.5))
    for index, (corners, box) in enumerate(zip(np.split(router.vertices, len(BUILDINGS)), BUILDINGS), 1):
        x0, x1, y0, y1, _ = box
        ax.add_patch(Polygon(corners, closed=True, fc="#EAF0F3", ec="#A6B2BB", lw=0.6,
                             ls=(0, (3, 2)), zorder=2))
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="#C5CDD4", ec=GREY, lw=0.65, zorder=3))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, f"B{index}", ha="center", va="center",
                color=INK, fontsize=8.5, zorder=4)

    waypoints = np.unique(np.vstack(paths), axis=0)
    a, b = np.nonzero(np.triu(np.isfinite(router.direct_matrix(waypoints, waypoints)), k=1))
    ax.add_collection(LineCollection(np.stack((waypoints[a], waypoints[b]), axis=1),
                                     color="#BAC7D1", lw=0.47, alpha=0.64, zorder=1))
    styles = ["-", "-", "-", "-", (0, (2.2, 1.65))]
    for index, (path, color, style) in enumerate(zip(paths, ROUTE_COLORS, styles), 1):
        if index == 5:
            # The white underlay is narrower than V3 so both shared routes remain visible.
            ax.plot(*path.T, color="white", lw=1.55, ls=style, zorder=5)
        ax.plot(*path.T, color=color, ls=style, lw=1.0 if index == 5 else 1.8,
                zorder=6 + index / 10)
        ax.scatter(*path[0], marker="s", s=27, color=color, ec="white", lw=0.6, zorder=8)
    ax.scatter(*station, marker="*", s=85, color="#FCDF86", ec=INK, lw=0.8, zorder=9)
    ax.annotate("P*", station, xytext=(6, -4), textcoords="offset points",
                va="top", ha="left", color=INK, fontsize=9, weight="bold", bbox=LABEL_BOX, zorder=10)
    offsets = [(7, -4), (-7, -1), (6, 5), None, (4, 5)]
    for index, (path, color, offset) in enumerate(zip(paths, ROUTE_COLORS, offsets), 1):
        label = f"V{index}  {path_length(path):.1f} m"
        if offset is None:
            ax.annotate(label, path[0], xytext=(93, 84), textcoords="data",
                        ha="right", va="center", color=color, fontsize=8, zorder=10,
                        arrowprops={"arrowstyle": "-", "color": color, "lw": 0.65,
                                    "shrinkA": 3, "shrinkB": 4})
        else:
            ax.annotate(label, path[0], xytext=offset, textcoords="offset points",
                        ha="right" if index == 2 else "left", va="center",
                        color=color, fontsize=8, bbox=LABEL_BOX, zorder=10)
    ax.set(xlim=(0, 100), ylim=(0, 100), aspect="equal",
           xticks=range(0, 101, 20), yticks=range(0, 101, 20),
           xlabel="East (m)", ylabel="North (m)")
    ax.tick_params(labelsize=8)
    ax.grid(color=GRID, linewidth=0.55)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(GRID)


def make_figures(root: Path):
    results = json.loads((root / "results" / "summary.json").read_text(encoding="utf-8-sig"))
    station = np.array(results["q1"]["selected_station_m"])
    figure = plt.figure(figsize=(180 / 25.4, 96 / 25.4))
    left = figure.add_axes((0.036, 0.081, 0.56, 0.855))
    right = figure.add_axes((0.654, 0.174, 0.321, 0.735))
    _scene(left)
    _routes(right, station, Router())
    heading_y = 0.946
    figure.text(0.038, heading_y, "a", fontsize=11, weight="bold", color=INK)
    figure.text(0.069, heading_y, "Initial lines of sight", fontsize=9.5, color=INK)
    figure.text(0.630, heading_y, "b", fontsize=11, weight="bold", color=INK)
    figure.text(0.660, heading_y, "Routes to station P*", fontsize=9.5, color=INK)
    return {"fig00_scene_and_routes": figure}
