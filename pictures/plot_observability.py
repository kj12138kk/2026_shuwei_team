"""Map of T1 observability and detail views of T1 and T2 station geometry."""
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Patch, Polygon, Rectangle

from geometry import BUILDINGS, RADIUS, STARTS, TARGETS
from problem1 import _shadow_polygons

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GREY = "#687280"
INK = "#22272D"
PALE_GREEN = "#D9EEE7"
PALE_BLUE = "#E0EEF6"
PALE_ORANGE = "#F4D9CE"
PALE_GREY = "#EDF0F2"
BUILDING_FILL = "#CED4DA"
PT = 8.5


def _rounded_footprint(box):
    x0, x1, y0, y1, _ = box
    centers = ((x1, y1), (x0, y1), (x0, y0), (x1, y0))
    phi = np.linspace(0., np.pi / 2., 33)
    vertices = []
    for i, center in enumerate(centers):
        angle = phi + i * np.pi / 2.
        vertices.extend(np.asarray(center) + RADIUS * np.column_stack((np.cos(angle), np.sin(angle))))
    return vertices


def _building(ax, box, **style):
    x0, x1, y0, y1, _ = box
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, **style))


def _observability(ax, shadows, target, radius):
    ax.set_facecolor(PALE_GREY)
    admissible_field = Rectangle((RADIUS, RADIUS), 100. - 2. * RADIUS, 100. - 2. * RADIUS,
                                 facecolor=PALE_GREEN, edgecolor="none", zorder=0.1)
    ax.add_patch(admissible_field)
    for shadow in shadows:
        patch = Polygon(shadow, facecolor=PALE_ORANGE, edgecolor="none", zorder=0.2)
        ax.add_patch(patch)
        patch.set_clip_path(admissible_field.get_path(), admissible_field.get_transform())
    cap = Circle(target, radius, facecolor=PALE_BLUE, edgecolor="none", zorder=0.3)
    ax.add_patch(cap)
    cap.set_clip_path(admissible_field.get_path(), admissible_field.get_transform())
    for box in BUILDINGS:
        patch = Polygon(_rounded_footprint(box), facecolor=PALE_GREY, edgecolor="none", zorder=0.4)
        ax.add_patch(patch)
        patch.set_clip_path(admissible_field.get_path(), admissible_field.get_transform())
        _building(ax, box, facecolor=BUILDING_FILL, edgecolor=GREY, linewidth=.55, zorder=.5)


def _axes(ax, letter, xlim, ylim, xticks, yticks):
    ax.set(xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks)
    ax.set_aspect("equal", adjustable="box", anchor="W")
    ax.tick_params(direction="out", length=2.5, width=.55, pad=2, labelsize=PT, color=GREY)
    for edge in ("left", "bottom"):
        ax.spines[edge].set(linewidth=.55, color=GREY)
    for edge in ("top", "right"):
        ax.spines[edge].set_visible(False)
    ax.text(-.075, 1.02, letter, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10., weight="bold", color=INK)


def _target(ax, xy):
    ax.scatter(*xy, marker="^", s=33, color=ORANGE, edgecolors="white", linewidths=.5, zorder=5)


def _station(ax, xy, marker="*", color=GREEN, size=70):
    ax.scatter(*xy, marker=marker, s=size, color=color, edgecolors="white", linewidths=.6, zorder=5)


def _overview(ax, q1, shadows):
    target = np.asarray(q1["target_xy_m"])
    _observability(ax, shadows, target, q1["range_m"])
    for i, (x0, x1, y0, y1, _) in enumerate(BUILDINGS, 1):
        ax.text((x0 + x1) / 2., (y0 + y1) / 2., f"B{i}", ha="center", va="center",
                fontsize=PT, color=INK, zorder=2)
    ax.scatter(*STARTS.T, marker="s", s=18, color=BLUE, edgecolors="white", linewidths=.45, zorder=5)
    for i, start in enumerate(STARTS, 1):
        ax.annotate(f"V{i}", start, xytext=(4, 3), textcoords="offset points", color=INK,
                    fontsize=PT, zorder=6)
    _target(ax, target)
    ax.annotate("T1", target, xytext=(-2, 5), textcoords="offset points", ha="center",
                fontsize=PT, color=INK, zorder=6)
    _station(ax, q1["selected_station_m"], size=58)
    area = q1["continuous_areas_m2"]["observable"]
    ax.text(.035, .965, f"{area:,.1f} m² observable", transform=ax.transAxes,
            va="top", fontsize=PT, color=INK)
    _axes(ax, "a", (0, 100), (0, 100), np.arange(0, 101, 20), np.arange(0, 101, 20))
    ax.set_xlabel("East (m)", fontsize=PT, labelpad=3)
    ax.set_ylabel("North (m)", fontsize=PT, labelpad=3)


def _arc_detail(ax, q1, shadows):
    target = np.asarray(q1["target_xy_m"])
    radius = q1["range_m"]
    path = np.asarray(q1["path_xy_m"])
    selected = path[-1]
    _observability(ax, shadows, target, radius)
    ax.add_patch(Circle(target, radius, fill=False, linestyle=(0, (3, 2)), linewidth=.7,
                        edgecolor=GREY, zorder=2))
    for interval in q1["optimal_arc_intervals_deg"]:
        low, high = np.deg2rad([interval["start_deg"], interval["end_deg"]])
        phi = np.linspace(low, high, 241)
        coordinates = target + radius * np.column_stack((np.cos(phi), np.sin(phi)))
        ax.plot(*coordinates.T, color=GREEN, linewidth=1.5, solid_capstyle="round", zorder=3)
    ax.plot(*path.T, color=BLUE, linewidth=1.1, zorder=4)
    # Overplot just the final portion with an arrowhead to preserve the drawn polyline.
    final_link = selected - path[-2]
    ax.annotate("", xy=selected, xytext=selected - .30 * final_link,
                arrowprops={"arrowstyle": "-|>", "color": BLUE, "lw": 1.1, "shrinkB": 5., "shrinkA": 0.},
                zorder=4)
    _target(ax, target)
    _station(ax, path[0], marker="s", color=BLUE, size=23)
    _station(ax, selected)
    ax.annotate("T1", target, xytext=(3, 5), textcoords="offset points", fontsize=PT, color=INK, zorder=6)
    ax.annotate(q1["selected_vehicle"], path[0], xytext=(5, 2), textcoords="offset points",
                fontsize=PT, color=INK, zorder=6)
    ax.annotate(f"{q1['ready_time_s']:.2f} s", selected, xytext=(6, -6), textcoords="offset points",
                fontsize=PT, color=INK, va="top", zorder=6)
    ax.text(76., 65.8, "B2", fontsize=PT, color=INK, zorder=2)
    ax.text(79.9, 72.9, f"{radius:.2f} m radius", fontsize=PT, color=INK, zorder=2)
    _axes(ax, "b", (73.7, 96.7), (59.6, 79.1), [75, 80, 85, 90, 95], [60, 65, 70, 75])


def _intersection(ax, q2):
    target = np.asarray(q2["target_xy_m"])
    first, second = stations = np.asarray(q2["stations_m"])
    for box in BUILDINGS:
        _building(ax, box, facecolor=BUILDING_FILL, edgecolor=GREY, linewidth=.55, zorder=0.5)
    for station in stations:
        ax.plot(*np.vstack((station, target)).T, color=BLUE, linewidth=1.15, zorder=2)
    ax.plot(*stations.T, color=GREY, linewidth=.9, linestyle=(0, (4, 2)), zorder=2)
    rays = (stations - target) / np.linalg.norm(stations - target, axis=1)[:, None]
    a, b = .9 * rays
    angle_box = target + np.vstack((a, a + b, b))
    ax.plot(*angle_box.T, color=BLUE, linewidth=.7, zorder=3)
    ax.text(*(target + 1.8 * rays.sum(axis=0)), f"{q2['angle_deg']:.0f}°", color=INK,
            ha="center", va="center", fontsize=PT, zorder=4)
    _target(ax, target)
    _station(ax, first, marker="s", color=BLUE, size=24)
    _station(ax, second, marker="s", color=BLUE, size=24)
    ax.annotate(q2["vehicles"][0], first, xytext=(5, 0), textcoords="offset points",
                va="center", fontsize=PT, color=INK, zorder=6)
    ax.annotate(q2["vehicles"][1], second, xytext=(-5, -1), textcoords="offset points",
                ha="right", va="center", fontsize=PT, color=INK, zorder=6)
    ax.annotate("T2", target, xytext=(4, 3), textcoords="offset points", fontsize=PT, color=INK, zorder=6)
    midpoint = (first + second) / 2.
    ax.annotate(f"{q2['baseline_m']:.2f} m", midpoint, xytext=(-7, 7), textcoords="offset points",
                ha="right", fontsize=PT, color=INK, zorder=6)
    ax.text(18.6, 69., "B3", color=INK, fontsize=PT)
    ax.text(.97, .96, f"DRMS  {q2['minimum_drms_m']:.3f} m", transform=ax.transAxes,
            ha="right", va="top", fontsize=PT, color=INK)
    _axes(ax, "c", (.5, 23.5), (64.3, 82.1), [5, 10, 15, 20], [65, 70, 75, 80])
    ax.set_xlabel("East (m)", fontsize=PT, labelpad=3)


def make_figures(root: Path) -> dict[str, Figure]:
    data = json.loads((root / "results" / "summary.json").read_text(encoding="utf-8"))
    q1, q2 = data["q1"], data["q2_accuracy"]
    shadows = _shadow_polygons(TARGETS[0])
    figure = Figure(figsize=(180. / 25.4, 107. / 25.4), facecolor="white")
    field = figure.add_axes((.073, .195, .443, .748))
    arc = figure.add_axes((.625, .576, .341, .367))
    intersection = figure.add_axes((.625, .127, .341, .367))
    _overview(field, q1, shadows)
    _arc_detail(arc, q1, shadows)
    _intersection(intersection, q2)
    region_labels = (
        ("Observable", PALE_GREEN),
        ("Elevation limited", PALE_BLUE),
        ("Occluded", PALE_ORANGE),
        ("Vehicle excluded", PALE_GREY),
    )
    handles = [Patch(facecolor=color, edgecolor=GREY, linewidth=.45, label=label)
               for label, color in region_labels]
    figure.legend(handles=handles, ncol=2, loc="lower left", bbox_to_anchor=(.071, .025),
                  frameon=False, fontsize=PT, handlelength=1.1, columnspacing=1.25,
                  handletextpad=.55, borderaxespad=0., labelspacing=.45)
    return {"fig01_observability": figure}
