"""Numerical bounds and report figures."""
from itertools import permutations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
from scipy.spatial import cKDTree

from disturbances import plan_events, scenarios
from errors import probability
from geometry import BUILDINGS, CAMERA, RADIUS, SPEED, STARTS, TARGETS, observable, pointing

COLORS = {"T1": "#0b7491", "T2": "#bc5536", "T3": "#62882c"}


def _observation_patch(step):
    axis = np.arange(step / 2., 100., step)
    points = np.column_stack([a.ravel() for a in np.meshgrid(axis, axis)])
    return observable(points, TARGETS[0]).reshape(len(axis), len(axis))


def _q1_map(out, station):
    steps = (.5, .25, .125)
    areas = dict()
    for step in steps:
        visible = _observation_patch(step)
        areas[str(step)] = float(visible.sum() * step * step)
    fig, ax = plt.subplots(figsize=(7.0, 6.25), constrained_layout=True)
    ax.imshow(np.ma.masked_where(~visible, visible), origin="lower", extent=(0, 100, 0, 100),
              cmap=matplotlib.colors.ListedColormap(["#bfe2dd"]), interpolation="nearest")
    ax.add_patch(Rectangle((RADIUS, RADIUS), 100 - 2 * RADIUS, 100 - 2 * RADIUS,
                          linewidth=1, ls="--", edgecolor="#767676", facecolor="none"))
    for n, (x0, x1, y0, y1, _) in enumerate(BUILDINGS, 1):
        ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, facecolor="#656f7c", edgecolor="white"))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, f"B{n}", color="white", ha="center", va="center")
    ax.scatter(*STARTS.T, marker="s", s=20, c="#485466", label="Vehicle initial position")
    ax.scatter(*TARGETS[0].xy, marker="^", s=85, c="#bf5240", label="T1 ground projection")
    ax.scatter(*station, marker="*", s=160, c="#053e53", label="Selected optimal station")
    ax.plot([station[0], TARGETS[0].xy[0]], [station[1], TARGETS[0].xy[1]], color="#053e53", lw=1)
    ax.set(xlabel="East x (m)", ylabel="North y (m)", title="T1 observable region (shaded)", xlim=(0, 100), ylim=(0, 100))
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.10), ncol=2, fontsize=8)
    fig.savefig(out / "q1_observable.png", dpi=180)
    plt.close(fig)
    return areas


def _hidden_cell(centers, half, target, box):
    """Certify that all points in a cell share a blocked ray parameter."""
    x0, x1, y0, y1, height = box
    low, high = np.array([x0, y0]), np.array([x1, y1])
    left, right = centers - half, centers + half
    offset = np.column_stack((left - low, high - right))
    slope = np.column_stack((target.xy - left, right - target.xy))
    flat = np.abs(slope) < 1e-12
    roots = -offset / np.where(flat, 1., slope)
    enter = np.maximum(0., np.max(np.where(slope > 1e-12, roots, -np.inf), axis=1))
    cap = min(1., (height - CAMERA) / (target.height - CAMERA))
    leave = np.minimum(cap, np.min(np.where(slope < -1e-12, roots, np.inf), axis=1))
    return (enter + 1e-10 < leave) & ~np.any(flat & (offset < 0.), axis=1)


def continuous_lower_bound(step, upper):
    """Relax travel obstacles, angle separation and all missions except one reuse chain."""
    edges = np.linspace(RADIUS, 100. - RADIUS, int(np.ceil((100. - 2 * RADIUS) / step)) + 1)
    half = (edges[1] - edges[0]) / 2.
    axis = (edges[1:] + edges[:-1]) / 2.
    centers = np.column_stack([a.ravel() for a in np.meshgrid(axis, axis)])
    radius = np.sqrt(2.) * half
    min_slew = []
    for target in TARGETS:
        delta = np.abs(centers - target.xy)
        max_range = np.linalg.norm(delta + half, axis=1)
        pan, _, _ = pointing(centers, target)
        range_center = np.linalg.norm(delta, axis=1)
        error = np.rad2deg(np.arcsin(np.minimum(1., radius / np.maximum(range_center, 1e-12))))
        error = np.where(range_center <= radius, 180., error)
        pan_lower = np.maximum(np.abs(pan) - error, 0.)
        tilt_lower = np.rad2deg(np.arctan2(target.height - CAMERA, max_range))
        possible = max_range >= target.min_range - 1e-10
        for box in BUILDINGS:
            x0, x1, y0, y1, _ = box
            deficit = np.maximum(np.maximum(np.array([x0, y0]) - (centers - half),
                                            centers + half - np.array([x1, y1])), 0.)
            possible &= np.linalg.norm(deficit, axis=1) >= RADIUS - 1e-10
            possible &= ~_hidden_cell(centers, half, target, box)
        min_slew.append(np.where(possible, np.maximum(pan_lower, tilt_lower) / 30., np.inf))
    min_slew = np.array(min_slew)
    starts_drive = np.linalg.norm(np.maximum(np.abs(STARTS[:, None, :] - centers) - half, 0.), axis=-1) / SPEED
    first_ready = starts_drive[None] + min_slew[:, None, :]
    individual = first_ready.min(axis=2)
    other_earliest = np.array([[min(np.delete(col, v)) for v in range(len(STARTS))] for col in individual])
    bound_by_pair = {}
    bound_by_chain = {}
    for j, k in permutations(range(len(TARGETS)), 2):
        arrival = np.maximum(first_ready[j], other_earliest[j, :, None])
        prefix = arrival + 5. + min_slew[j]
        chain = _bounded_chain(centers, half, prefix, min_slew[k], upper)
        name = TARGETS[j].name + "->" + TARGETS[k].name
        bound_by_pair[name] = float(chain.min())
        bound_by_chain[name] = chain.tolist()
    return dict(cell_actual_m=float(2 * half), lower_bound_s=min(bound_by_pair.values()),
                by_target_pair_s=bound_by_pair, by_target_pair_vehicle_s=bound_by_chain)


def _bounded_chain(centers, half, prefix, second_slew, upper):
    """Skip cell pairs certified unable to improve the known feasible upper bound."""
    global_slew = second_slew.min()
    prefix_min = prefix.min(axis=0)
    active = np.flatnonzero(prefix_min + global_slew < upper)
    chain = np.minimum(prefix.min(axis=1) + global_slew, upper)
    if not len(active):
        return prefix.min(axis=1) + global_slew
    valid = np.flatnonzero(np.isfinite(second_slew) & (second_slew <= upper - prefix_min[active].min()))
    tree = cKDTree(centers[valid])
    transfer = np.full(len(active), np.inf)
    for start in range(0, len(active), 128):
        block = active[start:start + 128]
        r = SPEED * (upper - prefix_min[block] - global_slew) + np.sqrt(8.) * half + 1e-10
        choices = tree.query_ball_point(centers[block], r)
        for i, (p, candidates) in enumerate(zip(block, choices), start):
            if not candidates:
                continue
            q = valid[candidates]
            separation = np.maximum(np.abs(centers[p] - centers[q]) - 2 * half, 0.)
            transfer[i] = np.min(np.linalg.norm(separation, axis=1) / SPEED + second_slew[q])
    eligible = prefix.min(axis=1) + global_slew < upper
    chain[eligible] = np.minimum((prefix[eligible][:, active] + transfer).min(axis=1), upper)
    return chain


def plot_schedule(out, optimized, nearest):
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.1), sharex=True, constrained_layout=True)
    shading = {"travel": .44, "slew": .33, "wait_sync": .11, "observe": 1., "reset": .2}
    for label, plan, ax in zip(("Optimized", "Nearest assignment"), (optimized, nearest), axes):
        for event in plan_events(plan):
            width = event.end - event.begin
            if width <= 0:
                continue
            color = COLORS[event.target]
            ax.barh(event.vehicle + 1, width, left=event.begin, height=.7, color=color, alpha=shading[event.phase])
            if event.phase == "observe":
                ax.text(event.begin + 2.5, event.vehicle + 1, event.target, ha="center", va="center", color="white")
        ax.set(yticks=np.arange(1, 6), yticklabels=[f"V{n}" for n in range(1, 6)], title=label)
        ax.grid(axis="x", alpha=.15)
        ax.invert_yaxis()
    axes[-1].set_xlabel("Time (s)")
    names = ("Travel", "Slew", "Wait", "Observe", "Reset")
    handles = [Patch(facecolor="#41566a", alpha=value, label=label) for value, label in zip(shading.values(), names)]
    axes[-1].legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.20), ncol=5, fontsize=8)
    fig.savefig(out / "q2_schedule.png", dpi=180)
    plt.close(fig)


def plot_probability(out, result):
    s = np.linspace(0, 95, 101)
    y = [probability(.3 + .02 * t) for t in s]
    fig, ax = plt.subplots(figsize=(7, 4.1), constrained_layout=True)
    ax.plot(s, y, c="#0b7491", lw=2, label="Nonlinear integration")
    sample = [t for t in result["trials"] if t["distance_m"] > 0]
    ax.errorbar([t["distance_m"] for t in sample], [t["monte_carlo"] for t in sample],
                yerr=[t["mc_95_halfwidth"] for t in sample], fmt="o", ms=3, c="#bc5536", label="Monte Carlo, 95% interval")
    ax.axhline(.95, c="#6b6b6b", ls="--", lw=1)
    ax.axvline(result["max_distance_m"], c="#6b6b6b", ls=":", lw=1)
    ax.axvline(80., c="#6b6b6b", ls=":", lw=1)
    ax.set(xlabel="Cumulative travel distance (m)", ylabel="Probability |bearing error| <= 5 degrees",
           title="Horizontal pointing reliability")
    ax.legend(loc="lower left")
    fig.savefig(out / "q3_probability.png", dpi=180)
    plt.close(fig)


def plot_scenarios(out, records):
    _, design = scenarios()
    kinds = list(dict.fromkeys(s.kind for s in design if s.kind != "nominal"))
    policies = ("retain", "retain_plus", "replan")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True)
    x = np.arange(len(kinds))
    for j, (policy, color) in enumerate(zip(policies, ("#a3aab4", "#bf6944", "#187e83"))):
        group = [[r for r in records if r["kind"] == kind and r["policy"] == policy] for kind in kinds]
        for ax, metric in zip(axes, ("eta", "t_last_s")):
            ax.bar(x + .24 * (j - 1), [np.mean([r[metric] for r in g]) for g in group], width=.23,
                   color=color, label=policy.replace("_", " "))
    for ax in axes:
        ax.set(xticks=x, xticklabels=[k.replace("+", "\n+") for k in kinds])
        ax.grid(axis="y", alpha=.18)
    axes[0].set(title="Mean completion ratio", ylim=(0, 1.07))
    axes[1].set(title="Mean latest start with 60 s penalty", ylabel="Seconds")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.savefig(out / "q4_comparison.png", dpi=180)
    plt.close(fig)
