"""Figure 3: vehicle reuse, dispatch comparisons and nominal timing bounds."""

import csv
import json
from collections import Counter
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#687280"
INK = "#202833"

PHASE_STYLES = {
    "travel": {"facecolor": "#AEB8C3", "edgecolor": GRAY, "linewidth": .35},
    "slew": {"facecolor": "#E0E6EC", "edgecolor": GRAY, "linewidth": .35},
    "wait_sync": {"facecolor": "white", "edgecolor": "#A1A9B2", "linewidth": .6},
    "observe": {"facecolor": BLUE, "edgecolor": "white", "linewidth": .35},
    "reset": {"facecolor": "#E0E6EC", "edgecolor": GRAY, "linewidth": .35, "hatch": "///"},
}
POLICIES = {
    "optimized": ("Optimized", BLUE, "o"),
    "nearest": ("Nearest", ORANGE, "s"),
    "earliest": ("Earliest", GRAY, "^"),
}
TIMELINE_CASES = (("los_0.00", "Nominal"), ("drms_0.50", "DRMS cap 0.50 m"))


def _csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _panel(ax, letter, heading=None):
    ax.text(-.12, 1.075, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", ha="left", va="bottom", color=INK)
    if heading:
        ax.text(0., 1.075, heading, transform=ax.transAxes, fontsize=9.5,
                ha="left", va="bottom", color=INK)


def _timeline(ax, events, limit, show_vehicle_label):
    counts = Counter(event["vehicle"] for event in events if event["phase"] == "observe")
    for event in events:
        style = PHASE_STYLES.get(event["phase"])
        if style is None:
            continue
        start, end = float(event["start_s"]), float(event["end_s"])
        if end <= start:
            continue
        row = int(event["vehicle"][1:])
        ax.barh(row, end - start, left=start, height=.54, **style, zorder=3)
        if event["phase"] == "observe":
            ax.text((start + end) / 2., row, event["target"], color="white",
                    fontsize=8, fontweight="bold", ha="center", va="center", zorder=4)
    ax.set_xlim(0., limit)
    ax.set_ylim(5.55, .45)
    ax.set_yticks(range(1, 6), labels=[f"V{i}" for i in range(1, 6)])
    for label in ax.get_yticklabels():
        if counts[label.get_text()] > 1:
            label.set_fontweight("bold")
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.grid(axis="x", color="#DDE3E8", lw=.55)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Time (s)")
    if show_vehicle_label:
        ax.set_ylabel("Vehicle")


def _latest_times(ax, experiments):
    offsets = {"optimized": .20, "nearest": 0., "earliest": -.20}
    case_labels = []
    handles = []
    for policy, (label, color, marker) in POLICIES.items():
        handles.append(Line2D([], [], marker=marker, linestyle="none",
                              markerfacecolor=color, markeredgecolor="white",
                              markeredgewidth=.45, markersize=5, label=label))
    for y, (experiment, title) in zip((1., 0.), TIMELINE_CASES):
        case_labels.append(title if experiment == "los_0.00" else "0.50 m cap")
        for policy, (_, color, marker) in POLICIES.items():
            time = experiments.get((experiment, policy))
            if time is None:
                continue
            row = y + offsets[policy]
            ax.hlines(row, 0., time, color=color, lw=.7, alpha=.45, zorder=2)
            ax.plot(time, row, marker=marker, color=color, markeredgecolor="white",
                    markeredgewidth=.45, markersize=5.5, zorder=3)
            ax.annotate(f"{time:.2f}", (time, row), xytext=(4, 0),
                        textcoords="offset points", fontsize=8,
                        ha="left", va="center", color=color)
    max_time = max(experiments.values())
    ax.set_xlim(0., 5. * ceil((max_time + 4.) / 5.))
    ax.set_ylim(-.40, 1.42)
    ax.set_yticks((1., 0.), labels=case_labels)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Latest start (s)")
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-.03, -.34),
              ncol=3, handletextpad=.2, columnspacing=.9, borderaxespad=0.,
              fontsize=8, handlelength=1.15)


def _sensitivity(ax, experiments, schedule):
    margins = (0., .05, .10)
    nominal = schedule["max_start_s"]
    feasible_ms = [
        1000. * (experiments[f"los_{margin:.2f}", "optimized"] - nominal)
        for margin in margins
    ]
    bound_ms = 1000. * (schedule["continuous_lower_bound_s"] - nominal)
    ax.axhline(0., color="#C0C7CE", lw=.65, linestyle="--", zorder=1)
    ax.vlines(0., bound_ms, 0., color=GREEN, lw=.85, linestyle="--", zorder=2)
    feasible, = ax.plot(margins, feasible_ms, color=BLUE, marker="o", lw=1.,
                        markeredgewidth=.45, markeredgecolor="white", markersize=5.5,
                        label="Feasible plan", zorder=3)
    bound, = ax.plot(0., bound_ms, linestyle="none", color=GREEN, marker="D",
                     markeredgecolor="white", markeredgewidth=.45, markersize=5.4,
                     label="Nominal lower bound", zorder=3)
    points = [(x, y, BLUE) for x, y in zip(margins, feasible_ms)] + [(0., bound_ms, GREEN)]
    for x, y, color in points:
        ax.annotate("0.0" if abs(y) < .05 else f"{y:+.1f}", (x, y), xytext=(0, 5),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, color=color, zorder=4)
    ax.set_xlim(-.014, .114)
    ax.set_ylim(min(-10., bound_ms) - 13., max(feasible_ms) + 19.)
    ax.set_xticks(margins, labels=("0", "0.05", "0.10"))
    ax.set_xlabel("Required LOS clearance (m)")
    ax.set_ylabel("Change in latest start (ms)")
    ax.legend(handles=[feasible, bound], loc="upper left", bbox_to_anchor=(-.02, -.34),
              ncol=2, handletextpad=.4, columnspacing=1., borderaxespad=0.,
              fontsize=8, handlelength=1.3)


def make_figures(root: Path) -> dict[str, Figure]:
    results = root / "results"
    schedule = json.loads((results / "summary.json").read_text(encoding="utf-8"))["q2_schedule"]
    experiments = {
        (row["experiment"], row["policy"]): float(row["max_start_s"])
        for row in _csv(results / "q2_schedule_experiments.csv")
    }
    events = _csv(results / "q2_experiment_timelines.csv")
    cases = [
        [row for row in events if row["scenario"] == experiment and row["policy"] == "optimized"
         and row["phase"] in PHASE_STYLES]
        for experiment, _ in TIMELINE_CASES
    ]
    limit = 5. * ceil(max(float(event["end_s"]) for case in cases for event in case) / 5.)

    fig = plt.figure(figsize=(180 / 25.4, 148 / 25.4))
    grid = fig.add_gridspec(2, 2, left=.14, right=.967, bottom=.158, top=.835,
                           wspace=.52, hspace=.68, height_ratios=(1., .95))
    first, second = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    latest, sensitivity = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    for pos, (ax, case) in enumerate(zip((first, second), cases)):
        _panel(ax, "ab"[pos], TIMELINE_CASES[pos][1])
        _timeline(ax, case, limit, show_vehicle_label=pos == 0)
    _panel(latest, "c")
    _latest_times(latest, experiments)
    _panel(sensitivity, "d")
    _sensitivity(sensitivity, experiments, schedule)

    labels = ("Drive", "Point", "Wait", "Observe", "Reset")
    handles = [Patch(**style, label=label) for style, label in zip(PHASE_STYLES.values(), labels)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .965),
               ncol=5, fontsize=9, handletextpad=.43, columnspacing=1.0,
               borderaxespad=0., handlelength=1.2)
    return {"fig03_schedule": fig}
