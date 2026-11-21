"""Q2: first-order geometry and nonlinear validation of bearing intersection."""

import csv
import json

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, ScalarFormatter
import numpy as np


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GREY = "#687280"
LIGHT = "#DDE2E6"


def _tag(ax, letter):
    ax.text(-0.14, 1.05, letter, transform=ax.transAxes, weight="bold", size=11)


def _style_angle_axis(ax):
    ax.set(xlim=(0, 180), xticks=[0, 30, 60, 90, 120, 150, 180])
    ax.set_axisbelow(True)
    ax.yaxis.grid(color=LIGHT, linewidth=0.55)


def _theory(ax, sweep, t2):
    angle = np.linspace(sweep[0]["angle_deg"], sweep[-1]["angle_deg"], 721)
    sigma = np.deg2rad(sweep[0]["bearing_sd_deg"])
    for r, color, style in ((10, BLUE, "-"), (20, ORANGE, "--"), (40, GREEN, "-.")):
        drms = sigma * np.hypot(r, r) / np.sin(np.deg2rad(angle))
        ax.plot(angle, drms, color=color, lw=1.4, ls=style, label=f"{r} m each")
    t2_drms = t2["minimum_drms_m"]
    ax.scatter(t2["angle_deg"], t2_drms, s=22, color="#202327", marker="D", zorder=3)
    ax.annotate(
        f'T2  {t2_drms:.3f} m', (t2["angle_deg"], t2_drms),
        xytext=(8, -3), textcoords="offset points", va="center",
    )
    ax.set(yscale="log", ylim=(0.10, 100), xlabel="Crossing angle (°)",
           ylabel="First-order DRMS (m)")
    ax.set_yticks([0.1, 1, 10, 100], labels=["0.1", "1", "10", "100"])
    _style_angle_axis(ax)
    ax.legend(title="Station-to-target range", title_fontsize=9, loc="upper center",
              handlelength=1.7, labelspacing=0.35)


def _errors(ax, angles, sweep):
    ax.plot(angles, [row["unfiltered_p95_m"] for row in sweep], color=ORANGE,
            marker="o", ms=2.9, lw=1.3, label="Nonlinear")
    ax.plot(angles, [row["linear_p95_m"] for row in sweep], color=BLUE,
            marker="s", ms=2.6, lw=1.0, ls="--", label="Linear")
    ax.set(yscale="log", ylim=(0.75, 220), yticks=[1, 10, 100],
           ylabel="Unfiltered error P95 (m)")
    ax.yaxis.set_major_formatter(ScalarFormatter())
    _style_angle_axis(ax)
    ax.tick_params(labelbottom=False)
    ax.legend(loc="upper center", handlelength=1.7, labelspacing=0.25, borderaxespad=0.15)


def _rejection(ax, angles, sweep):
    rate = [100 * row["rejection_rate"] for row in sweep]
    ax.plot(angles, rate, color=GREY, lw=1.0, marker="o", ms=3.1)
    ax.set(ylim=(-4, 106), yticks=[0, 50, 100], xlabel="Crossing angle (°)",
           ylabel="Rejected samples")
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    _style_angle_axis(ax)


def make_figures(root):
    with (root / "results" / "summary.json").open(encoding="utf-8-sig") as handle:
        t2 = json.load(handle)["q2_accuracy"]
    with (root / "results" / "q2_intersection_errors.csv").open(encoding="utf-8-sig", newline="") as handle:
        sweep = [
            {key: float(row[key]) for key in (
                "angle_deg", "bearing_sd_deg", "unfiltered_p95_m", "linear_p95_m", "rejection_rate",
            )}
            for row in csv.DictReader(handle)
            if float(row["range_1_m"]) == float(row["range_2_m"]) == 20.0
        ]
    sweep.sort(key=lambda row: row["angle_deg"])
    angles = [row["angle_deg"] for row in sweep]

    fig = plt.figure(figsize=(180 / 25.4, 100 / 25.4))
    grid = fig.add_gridspec(2, 2, left=0.105, right=0.975, bottom=0.135, top=0.915,
                            wspace=0.40, hspace=0.32)
    theory = fig.add_subplot(grid[:, 0])
    errors = fig.add_subplot(grid[0, 1])
    rejection = fig.add_subplot(grid[1, 1], sharex=errors)
    _theory(theory, sweep, t2)
    _errors(errors, angles, sweep)
    _rejection(rejection, angles, sweep)
    for ax, letter in zip((theory, errors, rejection), "abc"):
        _tag(ax, letter)
    return {"fig02_intersection": fig}
