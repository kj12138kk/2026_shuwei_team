"""Q3: projected-normal probability, variance shares and station checks."""

import csv
import json

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from errors import probability


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GREY = "#687280"
LIGHT = "#DDE2E6"


def _tag(ax, letter):
    ax.text(-0.09, 1.04, letter, transform=ax.transAxes, weight="bold", size=11)


def _percent_axis(ax):
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.yaxis.grid(color=LIGHT, linewidth=0.55)
    ax.set_axisbelow(True)


def _distance_probability(ax, q3):
    distance = np.linspace(0, 100, 201)
    limit = q3["max_distance_m"]
    exact = 100 * np.array([probability(0.3 + 0.02 * s) for s in distance])
    ax.plot(distance, exact, color=BLUE, lw=1.5, label="Probability integral")
    ax.axhline(95, color=GREY, lw=0.8, ls="--")
    ax.axvline(limit, color=GREY, lw=0.8, ls=":")
    trials = q3["trials"]
    positions = np.array([row["distance_m"] for row in trials])
    estimate = 100 * np.array([row["monte_carlo"] for row in trials])
    interval = 100 * np.array([row["mc_95_interval"] for row in trials]).T
    ax.errorbar(
        positions, estimate, yerr=[estimate - interval[0], interval[1] - estimate],
        fmt="o", ms=3.3, color=ORANGE, mfc="white", mew=0.9,
        lw=0.8, capsize=1.2, label="Simulation and 95% CI", zorder=3,
    )
    ax.annotate(
        f"{limit:.2f} m", (limit, 99.5), xytext=(5, 0),
        textcoords="offset points", ha="left", va="bottom", color=GREY,
    )
    ax.set(xlim=(-2, 102), ylim=(83, 101.5), xticks=[0, 20, 40, 60, 80, 100],
           yticks=[85, 90, 95, 100], xlabel="Cumulative travel (m)",
           ylabel="Pointing success at 40 m")
    _percent_axis(ax)
    ax.legend(loc="lower left", handlelength=1.6, borderaxespad=0.25)


def _variance_shares(ax, q3, rows):
    travel = np.array([float(row["distance_m"]) for row in rows])
    for key, name, color, style in (
        ("position", "Position", BLUE, "-"),
        ("zero_offset", "Zero offset", ORANGE, "--"),
        ("control", "Control", GREEN, "-."),
    ):
        shares = [100 * float(row[key]) for row in rows]
        ax.plot(travel, shares, label=name, color=color, ls=style, lw=1.5)
    limit = q3["max_distance_m"]
    ax.axvline(limit, color=GREY, lw=0.8, ls=":")
    ax.scatter(limit, 100 * q3["variance_shares"]["position"], color=BLUE, s=13, zorder=3)
    ax.annotate(
        f'{q3["variance_shares"]["position"]:.1%}',
        (limit, 100 * q3["variance_shares"]["position"]),
        xytext=(-6, 4), textcoords="offset points", ha="right", color=BLUE,
    )
    ax.set(xlim=(-2, 102), ylim=(0, 103), xticks=[0, 20, 40, 60, 80, 100],
           yticks=[0, 25, 50, 75, 100], xlabel="Cumulative travel (m)",
           ylabel="First-order variance share")
    _percent_axis(ax)
    ax.legend(loc="center right", handlelength=1.5, labelspacing=0.3)


def _station_checks(ax, checks):
    maximum = max(row["sigma_over_range"] for rows in checks["schedules"].values() for row in rows)
    ratio = np.linspace(0, np.ceil(maximum * 100 + 1) / 100, 220)
    ax.plot(ratio * 100, 100 * np.array([probability(40 * x) for x in ratio]), color=GREY, lw=1)
    ratio_limit = 100 * checks["ratio_limit"]
    ax.axhline(95, color=GREY, lw=0.8, ls="--")
    ax.axvline(ratio_limit, color=GREY, lw=0.8, ls=":")
    ax.annotate(
        f"{ratio_limit:.2f}%", (ratio_limit, 26), xytext=(4, 0),
        textcoords="offset points", ha="left", va="bottom", color=GREY,
    )
    for key, label, color, marker in (
        ("nominal", "Nominal schedule", BLUE, "o"),
        ("drms_0.50", "0.50 m DRMS schedule", ORANGE, "s"),
        ("t2_accuracy_pair", "T2 minimum-DRMS pair", GREEN, "^"),
    ):
        rows = checks["schedules"][key]
        passes = sum(row["passes_95"] for row in rows)
        ax.scatter(
            [100 * row["sigma_over_range"] for row in rows],
            [100 * row["success_probability"] for row in rows],
            label=f"{label} ({passes}/{len(rows)} pass)",
            marker=marker, s=21, facecolors="white", edgecolors=color, linewidths=1.1, zorder=4,
        )
    ax.set(xlim=(-0.2, 100 * ratio[-1]), ylim=(25, 104),
           yticks=[30, 50, 70, 95], xlabel="Position SD / target range (%)",
           ylabel="Pointing success per visit")
    _percent_axis(ax)
    ax.legend(loc="upper right", bbox_to_anchor=(0.995, 0.86), handletextpad=0.4,
              labelspacing=0.38, borderaxespad=0)


def make_figures(root):
    with (root / "results" / "summary.json").open(encoding="utf-8-sig") as handle:
        q3 = json.load(handle)["q3"]
    with (root / "results" / "q3_variance_shares.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    fig = plt.figure(figsize=(180 / 25.4, 117 / 25.4))
    grid = fig.add_gridspec(2, 2, left=0.105, right=0.976, bottom=0.101, top=0.931,
                            hspace=0.60, wspace=0.39)
    distance = fig.add_subplot(grid[0, 0])
    variance = fig.add_subplot(grid[0, 1])
    stations = fig.add_subplot(grid[1, :])
    _distance_probability(distance, q3)
    _variance_shares(variance, q3, rows)
    _station_checks(stations, q3["station_checks"])
    for ax, letter in zip((distance, variance, stations), "abc"):
        _tag(ax, letter)
    return {"fig04_pointing": fig}
