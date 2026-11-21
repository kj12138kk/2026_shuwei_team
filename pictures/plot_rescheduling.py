"""Figure 5 from the published question 4 summary and event tables."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


POLICIES = {
    "retain": ("Retain", "#687280", "s"),
    "retain_plus": ("Retain + append", "#D55E00", "^"),
    "repair": ("Local repair", "#009E73", "o"),
    "replan": ("Best policy", "#0072B2", "D"),
}
GROUPS = {
    "outage": "Outage",
    "failure": "Failure",
    "failure+outage": "Failure + outage",
    "new": "New alert",
    "new+outage": "Alert + outage",
    "new+failure": "Alert + failure",
    "new+failure+outage": "All three events",
}
CASE = "N270_50_V5_R30"
MODEL = "drms_0.50"
PHASES = {
    "travel": ("Travel", "#687280"),
    "slew": ("Slew / reset", "#AAB2BD"),
    "wait_sync": ("Wait", "#E1E5EA"),
    "observe": ("Observe", "#009E73"),
}


def _rows(root: Path, filename: str) -> list[dict[str, str]]:
    with (root / "results" / filename).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _letter(ax, letter: str, x: float = 0.0, y: float = 1.11) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, va="bottom",
            ha="left", fontweight="bold", fontsize=10)


def _policy_handles() -> list[Line2D]:
    return [
        Line2D([], [], linestyle="none", marker=marker, markersize=5,
               color=color, label=label)
        for label, color, marker in POLICIES.values()
    ]


def _by_disturbance(ax, rows: list[dict[str, str]]) -> None:
    data = {(row["kind"], row["policy"]): row for row in rows}
    ticks = np.arange(len(GROUPS))[::-1]
    labels = [
        f"{label} ({data[kind, 'retain']['scenarios']})"
        for kind, label in GROUPS.items()
    ]
    for offset, (policy, (_, color, shape)) in zip(
            (-0.28, -0.095, 0.095, 0.28), POLICIES.items()):
        latest = [float(data[kind, policy]["mean_t_last_s"]) for kind in GROUPS]
        ax.plot(latest, ticks + offset, linestyle="none", marker=shape,
                markersize=4.4, color=color, markeredgecolor="white", markeredgewidth=0.4)
    for y in ticks:
        ax.axhline(y, color="#EBEDF0", lw=0.5, zorder=0)
    ax.axvline(60, color="#AAB2BD", ls="--", lw=0.7, zorder=0)
    ax.set(xlim=(0, 64), ylim=(-0.65, len(GROUPS) - 0.35),
           xticks=[0, 20, 40, 60], yticks=ticks, yticklabels=labels,
           xlabel="Mean latest start (s, with penalty)")
    ax.tick_params(axis="y", length=0, pad=4)
    _letter(ax, "a", -0.46)


def _pressure(ax, rows: list[dict[str, str]]) -> None:
    data = {(row["model"], row["policy"]): row for row in rows}
    order = list(POLICIES)
    ticks = np.arange(len(order))[::-1]
    for y, policy in zip(ticks, order):
        _, color, _ = POLICIES[policy]
        for model, dy, fill in (("nominal", 0.13, "white"), (MODEL, -0.13, color)):
            row = data[model, policy]
            n, total = int(row["all_started"]), int(row["scenarios"])
            ax.plot(n, y + dy, marker="o", markersize=4.8, linestyle="none",
                    markerfacecolor=fill, markeredgecolor=color, markeredgewidth=1)
            if n and dy < 0:
                ax.annotate(str(n), (n, y + dy), xytext=(5, 0),
                            textcoords="offset points", va="center", color=color, fontsize=8)
            if total != 40:
                raise ValueError("Pressure panel expects the common 40-scenario design.")
    for y in ticks:
        ax.axhline(y, color="#EBEDF0", lw=0.5, zorder=0)
    ax.set(xlim=(-2, 44), ylim=(-0.58, len(order) - 0.35), xticks=[0, 10, 20, 30, 40],
           yticks=ticks, yticklabels=[POLICIES[p][0] for p in order],
           xlabel="Scenarios starting all alerts (of 40)")
    ax.tick_params(axis="y", length=0, pad=4)
    ax.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", markersize=4, color="#46505C",
                   markerfacecolor="white", label="Nominal"),
            Line2D([], [], marker="o", linestyle="none", markersize=4, color="#46505C",
                   label=r"DRMS $\leq 0.50$ m"),
        ],
        loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2,
        borderaxespad=0, handletextpad=0.2, columnspacing=0.8, fontsize=8,
    )
    _letter(ax, "b", -0.39, 1.035)


def _outage(ax, fault: float, recovered: float) -> None:
    ax.axvspan(fault, recovered, color="#ECEFF2", alpha=0.75, zorder=-2, lw=0)
    for time in (fault, recovered):
        ax.axvline(time, color="#687280", lw=0.7, ls="--", zorder=1)


def _timeline(ax, rows: list[dict[str, str]], fault: float, recovered: float) -> None:
    _outage(ax, fault, recovered)
    v_order = ("V1", "V2", "V3", "V4", "V5")
    lanes = {name: 4 - index for index, name in enumerate(v_order)}
    for row in rows:
        phase = row["phase"]
        y = lanes[row["vehicle"]]
        begin, end = float(row["start_s"]), min(float(row["end_s"]), 60)
        if phase == "failed":
            ax.plot(begin, y, marker="x", color="#303942", mew=1, ms=5, zorder=5)
            continue
        if phase in ("reset", "abort_reset"):
            phase = "slew"
        if phase not in PHASES or end <= begin:
            continue
        _, color = PHASES[phase]
        ax.barh(y, end - begin, left=begin, height=0.61, color=color, lw=0,
                alpha=0.78 if phase == "observe" else 1.0,
                zorder=3 if phase == "observe" else 2)
        if phase == "observe":
            target = row["target"] if row["target"].startswith("T") else "New"
            ax.text((begin + end) / 2, y, target, fontsize=8,
                    color="#111B20", ha="center", va="center", zorder=4)
    top = ax.get_xaxis_transform()
    ax.text(fault, 1.035, "Failure / new alert", transform=top,
            ha="left", va="bottom", fontsize=8, color="#414954")
    ax.text(recovered, 1.035, "Communication restored", transform=top,
            ha="left", va="bottom", fontsize=8, color="#414954")
    ax.set(ylim=(-0.52, 4.52), yticks=list(lanes.values()), yticklabels=v_order)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.spines["bottom"].set_visible(False)
    _letter(ax, "c", -0.04)


def _start_steps(ax, cases: dict[str, dict[str, str]],
                 fault: float, recovered: float) -> None:
    _outage(ax, fault, recovered)
    fieldnames = ("t1_s", "t2_s", "t3_s", "new_alert_s")
    for policy in ("retain", "retain_plus", "repair"):
        row = cases[policy]
        events = sorted(float(row[k]) for k in fieldnames if row[k] and float(row[k]) <= 60)
        _, color, marker = POLICIES[policy]
        count = len(events)
        ax.step([0, *events, 60], [0, *range(1, count + 1), count],
                where="post", lw=1.25, color=color, zorder=2)
        ax.plot(events, range(1, count + 1), marker=marker, color=color,
                linestyle="none", ms=3.0, zorder=3)
        ax.annotate(f"{count}/4", (60, count), xytext=(4, 0), textcoords="offset points",
                    color=color, ha="left", va="center", fontsize=8)
    ax.set(xlim=(0, 60), xticks=np.arange(0, 61, 10), ylim=(-0.16, 4.33), yticks=np.arange(5),
           xlabel="Time since dispatch (s)", ylabel="Alerts started")
    ax.grid(axis="y", lw=0.5, alpha=0.45, color="#C9CED5")
    _letter(ax, "d", -0.04)


def make_figures(root: Path) -> dict[str, Figure]:
    """Return the summary and a constrained-case replay at 180 mm wide."""
    ordinary = _rows(root, "q4_aggregate.csv")
    pressure = _rows(root, "q4_pressure_aggregate.csv")
    cases = {row["policy"]: row for row in _rows(root, "q4_pressure_scenarios.csv")
             if row["model"] == MODEL and row["scenario"] == CASE}
    events = [row for row in _rows(root, "q4_pressure_timelines.csv")
              if row["model"] == MODEL and row["scenario"] == CASE and row["policy"] == "repair"]
    if not events or not POLICIES.keys() <= cases.keys():
        raise ValueError(f"Missing event or scenario data for {CASE} ({MODEL}).")

    selected = cases["repair"]
    fault = min(float(row["start_s"]) for row in events if row["phase"] == "failed")
    recovery = float(selected["recovery_s"])
    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.linewidth": 0.65,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }):
        figure = plt.figure(figsize=(180 / 25.4, 147 / 25.4))
        a = figure.add_axes([0.200, 0.679, 0.345, 0.252])
        b = figure.add_axes([0.718, 0.679, 0.251, 0.252])
        c = figure.add_axes([0.104, 0.329, 0.835, 0.214])
        d = figure.add_axes([0.104, 0.090, 0.835, 0.174], sharex=c)
        _by_disturbance(a, ordinary)
        _pressure(b, pressure)
        _timeline(c, events, fault, recovery)
        _start_steps(d, cases, fault, recovery)
        case_label = (
            f"Local repair, DRMS cap 0.50 m, "
            f"new alert {float(selected['new_theta_deg']):.0f}\N{DEGREE SIGN} "
            f"at {float(selected['new_range_m']):.0f} m, {selected['failure']} failure"
        )
        figure.text(0.105, 0.597, case_label, ha="left", va="center", fontsize=8)
        figure.legend(handles=_policy_handles(), loc="upper left", bbox_to_anchor=(0.065, 0.994),
                      bbox_transform=figure.transFigure, ncol=4, borderaxespad=0,
                      columnspacing=1.10, handletextpad=0.28, fontsize=8)
        figure.legend(
            handles=[Patch(facecolor=color, label=label) for label, color in PHASES.values()],
            loc="upper center", bbox_to_anchor=(0.58, 0.323), bbox_transform=figure.transFigure,
            ncol=4, borderaxespad=0, columnspacing=1.15, handletextpad=0.35, fontsize=8,
        )
    return {"fig05_rescheduling": figure}
