"""Solve all four tasks and export reproducible results."""
import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from analysis_metrics import continuous_lower_bound, plot_probability, plot_scenarios, plot_schedule, plot_t1
from disturbances import WINDOW, Event, plan_events, pressure_scenarios, run_scenario, scenarios, score
from errors import pointing_error_results
from experiments import plan_stations, pointing_checks, polar_stations, schedule_experiments
from geometry import BUILDINGS, EPS, RADIUS, SPEED, STARTS, TARGETS, Router, _box_interval, los_clearance, observable, pair_geometry, path_length, pointing
from problem1 import solve_t1
from problem2_accuracy import solve_t2_accuracy
from scheduler import MIN_ANGLE, State, Space, build_plan, refine, single_reuse, solve


def _as_float_list(points):
    return np.asarray(points, dtype=float).tolist()


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _leg_summary(plan, targets):
    rows = []
    for mission in plan:
        stations = [leg.path[-1] for leg in mission.legs]
        rows.append(dict(target=mission.target, start_s=mission.start, end_s=mission.start + 5.,
                         **pair_geometry(*stations, targets[mission.target]),
                         legs=[dict(vehicle=f"V{leg.vehicle + 1}", station=_as_float_list(leg.path[-1]),
                                    depart_s=leg.depart, arrive_s=leg.arrive, pointing_ready_s=leg.ready,
                                    drive_m=path_length(leg.path), slew_s=leg.slew,
                                    los_clearance_m=float(los_clearance(leg.path[-1], targets[mission.target])),
                                    wait_s=max(0., mission.start - leg.ready),
                                    available_s=mission.start + 5. + leg.slew)
                               for leg in mission.legs]))
    return rows


def _independent_clearance(p, q, box):
    x0, x1, y0, y1, _ = box
    lo, hi = np.array([x0, y0]), np.array([x1, y1])
    if _box_interval(p, q, lo, hi):
        return 0.
    points = np.vstack([p, q])
    offset = np.maximum(np.maximum(lo - points, points - hi), 0.)
    nearest = np.linalg.norm(offset, axis=1).min()
    edge = q - p
    squared = edge @ edge
    if squared > EPS:
        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        t = np.clip((corners - p) @ edge / squared, 0., 1.)
        nearest = min(nearest, np.linalg.norm(corners - p - t[:, None] * edge, axis=1).min())
    return nearest


def verify(events, starts, targets, *, min_angle=MIN_ANGLE, max_rmse=None, los_margin=0.):
    by_vehicle = defaultdict(list)
    observations = defaultdict(list)
    for event in events:
        if event.phase == "observe":
            observations[event.target].append(event)
        if event.phase == "travel":
            if abs(path_length(event.path) - SPEED * (event.end - event.begin)) > 2e-6:
                raise ValueError("Travel duration is inconsistent with path length.")
            for p, q in zip(event.path[:-1], event.path[1:]):
                if any(_independent_clearance(p, q, box) < RADIUS - 1e-6 for box in BUILDINGS):
                    raise ValueError("Travel path violates physical building clearance.")
        if np.any(event.path < RADIUS - 1e-6) or np.any(event.path > 100. - RADIUS + 1e-6):
            raise ValueError("Vehicle outside the field.")
        if event.end > event.begin + EPS or event.phase == "failed":
            by_vehicle[event.vehicle].append(event)
    for timeline in by_vehicle.values():
        timeline.sort(key=lambda event: event.begin)
        for a, b in zip(timeline, timeline[1:]):
            if a.end > b.begin + 1e-7 or a.phase == "failed":
                raise ValueError(f"Inconsistent event times: {a} and {b}")
            if np.linalg.norm(a.path[-1] - b.path[0]) > 1e-6:
                raise ValueError(f"Movement discontinuity: {a} and {b}")
            if max(abs(a.pan1 - b.pan0), abs(a.tilt1 - b.tilt0)) > 1e-6 and b.phase != "failed":
                raise ValueError(f"Pointing discontinuity: {a} and {b}")
    if set(observations) != set(starts):
        raise ValueError("Observation event inventory mismatch.")
    for name, begin in starts.items():
        pair = observations[name]
        if len(pair) != 2 or pair[0].vehicle == pair[1].vehicle or any(abs(e.begin - begin) > 1e-7 for e in pair):
            raise ValueError("Observation is not synchronized.")
        locations = [event.path[0] for event in pair]
        target = targets[name]
        if not np.all(observable(locations, target)):
            raise ValueError("Observation location is infeasible.")
        geom = pair_geometry(*locations, target)
        if geom["baseline_m"] < 3. - 1e-6 or min(geom["angle_deg"], 180. - geom["angle_deg"]) < min_angle - 1e-6:
            raise ValueError("Scenario has an invalid pair.")
        if max_rmse is not None and geom["linear_rmse_m"] > max_rmse + 1e-8:
            raise ValueError("Scenario exceeds the DRMS limit.")
        if los_margin and np.any(los_clearance(locations, target) < los_margin - 1e-8):
            raise ValueError("Scenario violates the requested line-of-sight margin.")
        for event in pair:
            pan, tilt, _ = pointing(event.path[0], target)
            if max(abs(pan - event.pan0), abs(tilt - event.tilt0)) > 1e-7:
                raise ValueError("Observation started before the gimbal pointed at the target.")


def _timeline_rows(events, scenario, policy, horizon):
    rows = []
    for v, origin in enumerate(STARTS):
        phase_order = {"depart": 0, "at_station": 0, "arrive": 0, "abort": 0, "failed": 2}
        items = sorted((e for e in events if e.vehicle == v), key=lambda e: (e.begin, phase_order.get(e.phase, 1)))
        cursor, location, angles = 0., origin, (0., 0.)
        timeline = []
        for event in items:
            if event.begin > cursor + EPS:
                timeline.append(Event(v, "", "idle", cursor, event.begin, np.array([location]), *angles, *angles))
            timeline.append(event)
            cursor, location, angles = event.end, event.path[-1], (event.pan1, event.tilt1)
        if cursor < horizon and (not items or items[-1].phase != "failed"):
            timeline.append(Event(v, "", "idle", cursor, horizon, np.array([location]), *angles, *angles))
        for event in timeline:
            rows.append(dict(scenario=scenario, policy=policy, vehicle=f"V{v + 1}", target=event.target,
                             phase=event.phase, start_s=round(event.begin, 8), end_s=round(event.end, 8),
                             x_start_m=round(float(event.path[0, 0]), 8), y_start_m=round(float(event.path[0, 1]), 8),
                             x_end_m=round(float(event.path[-1, 0]), 8), y_end_m=round(float(event.path[-1, 1]), 8),
                             pan_start_deg=round(event.pan0, 8), elevation_start_deg=round(event.tilt0, 8),
                             pan_end_deg=round(event.pan1, 8), elevation_end_deg=round(event.tilt1, 8),
                             path_xy=json.dumps(np.round(event.path, 8).tolist(), separators=(",", ":"))))
    return rows


def _solve_q2_schedule(router, targets):
    sensitivity, candidates = [], []
    for grid in (5., 3., 2.):
        space = Space(router, TARGETS, grid, STARTS)
        initial = State(space.origins(STARTS), (0.,) * 5)
        beam = solve(space, initial, tuple(targets), width=24, limit=14, branch=3)
        exact_reuse = single_reuse(space, initial)
        recovered = [build_plan(space, initial, state)[0] for state in (beam, exact_reuse)]
        improved = min((refine(router, targets, result, STARTS, (0.,) * 5) for result in recovered),
                       key=lambda result: (round(result[-1].start, 9), sum(m.start for m in result)))
        baseline = solve(space, initial, tuple(targets), nearest=True)
        nearest, _, _ = build_plan(space, initial, baseline)
        candidates.append((improved, nearest))
        sensitivity.append(dict(grid_m=grid, n_stations=len(space.points), beam_s=recovered[0][-1].start,
                                exact_one_reuse_s=recovered[1][-1].start, refined_s=improved[-1].start, nearest_s=nearest[-1].start))
    optimized, nearest = min(candidates, key=lambda x: (round(x[0][-1].start, 8), sum(m.start for m in x[0])))
    assert len(optimized) == len(nearest) == 3
    return optimized, nearest, sensitivity


def _q4_records(space, plan, design, out):
    metrics, timelines = [], []
    for i, scenario in enumerate(design, 1):
        policies = run_scenario(space, plan, scenario)
        n = 3 + int(scenario.new is not None)
        for policy, (starts, events) in policies.items():
            verify(events, starts, space.targets)
            eta, t_last = score(starts, n)
            latest_observed = max((time for time in starts.values() if time <= WINDOW + EPS), default=0.)
            selected_policy = policy
            if policy == "replan":
                selected_policy = "nominal" if i == 1 else ("replan_raw" if policies[policy] is policies["replan_raw"] else "retain_plus")
            metrics.append(dict(scenario=scenario.key, kind=scenario.kind, policy=policy, selected_policy=selected_policy,
                                n=n, k=round(n * eta), eta=eta,
                                t_last_s=t_last, latest_success_s=latest_observed, command_s=scenario.cutoff if i > 1 else "",
                                failure=f"V{scenario.failure + 1}" if scenario.failure is not None else "",
                                recovery_s=scenario.recovery if scenario.recovery is not None else "",
                                new_theta_deg=scenario.new.theta if scenario.new else "",
                                new_range_m=scenario.new.distance if scenario.new else "",
                                t1_s=starts.get("T1", ""), t2_s=starts.get("T2", ""), t3_s=starts.get("T3", ""),
                                new_alert_s=starts.get(scenario.new.name, "") if scenario.new else ""))
            timelines.extend(_timeline_rows(events, scenario.key, policy, WINDOW))
        if i % 15 == 0:
            print(f"Scenario progress {i}/{len(design)}", flush=True)
    _write_csv(out / "q4_scenarios.csv", metrics)
    _write_csv(out / "q4_vehicle_timelines.csv", timelines)
    plot_scenarios(out, metrics)
    by_kind = defaultdict(list)
    for record in metrics:
        by_kind[(record["kind"], record["policy"])].append(record)
    aggregate = [{"kind": kind, "policy": policy, "scenarios": len(rows),
                  "mean_eta": float(np.mean([r["eta"] for r in rows])),
                  "mean_t_last_s": float(np.mean([r["t_last_s"] for r in rows]))}
                 for (kind, policy), rows in by_kind.items()]
    _write_csv(out / "q4_aggregate.csv", aggregate)
    return dict(scenario_count=len(design), timeline_rows=len(timelines), aggregate=aggregate)


def _export_q2_experiments(design, targets, out):
    rows, timelines, metadata = [], [], {}
    for name, trial in design.items():
        constraints = trial["constraints"]
        details = {}
        for policy in ("optimized", "nearest"):
            plan = trial[policy]
            events = plan_events(plan)
            verify(events, {m.target: m.start for m in plan}, targets, **constraints)
            missions = _leg_summary(plan, targets)
            details[policy] = dict(max_start_s=plan[-1].start, missions=missions)
            for m in missions:
                rows.append(dict(experiment=name, policy=policy, target=m["target"], start_s=m["start_s"],
                                 max_start_s=plan[-1].start, vehicle_pair="+".join(l["vehicle"] for l in m["legs"]),
                                 baseline_m=m["baseline_m"], angle_deg=m["angle_deg"], drms_m=m["linear_rmse_m"],
                                 min_los_clearance_m=min(l["los_clearance_m"] for l in m["legs"]),
                                 drms_limit_m=constraints.get("max_rmse", ""),
                                 min_angle_deg=constraints.get("min_angle", MIN_ANGLE),
                                 los_margin_m=constraints.get("los_margin", 0.)))
            timelines.extend(_timeline_rows(events, name, policy, max(e.end for e in events)))
        metadata[name] = dict(grid_m=trial["grid_m"], candidate_points=trial["points"],
                              constraints=constraints, best_discrete_s=trial["best_discrete_s"], **details)
    _write_csv(out / "q2_schedule_experiments.csv", rows)
    _write_csv(out / "q2_experiment_timelines.csv", timelines)
    return metadata


def _q4_pressure(router, baseline_models, out):
    new_targets, design = pressure_scenarios()
    metrics, timelines = [], []
    for name, (plan, constraints) in baseline_models.items():
        nominal_times = {m.target: m.start for m in plan}
        anchors = np.vstack((STARTS, plan_stations(plan)))
        for new in new_targets:
            pool = anchors
            if constraints.get("max_rmse") is not None:
                pool = np.vstack((anchors, polar_stations((*TARGETS, new), 15., 2.5)))
            space = Space(router, (*TARGETS, new), 5., pool, **constraints)
            for scenario in (case for case in design if case.new == new):
                policies = run_scenario(space, plan, scenario, width=12, repair=True)
                strictly_before = None
                for policy, (starts, events) in policies.items():
                    verify(events, starts, space.targets, **constraints)
                    locked = {alert: t for alert, t in starts.items() if t < scenario.cutoff - EPS}
                    if strictly_before is None:
                        strictly_before = locked
                    elif locked != strictly_before:
                        raise ValueError("Pressure-test policies used different pre-command histories.")
                    eta, last = score(starts, len(plan) + 1)
                    delays = {alert: starts[alert] - t for alert, t in nominal_times.items()
                              if alert in starts and starts[alert] <= WINDOW + EPS}
                    delay_avg = float(np.mean([max(0., d) for d in delays.values()])) if delays else None
                    chosen_policy = next((k for k, v in policies.items() if k != "replan" and policies[policy] is v), policy)
                    metrics.append(dict(model=name, scenario=scenario.key, policy=policy, selected_policy=chosen_policy,
                                        n=len(plan)+1, k=round((len(plan)+1)*eta), eta=eta, t_last_s=last,
                                        all_started=int(eta == 1.), command_s=scenario.cutoff,
                                        failure=f"V{scenario.failure + 1}", new_theta_deg=new.theta, new_range_m=new.distance,
                                        t1_s=starts.get("T1", ""), t2_s=starts.get("T2", ""), t3_s=starts.get("T3", ""),
                                        new_alert_s=starts.get(new.name, ""), locked_at_command=",".join(sorted(locked)),
                                        original_missing=len(plan)-len(delays),
                                        t1_delay_s=delays.get("T1", ""), t2_delay_s=delays.get("T2", ""),
                                        t3_delay_s=delays.get("T3", ""), mean_positive_original_delay_s=delay_avg))
                    timelines.extend(dict(model=name, **row) for row in
                                     _timeline_rows(events, scenario.key, policy, WINDOW))
        print(f"Pressure model {name}: {len(design)} paired scenarios", flush=True)
    _write_csv(out / "q4_pressure_scenarios.csv", metrics)
    _write_csv(out / "q4_pressure_timelines.csv", timelines)
    groups = defaultdict(list)
    for row in metrics:
        groups[row["model"], row["policy"]].append(row)
    aggregate = []
    for (name, policy), group in groups.items():
        worst = min(group, key=lambda row: (row["eta"], -row["t_last_s"], row["scenario"]))
        delays = [max(0., row[f"t{i}_delay_s"]) for row in group for i in range(1, 4)
                  if row[f"t{i}_delay_s"] != ""]
        aggregate.append(dict(model=name, policy=policy, cases=len(group),
                              all_started=sum(r["all_started"] for r in group),
                              mean_eta=float(np.mean([r["eta"] for r in group])),
                              mean_t_last_s=float(np.mean([r["t_last_s"] for r in group])),
                              mean_positive_original_delay_s=float(np.mean(delays)) if delays else None,
                              original_missing=sum(r["original_missing"] for r in group),
                              worst_scenario=worst["scenario"], worst_eta=worst["eta"],
                              worst_t_last_s=worst["t_last_s"]))
    _write_csv(out / "q4_pressure_aggregate.csv", aggregate)
    return dict(scenarios_per_model=len(design), timeline_rows=len(timelines), aggregate=aggregate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    tick = perf_counter()

    router = Router()
    targets = {target.name: target for target in TARGETS}
    q1 = solve_t1(router)
    plot_t1(out, q1)

    q2_accuracy = solve_t2_accuracy(router)

    optimized, nearest, sensitivity = _solve_q2_schedule(router, targets)
    timelines = []
    for policy, plan in (("optimized", optimized), ("nearest", nearest)):
        events = plan_events(plan)
        verify(events, {m.target: m.start for m in plan}, targets)
        timelines += _timeline_rows(events, "nominal", policy, max(e.end for e in events))
    _write_csv(out / "q2_vehicle_timelines.csv", timelines)
    plot_schedule(out, optimized, nearest)
    upper = optimized[-1].start
    bounds = [continuous_lower_bound(step, upper) for step in (1., .5, .25, .125)]
    lower = max(x["lower_bound_s"] for x in bounds)
    assert lower <= upper + EPS
    q2_schedule = dict(optimized=_leg_summary(optimized, targets), nearest=_leg_summary(nearest, targets),
                       max_start_s=upper, nearest_max_start_s=nearest[-1].start,
                       improvement_s=nearest[-1].start - upper, improvement_ratio=1. - upper / nearest[-1].start,
                       continuous_lower_bound_s=lower, certified_gap_s=upper - lower,
                       refinements=sensitivity, cell_bounds=bounds)
    trials = schedule_experiments(router, targets, optimized)
    q2_schedule["experiments"] = _export_q2_experiments(trials, targets, out)
    precision = trials["drms_0.50"]
    q3 = pointing_error_results()
    q3["station_checks"] = pointing_checks({"nominal": optimized, "drms_0.50": precision["optimized"]},
                                           targets, q2_accuracy, q3["sigma_limit_m"])
    plot_probability(out, q3)
    print(f"Q1-Q3 validated in {perf_counter()-tick:.1f} s", flush=True)

    new_targets, design = scenarios()
    anchors = np.vstack([STARTS, *(leg.path[-1] for m in optimized for leg in m.legs)])
    space = Space(router, (*TARGETS, *new_targets), 5., anchors)
    q4 = _q4_records(space, optimized, design, out)
    q4["pressure"] = _q4_pressure(router, {"nominal": (optimized, {}),
                                            "drms_0.50": (precision["optimized"], precision["constraints"])}, out)
    result = dict(q1=q1, q2_accuracy=q2_accuracy, q2_schedule=q2_schedule, q3=q3, q4=q4)
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(list(out.iterdir()))} results in {perf_counter()-tick:.1f} s to {out}", flush=True)


if __name__ == "__main__":
    main()
