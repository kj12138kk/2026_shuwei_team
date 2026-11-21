"""Solve all four tasks and export reproducible results."""
import argparse
import csv
from collections import defaultdict
from itertools import groupby
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from analysis_metrics import (continuous_lower_bound, plot_intersection_sweep, plot_probability, plot_scenarios,
                              plot_schedule, plot_t1, plot_variance_shares)
from continuous_schedule import adaptive_chain_bound
from disturbance_experiments import diagnostic_results
from disturbances import WINDOW, Event, locked_at_command, plan_events, pressure_scenarios, run_scenario, scenarios, score
from event_checks import separation_audit, verify_history
from errors import intersection_sweep, pointing_error_results, pointing_variance_curve
from experiments import plan_stations, pointing_checks, polar_stations, schedule_experiments
from geometry import ANGULAR_SPEED, BUILDINGS, EPS, OBSERVE, RADIUS, SPEED, STARTS, TARGETS, Router, _box_interval, los_clearance, observable, pair_geometry, path_length, pointing
from problem1 import solve_t1
from problem2_accuracy import solve_t2_accuracy
from scheduler import MIN_ANGLE, State, Space, build_plan, refine, reuse_chain_lower_bound, single_reuse, solve


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
        rows.append(dict(target=mission.target, start_s=mission.start, end_s=mission.start + OBSERVE,
                         **pair_geometry(*stations, targets[mission.target]),
                         legs=[dict(vehicle=f"V{leg.vehicle + 1}", station=_as_float_list(leg.path[-1]),
                                    depart_s=leg.depart, arrive_s=leg.arrive, pointing_ready_s=leg.ready,
                                    drive_m=path_length(leg.path), slew_s=leg.slew,
                                    los_clearance_m=float(los_clearance(leg.path[-1], targets[mission.target])),
                                    wait_s=max(0., mission.start - leg.ready),
                                     available_s=mission.start + OBSERVE + leg.slew)
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
    failures = {event.vehicle: event.begin for event in events if event.phase == "failed"}
    for event in events:
        if event.begin < -EPS or event.end < event.begin - EPS:
            raise ValueError("Invalid event interval.")
        change = max(abs(event.pan1 - event.pan0), abs(event.tilt1 - event.tilt0))
        if event.phase in ("slew", "reset", "abort_reset"):
            if change > ANGULAR_SPEED * (event.end - event.begin) + 1e-7:
                raise ValueError("Gimbal exceeded its angular rate.")
        elif change > 1e-7:
            raise ValueError("Gimbal moved outside a turning phase.")
        if event.phase == "observe":
            observations[event.target].append(event)
            expected_end = min(event.begin + OBSERVE, failures.get(event.vehicle, np.inf))
            if expected_end <= event.begin or abs(event.end - expected_end) > 1e-7:
                raise ValueError("Unexpected observation duration.")
        if event.phase == "travel":
            if max(abs(event.pan0), abs(event.tilt0)) > 1e-7:
                raise ValueError("Gimbal must be level and east-pointing while driving.")
            if abs(path_length(event.path) - SPEED * (event.end - event.begin)) > 2e-6:
                raise ValueError("Travel duration is inconsistent with path length.")
            for p, q in zip(event.path[:-1], event.path[1:]):
                if any(_independent_clearance(p, q, box) < RADIUS - 1e-6 for box in BUILDINGS):
                    raise ValueError("Travel path violates physical building clearance.")
        if np.any(event.path < RADIUS - 1e-6) or np.any(event.path > 100. - RADIUS + 1e-6):
            raise ValueError("Vehicle outside the field.")
        if event.end > event.begin + EPS or event.phase == "failed":
            by_vehicle[event.vehicle].append(event)
    for vehicle, timeline in by_vehicle.items():
        timeline.sort(key=lambda event: event.begin)
        if np.linalg.norm(timeline[0].path[0] - STARTS[vehicle]) > 1e-6:
            raise ValueError("Vehicle timeline does not begin at its initial position.")
        for a, b in zip(timeline, timeline[1:]):
            if a.end > b.begin + 1e-7 or a.phase == "failed":
                raise ValueError(f"Inconsistent event times: {a} and {b}")
            if np.linalg.norm(a.path[-1] - b.path[0]) > 1e-6:
                raise ValueError(f"Movement discontinuity: {a} and {b}")
            if max(abs(a.pan1 - b.pan0), abs(a.tilt1 - b.tilt0)) > 1e-6:
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
                                 exact_one_reuse_s=recovered[1][-1].start, refined_s=improved[-1].start, nearest_s=nearest[-1].start,
                                 discrete_bound=reuse_chain_lower_bound(space, initial)))
    optimized, nearest = min(candidates, key=lambda x: (round(x[0][-1].start, 8), sum(m.start for m in x[0])))
    assert len(optimized) == len(nearest) == 3
    return optimized, nearest, sensitivity


def _q4_case(space, plan, scenario, constraints):
    policies = run_scenario(space, plan, scenario)
    new = scenario.new
    nominal_times = {m.target: m.start for m in plan}
    n = len(nominal_times) + int(new is not None)
    cutoff = scenario.cutoff if scenario.kind != "nominal" else 0.
    common_history = locked_at_command(policies["retain"].starts, scenario)
    metrics, timelines = [], []
    for policy, result in policies.items():
        starts = result.starts
        verify(result.events, starts, space.targets, **constraints)
        verify_history(policies["retain"], result, scenario)
        eta, t_last = score(starts, n)
        observed = {alert: time for alert, time in starts.items() if time <= WINDOW + EPS}
        delays = {alert: observed[alert] - t for alert, t in nominal_times.items() if alert in observed}
        metrics.append(dict(scenario=scenario.key, kind=scenario.kind, policy=policy, selected_policy=result.source,
                            n=n, k=len(observed), eta=eta, all_started=int(len(observed) == n), t_last_s=t_last,
                            latest_success_s=max(observed.values(), default=0.),
                            command_s=cutoff if scenario.kind != "nominal" else "",
                            failure=f"V{scenario.failure + 1}" if scenario.failure is not None else "",
                            recovery_s=scenario.recovery if scenario.recovery is not None else "",
                            new_theta_deg=new.theta if new else "", new_range_m=new.distance if new else "",
                            t1_s=starts.get("T1", ""), t2_s=starts.get("T2", ""), t3_s=starts.get("T3", ""),
                            new_alert_s=starts.get(new.name, "") if new else "",
                            locked_at_command="|".join(sorted(common_history)), kept_jobs="|".join(result.kept),
                            continued_legs="|".join(f"V{v + 1}:{name}" for v, name in result.continued),
                            original_missing=len(plan) - len(delays), t1_delay_s=delays.get("T1", ""),
                            t2_delay_s=delays.get("T2", ""), t3_delay_s=delays.get("T3", ""),
                            mean_positive_original_delay_s=float(np.mean([max(0., d) for d in delays.values()])) if delays else "",
                            **separation_audit(result.events, WINDOW)))
        timelines.extend(_timeline_rows(result.events, scenario.key, policy, WINDOW))
    return metrics, timelines


def _q4_batch(router, plan, design, constraints):
    anchors = np.vstack((STARTS, plan_stations(plan)))
    metrics, timelines = [], []
    done = 0
    ordered = sorted(design, key=lambda s: s.new.name if s.new else "")
    for new, cases in groupby(ordered, key=lambda s: s.new):
        active = (*TARGETS, new) if new else TARGETS
        pool = anchors
        if "max_rmse" in constraints:
            pool = np.vstack((anchors, polar_stations(active, 15., 2.5, constraints["max_rmse"])))
        space = Space(router, active, 5., pool, **constraints)
        for scenario in cases:
            rows, events = _q4_case(space, plan, scenario, constraints)
            metrics.extend(rows)
            timelines.extend(events)
            done += 1
            if done % 40 == 0:
                print(f"Scenario progress {done}/{len(design)}", flush=True)
    order = {scenario.key: i for i, scenario in enumerate(design)}
    return sorted(metrics, key=lambda r: order[r["scenario"]]), sorted(timelines, key=lambda r: order[r["scenario"]])


def _q4_aggregate(records, keys):
    groups = defaultdict(list)
    for row in records:
        groups[tuple(row[k] for k in keys)].append(row)
    aggregate = []
    for group, rows in groups.items():
        worst = min(rows, key=lambda row: (row["eta"], -row["t_last_s"], row["scenario"]))
        delays = [max(0., row[f"t{i}_delay_s"]) for row in rows for i in range(1, 4)
                  if row[f"t{i}_delay_s"] != ""]
        aggregate.append(dict(zip(keys, group), scenarios=len(rows), all_started=sum(r["all_started"] for r in rows),
                              mean_eta=float(np.mean([r["eta"] for r in rows])),
                              mean_t_last_s=float(np.mean([r["t_last_s"] for r in rows])),
                              mean_positive_original_delay_s=float(np.mean(delays)) if delays else None,
                              original_missing=sum(r["original_missing"] for r in rows),
                              conflicts_found=sum(r["conflicting_pairs"] > 0 for r in rows),
                              min_separation_m=min(r["min_separation_m"] for r in rows),
                              worst_scenario=worst["scenario"], worst_eta=worst["eta"], worst_t_last_s=worst["t_last_s"]))
    return aggregate


def _q4_records(router, plan, design, out):
    metrics, timelines = _q4_batch(router, plan, design, {})
    _write_csv(out / "q4_scenarios.csv", metrics)
    _write_csv(out / "q4_vehicle_timelines.csv", timelines)
    plot_scenarios(out, metrics)
    aggregate = _q4_aggregate(metrics, ("kind", "policy"))
    _write_csv(out / "q4_aggregate.csv", aggregate)
    grouped = [dict(group_by=key, **row) for key in ("failure", "recovery_s") for row in
               _q4_aggregate([dict(r, value=r[key]) for r in metrics if r[key] != ""], ("kind", "policy", "value"))]
    _write_csv(out / "q4_grouped.csv", grouped)
    return dict(scenario_count=len(design), timeline_rows=len(timelines), aggregate=aggregate), metrics, timelines


def _export_q2_experiments(design, targets, out):
    rows, timelines, metadata = [], [], {}
    for name, trial in design.items():
        constraints = trial["constraints"]
        details = {}
        for policy in (p for p in ("optimized", "nearest", "earliest") if p in trial):
            plan = trial[policy]
            events = plan_events(plan)
            verify(events, {m.target: m.start for m in plan}, targets, **constraints)
            missions = _leg_summary(plan, targets)
            details[policy] = dict(max_start_s=plan[-1].start, missions=missions,
                                   **separation_audit(events, max(e.end for e in events)))
            for m in missions:
                rows.append(dict(experiment=name, policy=policy, target=m["target"], start_s=m["start_s"],
                                 max_start_s=plan[-1].start, vehicle_pair="+".join(l["vehicle"] for l in m["legs"]),
                                 baseline_m=m["baseline_m"], angle_deg=m["angle_deg"], drms_m=m["linear_rmse_m"],
                                 min_los_clearance_m=min(l["los_clearance_m"] for l in m["legs"]),
                                 drms_limit_m=constraints.get("max_rmse", ""),
                                 min_angle_deg=constraints.get("min_angle", MIN_ANGLE),
                                 los_margin_m=constraints.get("los_margin", 0.)))
            timelines.extend(_timeline_rows(events, name, policy, max(e.end for e in events)))
        metadata[name] = dict(grid_m=trial["grid_m"], candidate_points=trial["points"], constraints=constraints,
                              best_discrete_s=trial["best_discrete_s"], discrete_bound=trial["discrete_bound"],
                              coordinate_refined_s=trial["refined_s"], pair_polishing=trial["polishing"],
                              joint_refinement=trial["joint"], **details)
    _write_csv(out / "q2_schedule_experiments.csv", rows)
    _write_csv(out / "q2_experiment_timelines.csv", timelines)
    return metadata


def _q4_pressure(router, precision, nominal_metrics, nominal_timelines, out):
    _, design = pressure_scenarios()
    keep = {case.key for case in design}
    checked, paths = _q4_batch(router, precision["optimized"], design, precision["constraints"])
    metrics = [dict(model="nominal", **r) for r in nominal_metrics if r["scenario"] in keep]
    metrics.extend(dict(model="drms_0.50", **r) for r in checked)
    timelines = [dict(model="nominal", **r) for r in nominal_timelines if r["scenario"] in keep]
    timelines.extend(dict(model="drms_0.50", **r) for r in paths)
    _write_csv(out / "q4_pressure_scenarios.csv", metrics)
    _write_csv(out / "q4_pressure_timelines.csv", timelines)
    aggregate = _q4_aggregate(metrics, ("model", "policy"))
    _write_csv(out / "q4_pressure_aggregate.csv", aggregate)
    return dict(scenarios_per_model=len(design), timeline_rows=len(timelines), aggregate=aggregate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--q4-diagnostics", action="store_true",
                        help="also run paired ablations, finite-pool search and targeted worst-case analysis")
    parser.add_argument("--q4-state-budget", type=int, default=20_000,
                        help="maximum states per finite-pool search in --q4-diagnostics")
    parser.add_argument("--q2-bound-budget", type=int, default=60_000,
                        help="maximum subdivisions for the adaptive continuous bound")
    parser.add_argument("--q2-relative-gap", type=float, default=.005,
                        help="target relative gap for the continuous bound (default: 0.005)")
    args = parser.parse_args()
    if args.q4_state_budget < 1 or args.q2_bound_budget < 1:
        parser.error("search budgets must be positive")
    if not 0. < args.q2_relative_gap < 1.:
        parser.error("the target relative gap must be between 0 and 1")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    tick = perf_counter()

    router = Router()
    targets = {target.name: target for target in TARGETS}
    q1 = solve_t1(router)
    plot_t1(out, q1)

    q2_accuracy = solve_t2_accuracy(router)
    intersection = intersection_sweep()
    _write_csv(out / "q2_intersection_errors.csv", intersection)
    plot_intersection_sweep(out, intersection)
    q2_accuracy["sweep"] = dict(cases=len(intersection), samples_per_case=intersection[0]["samples"],
                                rejection_cutoff_deg=intersection[0]["reject_within_deg"],
                                table="q2_intersection_errors.csv")

    initial_optimized, _, sensitivity = _solve_q2_schedule(router, targets)
    trials = schedule_experiments(router, targets, initial_optimized)
    nominal = trials["los_0.00"]
    optimized, nearest, earliest = (nominal[policy] for policy in ("optimized", "nearest", "earliest"))
    timelines = []
    separation = {}
    for policy, plan in (("optimized", optimized), ("nearest", nearest), ("earliest", earliest)):
        events = plan_events(plan)
        verify(events, {m.target: m.start for m in plan}, targets)
        horizon = max(e.end for e in events)
        separation[policy] = separation_audit(events, horizon)
        timelines += _timeline_rows(events, "nominal", policy, horizon)
    _write_csv(out / "q2_vehicle_timelines.csv", timelines)
    plot_schedule(out, optimized, nearest, earliest)
    upper = optimized[-1].start
    bounds = [continuous_lower_bound(step, upper) for step in (1., .5, .25, .125)]
    adaptive = adaptive_chain_bound(targets, upper, bounds[-1], relative_gap=args.q2_relative_gap,
                                    node_budget=args.q2_bound_budget)
    lower = max(adaptive["lower_bound_s"], *(x["lower_bound_s"] for x in bounds))
    if lower > upper + EPS:
        raise ValueError("The continuous lower bound exceeds a verified feasible schedule.")
    q2_schedule = dict(optimized=_leg_summary(optimized, targets), nearest=_leg_summary(nearest, targets),
                       earliest=_leg_summary(earliest, targets), max_start_s=upper,
                       nearest_max_start_s=nearest[-1].start, earliest_max_start_s=earliest[-1].start,
                       improvement_s=nearest[-1].start - upper, improvement_ratio=1. - upper / nearest[-1].start,
                       continuous_lower_bound_s=lower, certified_gap_s=upper - lower,
                       refinements=sensitivity, cell_bounds=bounds, adaptive_bound=adaptive,
                       vehicle_separation=separation,
                       same_pool_comparison_s=dict(nearest=nearest[-1].start, earliest=earliest[-1].start,
                                                   optimized_discrete=nominal["best_discrete_s"]))
    q2_schedule["experiments"] = _export_q2_experiments(trials, targets, out)
    precision = trials["drms_0.50"]
    q3 = pointing_error_results()
    q3["station_checks"] = pointing_checks({"nominal": optimized, "drms_0.50": precision["optimized"]},
                                           targets, q2_accuracy, q3["sigma_limit_m"])
    plot_probability(out, q3)
    variance = pointing_variance_curve()
    _write_csv(out / "q3_variance_shares.csv", variance)
    plot_variance_shares(out, variance, q3["station_checks"])
    print(f"Q1-Q3 validated in {perf_counter()-tick:.1f} s", flush=True)

    _, design = scenarios()
    q4, metrics, timeline = _q4_records(router, optimized, design, out)
    q4["pressure"] = _q4_pressure(router, precision, metrics, timeline, out)
    if args.q4_diagnostics:
        ablation, efficiency, runtime, exact, plans, worst = diagnostic_results(
            router, optimized, design, metrics, verify, state_budget=args.q4_state_budget)
        for key, rows in (("ablation", ablation), ("search_efficiency", efficiency), ("runtime", runtime),
                          ("exact_search", exact), ("worst_search", worst)):
            _write_csv(out / f"q4_{key}.csv", rows)
        exact_timeline = [row for case, variant, events in plans for row in _timeline_rows(events, case, variant, WINDOW)]
        if exact_timeline:
            _write_csv(out / "q4_exact_timelines.csv", exact_timeline)
        q4["diagnostics"] = dict(ablation_rows=len(ablation), exact_cases=len(exact),
                                 exact_certificates=sum(row["confirmed_on_pool"] for row in exact),
                                 worst_cases_evaluated=len(worst), worst_found=worst[0])
    result = dict(q1=q1, q2_accuracy=q2_accuracy, q2_schedule=q2_schedule, q3=q3, q4=q4)
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(list(out.iterdir()))} results in {perf_counter()-tick:.1f} s to {out}", flush=True)


if __name__ == "__main__":
    main()
