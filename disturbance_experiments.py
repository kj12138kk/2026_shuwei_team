"""Optional paired diagnostics and targeted search over the nominal disturbances."""
from functools import lru_cache
from time import perf_counter

import numpy as np

from disturbances import DISTURBANCE, WINDOW, Scenario, exact_scenario, run_scenario, score
from event_checks import separation_audit, verify_history
from experiments import plan_stations
from geometry import EPS, OBSERVE, STARTS, TARGETS, Target, path_length
from scheduler import Space


def _policy_metrics(result, nominal, scenario):
    n = len(nominal) + int(scenario.new is not None)
    eta, last = score(result.starts, n)
    delays = [max(0., result.starts[m.target] - m.start) for m in nominal
              if result.starts.get(m.target, np.inf) <= WINDOW + EPS]
    traveled = sum(path_length(e.path) * min(1., (WINDOW - e.begin) / (e.end - e.begin))
                   for e in result.events if e.phase == "travel" and e.begin < WINDOW)
    original_distance = sum(path_length(leg.path) * min(1., (WINDOW - leg.depart) / (leg.arrive - leg.depart))
                            for m in nominal for leg in m.legs if leg.depart < WINDOW and leg.arrive > leg.depart)
    return dict(eta=eta, t_last_s=last, served=int(round(eta * n)), total=n,
                latest_success_s=max((t for t in result.starts.values() if t <= WINDOW + EPS), default=None),
                original_missing=len(nominal) - len(delays), mean_original_delay_s=float(np.mean(delays)) if delays else None,
                distance_m=traveled, extra_distance_vs_nominal_m=traveled - original_distance,
                **separation_audit(result.events, WINDOW))


def _representatives(plan, design, metrics):
    by_key = {s.key: s for s in design}
    worst = sorted((r for r in metrics if r["policy"] == "replan" and r["kind"] != "nominal"),
                   key=lambda r: (r["eta"], -r["t_last_s"], r["scenario"]))
    choices = [next(r for r in worst if r["kind"] == kind)["scenario"] for kind in
               ("new+failure+outage", "new+failure", "failure+outage")]
    choices.extend(("N000_50", "V4_R10", "R10"))
    cases = [by_key[key] for key in dict.fromkeys(choices)]
    tie = next((m.start for m in plan if DISTURBANCE < m.start <= 30.), None)
    travel = next(((max(leg.depart, DISTURBANCE) + min(leg.arrive, 30.)) / 2. for m in plan for leg in m.legs
                   if min(leg.arrive, 30.) > max(leg.depart, DISTURBANCE) + EPS), None)
    new = by_key["N000_50"].new
    if tie is not None:
        cases.append(Scenario("recovery_at_start", "new+outage", new=new, recovery=tie))
    if travel is not None:
        cases.append(Scenario("recovery_in_transit", "new+outage", new=new, recovery=travel))
    return cases


def _worst_search(plan, metrics, get_space, verify):
    original = [r for r in metrics if r["policy"] == "replan" and r["kind"] == "new+failure+outage"]

    def parameter_key(theta, radius, failure, recovery):
        return round(theta % 360., 6) % 360., round(radius, 6), failure, round(recovery, 12)

    def worse(row):
        return row["eta"], -row["t_last_s"], row["scenario"]

    cache = {}
    for row in original:
        failure = int(row["failure"][1:]) - 1
        key = parameter_key(row["new_theta_deg"], row["new_range_m"], failure, row["recovery_s"])
        cache[key] = dict(scenario=row["scenario"], theta_deg=key[0], range_m=key[1], failure=key[2] + 1,
                          recovery_s=key[3], eta=row["eta"], t_last_s=row["t_last_s"],
                          selected_policy=row["selected_policy"], stage="coarse")

    def evaluate(key, stage):
        if key not in cache:
            angle, radius, v, t = key
            name = f"N{angle:.6f}_{radius:.6f}"
            new = Target(name, angle, radius, 50.)
            scenario = Scenario(f"{name}_V{v + 1}_R{t:.12g}", "new+failure+outage", new, v, t)
            space = get_space(new, 5.)
            policies = run_scenario(space, plan, scenario)
            outcome = policies["replan"]
            verify(outcome.events, outcome.starts, space.targets)
            verify_history(policies["retain"], outcome, scenario)
            eta, last = score(outcome.starts, len(plan) + 1)
            cache[key] = dict(scenario=scenario.key, theta_deg=angle, range_m=radius, failure=v + 1,
                              recovery_s=t, eta=eta, t_last_s=last, selected_policy=outcome.source, stage=stage)
        return cache[key]

    # Include nearer alerts in the coarse search; the original triple design
    # only samples the far radius.
    for angle in range(0, 360, 45):
        for radius in (30., 40.):
            for v in range(len(STARTS)):
                evaluate(parameter_key(angle, radius, v, 30.), "coarse_inner")

    # Probe both sides of state changes, where recovery may lock a different task.
    boundaries = {5.01, 30.}
    for mission in plan:
        for time in (mission.start, *(mission.start + OBSERVE + leg.slew for leg in mission.legs)):
            for shift in (-.01, 0., .01):
                if DISTURBANCE < time + shift <= 30.:
                    boundaries.add(round(time + shift, 12))
    seeds = []
    for key in sorted(cache, key=lambda k: worse(cache[k])):
        if not seeds or key[0] != seeds[0][0]:
            seeds.append(key)
            if len(seeds) == 2:
                break
    for seed in seeds:
        angle, radius, _, recovery = seed
        local = [parameter_key(angle, radius, v, t) for v in range(len(STARTS))
                 for t in sorted(boundaries | {recovery})]
        best = min(local, key=lambda key: worse(evaluate(key, "event_boundaries")))
        for da, dr, dt in ((15., 5., 4.), (7.5, 2.5, 2.), (3.75, 1.25, 1.)):
            angle, radius, v, time = best
            neighbors = [(angle - da, radius, time), (angle + da, radius, time),
                         (angle, max(30., radius - dr), time), (angle, min(50., radius + dr), time),
                         (angle, radius, max(5.001, time - dt)), (angle, radius, min(30., time + dt))]
            local = [best, *(parameter_key(a, r, v, t) for a, r, t in neighbors)]
            best = min(local, key=lambda key: worse(evaluate(key, "local")))
    return sorted(cache.values(), key=worse)


def diagnostic_results(router, plan, design, metrics, verify, *, state_budget):
    anchors = np.vstack((STARTS, plan_stations(plan)))
    pool_time = {}

    @lru_cache(maxsize=4)
    def get_space(new, grid):
        tick = perf_counter()
        space = Space(router, (*TARGETS, new) if new is not None else TARGETS, grid, anchors)
        pool_time[new, grid] = perf_counter() - tick
        return space

    ablation, efficiency, exact, exact_plans = [], [], [], []
    cases = _representatives(plan, design, metrics)
    for scenario in cases:
        space = get_space(scenario.new, 5.)
        diagnostics, counts = {}, {}
        policies = run_scenario(space, plan, scenario, diagnostics=diagnostics, search_stats=counts)
        ref = policies["retain"]

        def check(result):
            verify(result.events, result.starts, space.targets)
            verify_history(ref, result, scenario)
            return _policy_metrics(result, plan, scenario)

        for variant, (outcome, stats) in diagnostics.items():
            ablation.append(dict(scenario=scenario.key, variant=variant, selected_policy=outcome.source,
                                 kept_jobs=len(outcome.kept), continued_legs=len(outcome.continued),
                                 **check(outcome), solver_s=stats["seconds"],
                                 expanded_states=stats.get("expanded_states", 0),
                                 generated_states=stats.get("generated_states", 0)))

        beam_by_grid = {5.: policies["replan"]}
        if scenario in cases[:3]:
            settings = ((w, 18, 5.) for w in (4, 8, 12, 24, 48, 96))
            settings = [*settings, (12, 8, 5.), (12, 36, 5.), (12, 18, 7.), (12, 18, 3.)]
            for width, limit, grid in settings:
                if (width, limit, grid) == (12, 18, 5.):
                    other, outcome, stats = space, policies["replan"], counts
                else:
                    other = get_space(scenario.new, grid)
                    stats = {}
                    outcome = run_scenario(other, plan, scenario, width=width, limit=limit, search_stats=stats)["replan"]
                    if (width, limit) == (12, 18):
                        beam_by_grid[grid] = outcome
                efficiency.append(dict(scenario=scenario.key, width=width, shortlist=limit, grid_m=grid,
                                       candidate_points=len(other.points),
                                       **check(outcome), pool_build_s=pool_time[scenario.new, grid],
                                       setup_s=stats["setup_s"], solver_s=stats["seconds"],
                                       expanded_states=stats["expanded_states"], generated_states=stats["generated_states"]))

        if scenario == cases[0] or scenario.key in ("N000_50", "V4_R10"):
            labels = [("full_pool", (), False, 5.)]
            keep = policies["repair"].kept
            if keep:
                labels.extend((("fixed_then_append", keep, True, 5.), ("fixed_with_insertion", keep, False, 5.)))
            if scenario == cases[0]:
                labels.append(("full_pool_dense", (), False, 3.))
            for label, kept, fixed_first, grid in labels:
                pool = space if grid == 5. else get_space(scenario.new, grid)
                beam_eta, beam_last = score(beam_by_grid[grid].starts, len(plan) + int(scenario.new is not None))
                tick = perf_counter()
                outcome, stats = exact_scenario(pool, plan, scenario, state_budget=state_budget, keep=kept,
                                                retained_first=fixed_first, incumbent=beam_by_grid[grid])
                comparison = check(outcome) if outcome is not None else dict.fromkeys(_policy_metrics(ref, plan, scenario))
                exact.append(dict(scenario=scenario.key, variant=label, kept_jobs="|".join(kept), grid_m=grid,
                                  **comparison, beam_eta=beam_eta, beam_t_last_s=beam_last,
                                  elapsed_s=perf_counter() - tick, setup_s=stats["setup_s"], solver_s=stats["seconds"],
                                  visited_states=stats["visited_states"], expanded_states=stats["expanded_states"],
                                  dominance_prunes=stats["dominance_prunes"], bound_prunes=stats["bound_prunes"],
                                  state_budget=state_budget, complete=stats["complete"], feasible=stats["feasible"],
                                  confirmed_on_pool=stats["confirmed_on_pool"], infeasible_on_pool=stats["infeasible_on_pool"],
                                  termination=stats["termination"]))
                if outcome is not None:
                    exact_plans.append((scenario.key, label, outcome.events))

    worst = _worst_search(plan, metrics, get_space, verify)
    runtime = []
    for config in dict.fromkeys((r["width"], r["shortlist"], r["grid_m"]) for r in efficiency):
        rows = [r for r in efficiency if (r["width"], r["shortlist"], r["grid_m"]) == config]
        times = [r["setup_s"] + r["solver_s"] for r in rows]
        runtime.append(dict(width=config[0], shortlist=config[1], grid_m=config[2], cases=len(rows),
                            mean_s=float(np.mean(times)), median_s=float(np.median(times)),
                            p95_s=float(np.quantile(times, .95)), max_s=max(times),
                            mean_pool_build_s=float(np.mean([r["pool_build_s"] for r in rows])),
                            expanded_states=sum(r["expanded_states"] for r in rows),
                            generated_states=sum(r["generated_states"] for r in rows),
                            mean_eta=float(np.mean([r["eta"] for r in rows])),
                            mean_t_last_s=float(np.mean([r["t_last_s"] for r in rows]))))
    return ablation, efficiency, runtime, exact, exact_plans, worst
