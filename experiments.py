"""Paired scheduling experiments and per-visit pointing diagnostics."""
import numpy as np

from continuous_schedule import refine_joint
from errors import probability
from geometry import EPS, SPEED, STARTS, observable, path_length
from scheduler import State, Space, build_plan, refine, reuse_chain_lower_bound, single_reuse, solve

ACCURACY_LIMIT = 0.5
ANGLE_LIMIT = 30.
LOS_MARGINS = (0., .05, .10)
BEARING_SIGMA = np.deg2rad(1.)
LOCAL_LATTICES = ((2., .5), (1., .25), (.5, .125), (.25, .0625), (.12, .03), (.06, .015))


def polar_stations(targets, angle_step, range_step, max_rmse=ACCURACY_LIMIT):
    """Generate near-target candidates that can meet the DRMS range bound."""
    bearing = np.deg2rad(np.arange(0., 360., angle_step))
    unit = np.column_stack((np.cos(bearing), np.sin(bearing)))
    anchors = []
    for target in targets:
        lo = target.min_range
        hi = np.sqrt(max(0., (max_rmse / BEARING_SIGMA) ** 2 - lo ** 2))
        ranges = np.arange(lo, hi + EPS, range_step)
        points = (target.xy + ranges[:, None, None] * unit[None]).reshape(-1, 2)
        anchors.append(points[observable(points, target)])
    return np.vstack(anchors)


def plan_stations(plan):
    return np.asarray([leg.path[-1] for mission in plan for leg in mission.legs])


def _near_nominal(plan):
    angle = np.deg2rad(np.arange(0., 360., 22.5))
    unit = np.column_stack((np.cos(angle), np.sin(angle)))
    offsets = np.array([.025, .05, .10, .20, .40, .80, 1.60])[:, None, None] * unit[None]
    anchors = plan_stations(plan)
    return np.vstack([anchors, *(p + offsets.reshape(-1, 2) for p in anchors)])


def _plan_rank(plan):
    return round(plan[-1].start, 9), sum(m.start for m in plan)


def polish_pairs(router, targets, plan, constraints):
    """Re-select compatible station pairs on successively smaller local lattices."""
    history = []
    for half, step in LOCAL_LATTICES:
        stations = plan_stations(plan)
        axis = np.arange(-half, half + EPS, step)
        offsets = np.column_stack([a.ravel() for a in np.meshgrid(axis, axis)])
        nearby = (stations[:, None] + offsets[None]).reshape(-1, 2)
        space = Space(router, tuple(targets.values()), None, np.vstack((stations, nearby)), **constraints)
        count = len(space.points)
        initial = State(space.origins(STARTS), (0.,) * len(STARTS))
        solution = single_reuse(space, initial)
        if solution is None:
            break
        proposal, _, _ = build_plan(space, initial, solution)
        lower = reuse_chain_lower_bound(space, initial)["lower_bound_s"]
        if _plan_rank(proposal) < _plan_rank(plan):
            plan = proposal
        history.append(dict(half_width_m=half, step_m=step, candidate_points=count,
                            lower_bound_s=lower, local_plan_s=proposal[-1].start, best_s=plan[-1].start))
    return plan, history


def _experiment(router, targets, grid, anchors, constraints, incumbents=()):
    space = Space(router, tuple(targets.values()), grid, anchors, **constraints)
    count = len(space.points)
    initial = State(space.origins(STARTS), (0.,) * len(STARTS))
    beam = solve(space, initial, tuple(targets), width=26, limit=30, branch=4, deadline=np.inf)
    exact = single_reuse(space, initial)
    discrete = [build_plan(space, initial, state)[0]
                for state in (beam, exact) if state is not None and len(state.choices) == len(targets)]
    refined = [refine(router, targets, plan, STARTS, initial.releases, **constraints)
               for plan in [*incumbents, *discrete]]
    candidates = [plan for plan in refined if plan is not None and len(plan) == len(targets)]
    if not candidates:
        raise RuntimeError(f"No complete experiment schedule found under {constraints}.")
    optimized = min(candidates, key=_plan_rank)
    refined_s = optimized[-1].start
    joint = []
    if "max_rmse" in constraints:
        optimized, polishing = polish_pairs(router, targets, optimized, constraints)
    else:
        polishing = []
    if constraints == {"los_margin": 0.}:
        optimized, joint = refine_joint(router, targets, optimized)
    greedy = solve(space, initial, tuple(targets), nearest=True, deadline=np.inf)
    nearest, _, _ = build_plan(space, initial, greedy)
    if len(nearest) != len(targets):
        raise RuntimeError(f"Nearest baseline did not cover all alerts under {constraints}.")
    result = dict(grid_m=grid, points=count, constraints=constraints, optimized=optimized,
                  nearest=nearest, best_discrete_s=min((p[-1].start for p in discrete), default=None),
                  discrete_bound=reuse_chain_lower_bound(space, initial), refined_s=refined_s, polishing=polishing,
                  joint=joint)
    if constraints == {"los_margin": 0.}:
        earliest = solve(space, initial, tuple(targets), earliest=True, deadline=np.inf)
        result["earliest"], _, _ = build_plan(space, initial, earliest)
        if len(result["earliest"]) != len(targets):
            raise RuntimeError("Earliest-ready baseline did not cover all alerts.")
    return result


def schedule_experiments(router, targets, nominal):
    """Use a common candidate set for margin comparisons and a denser local polar pool for accuracy."""
    anchors = np.vstack((STARTS, _near_nominal(nominal)))
    margin = {}
    incumbent = []
    for value in reversed(LOS_MARGINS):
        result = _experiment(router, targets, 3., anchors, {"los_margin": value},
                             [nominal, *incumbent])
        incumbent = [result["optimized"]]
        margin[value] = result
    strict = dict(max_rmse=ACCURACY_LIMIT, min_angle=ANGLE_LIMIT)
    rings = polar_stations(tuple(targets.values()), 10., 2.)
    precision = _experiment(router, targets, 3., np.vstack((STARTS, rings)), strict)
    return {**{f"los_{value:.2f}": margin[value] for value in LOS_MARGINS},
            "drms_0.50": precision}


def _pointing_visit(vehicle, target, run_m, range_m, sigma_limit):
    sigma = .3 + .02 * run_m
    ratio = sigma / range_m
    return dict(vehicle=vehicle, target=target, cumulative_travel_m=run_m, range_m=range_m,
                position_sd_m=sigma, sigma_over_range=ratio,
                success_probability=probability(40. * ratio), passes_95=ratio <= sigma_limit / 40.)


def pointing_checks(plans, targets, t2_accuracy, sigma_limit):
    """Apply the same projected-normal model using the distance of each actual visit."""
    checks = {}
    for name, plan in plans.items():
        total_distance = np.zeros(len(STARTS))
        rows = []
        for mission in plan:
            points = np.asarray([leg.path[-1] for leg in mission.legs])
            target = targets[mission.target]
            ranges = np.linalg.norm(points - target.xy, axis=1)
            for leg, r in zip(mission.legs, ranges):
                total_distance[leg.vehicle] += path_length(leg.path)
                rows.append(_pointing_visit(f"V{leg.vehicle + 1}", mission.target,
                                           float(total_distance[leg.vehicle]), float(r), sigma_limit))
        checks[name] = rows
    checks["t2_accuracy_pair"] = [_pointing_visit(v, "T2", float(t) * SPEED, float(r), sigma_limit)
                                for v, t, r in zip(t2_accuracy["vehicles"], t2_accuracy["travel_times_s"],
                                                    t2_accuracy["ranges_m"])]
    return dict(ratio_limit=sigma_limit / 40., zero_travel_min_range_m=.3 / (sigma_limit / 40.),
                position_overtakes_zero_offset_m=(40. * np.deg2rad(.8) - .3) / .02,
                position_overtakes_combined_gimbal_m=(40. * np.deg2rad(1.) - .3) / .02,
                schedules=checks)
