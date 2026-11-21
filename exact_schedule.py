"""Budgeted branch and bound over the complete finite station pool."""
from collections import defaultdict
from itertools import combinations

import numpy as np

from geometry import EPS
from scheduler import _advance, _options


def seed_from_layout(space, initial, layout, continuations, deadline):
    """Rebuild a feasible search state from fixed visits, without trusting their timestamps."""
    state = initial
    for name, pair, stations in layout:
        move = next(_options(space, state, name, pair, None, 1, initial.latest, deadline, continuations,
                             fixed_stations=stations, greedy="earliest"), None)
        if move is None:
            return None
        _, choice, slews = move
        state = _advance(state, choice, slews)
    return state


def solve_exact(space, initial, targets, *, continuations=None, fixed=None, retained_first=False,
                incumbent=None, deadline=60., state_budget=20_000):
    """Maximize starts within the window, then minimize their latest start.

    Fixed missions must be served at their specified vehicle pair and stations.
    Their insertion order in ``fixed`` is preserved on each shared vehicle.
    Setting ``retained_first`` also requires each flexible visit to follow that
    vehicle's fixed visits; otherwise it can be inserted before or between them.
    The certificate concerns the primary objectives only, on this station pool.
    """
    if state_budget < 1:
        raise ValueError("The exact search needs a positive state budget.")
    continuations, fixed = continuations or {}, fixed or {}
    names = tuple(targets)
    bits = {name: 1 << i for i, name in enumerate(names)}
    required = sum(bits[name] for name in fixed)
    incoming = defaultdict(int)
    retained_by_vehicle = defaultdict(int)
    for name, (vehicles, _) in fixed.items():
        for v in vehicles:
            incoming[name] |= retained_by_vehicle[v]
            retained_by_vehicle[v] |= bits[name]
    hold_mask = sum(1 << v for v in continuations)
    pairs = tuple(combinations([v for v, time in enumerate(initial.releases) if np.isfinite(time)], 2))

    def objective(state):
        return len(state.choices), -state.latest

    def actions(state, name, pair, cutoff, *, best_only=False):
        lock = fixed.get(name)
        return _options(space, state, name, pair, None, None, initial.latest, cutoff, continuations,
                        fixed_stations=lock[1] if lock else None, greedy="earliest" if best_only else None)

    # Warm-start both insertion modes with a feasible fixed-first schedule.
    # The unfixed mode can use the much faster beam search as its incumbent.
    if fixed:
        best = initial
        for name, (pair, _) in fixed.items():
            move = next(actions(best, name, pair, deadline, best_only=True), None)
            if move is None:
                best = None
                break
            best = _advance(best, *move[1:])
        if best is not None:
            pending = [name for name in names if name not in fixed]
            while pending:
                options = [move for name in pending for pair in pairs
                           for move in actions(best, name, pair, deadline, best_only=True)]
                if not options:
                    break
                _, choice, slew = min(options, key=lambda move: move[1].start)
                best = _advance(best, choice, slew)
                pending.remove(choice.target)
    else:
        best = incumbent if incumbent is not None else initial
    full = len(names)
    frontier = defaultdict(list)
    stats = dict(visited_states=0, expanded_states=0, dominance_prunes=0, bound_prunes=0)
    exhausted = False

    def dominates(a, b):
        latest, releases = a
        other_latest, other_releases = b
        return latest <= other_latest and all(x <= y for x, y in zip(releases, other_releases))

    def visit(state, mask):
        nonlocal best, exhausted
        if stats["visited_states"] >= state_budget:
            exhausted = True
            return
        stats["visited_states"] += 1
        if (mask & required) == required and (best is None or objective(state) > objective(best)):
            best = state
        if len(state.choices) == full:
            return

        signature = (mask, state.positions, state.used & hold_mask)
        labels = frontier[signature]
        label = (state.latest, state.releases)
        if any(dominates(old, label) for old in labels):
            stats["dominance_prunes"] += 1
            return
        frontier[signature] = [old for old in labels if not dominates(label, old)] + [label]

        remaining = [name for name in names if not (mask & bits[name])]
        reachable = {}
        for name in remaining:
            # The direct-route bound ignores resource contention. Inserting a
            # visit cannot make a vehicle ready sooner: travel obeys the
            # triangle inequality and any continued stop still needs a reset.
            lower = max(initial.latest, space.lower_ready(name, state, continuations) - EPS)
            if lower <= deadline + EPS:
                reachable[name] = lower
            elif name in fixed:
                stats["bound_prunes"] += 1
                return

        max_count = len(state.choices) + len(reachable)
        if best is not None and max_count <= len(best.choices):
            optimistic = max(state.latest, *reachable.values()) if reachable else state.latest
            if max_count < len(best.choices) or optimistic >= best.latest:
                stats["bound_prunes"] += 1
                return
        stats["expanded_states"] += 1

        for name in sorted(reachable, key=lambda name: (reachable[name], name)):
            lock = fixed.get(name)
            if lock is not None and incoming[name] & ~mask:
                continue
            available_pairs = (lock[0],) if lock is not None else pairs
            for pair in available_pairs:
                if lock is None and retained_first and any(retained_by_vehicle[v] & ~mask for v in pair):
                    continue
                all_served = best is not None and len(best.choices) == full
                if all_served and max(state.latest, reachable[name]) >= best.latest:
                    stats["bound_prunes"] += 1
                    continue
                cutoff = min(deadline, best.latest) if all_served else deadline
                if max(state.latest, reachable[name]) > cutoff + EPS:
                    continue
                for _, choice, slews in actions(state, name, pair, cutoff, best_only=len(reachable) == 1):
                    # Starts are ordered for each pair, and later visits cannot reduce the maximum.
                    if best is not None and len(best.choices) == full and choice.start >= best.latest:
                        stats["bound_prunes"] += 1
                        break
                    visit(_advance(state, choice, slews), mask | bits[name])
                    if exhausted:
                        return

    visit(initial, 0)
    return best, dict(**stats, state_budget=state_budget, complete=not exhausted, feasible=best is not None,
                      confirmed_on_pool=not exhausted and best is not None,
                      infeasible_on_pool=not exhausted and best is None,
                      termination="state_budget" if exhausted else "exhausted")
