"""Multiple-choice knapsack (MCKP) solvers.

Each object j has a set of options (mode m) with integer cost c_{jm} (bytes)
and value v_{jm} >= 0; at most one option per object; total cost <= budget.
`solve_exact` is a dynamic program over a byte grid.  `solve_greedy` is the
LP-dominance greedy that, combined with the best single option, attains at
least half of the optimum (Sinha & Zoltners 1979; Kellerer et al. 2004).
"""
from __future__ import annotations

import numpy as np

Option = tuple[int, int, float, float]  # (object index, mode index, cost, value)


def solve_exact(options: list[Option], budget: float, grid: float = 8.0,
                fixed_cost: float = 0.0, price: float = 0.0, free_bytes: float = np.inf
                ) -> list[tuple[int, int]]:
    """Exact MCKP on a byte grid with an optional linear price beyond `free_bytes`.

    Objective: sum(value) - price * max(0, cost + fixed_cost - free_bytes),
    subject to cost + fixed_cost <= budget when anything is selected.
    Costs are rounded *up* to the grid so feasibility is never overstated.
    """
    if not options or budget - fixed_cost < grid:
        return []
    cap = int((budget - fixed_cost) // grid)
    groups: dict[int, list[tuple[int, int, float]]] = {}
    for obj, mode, cost, value in options:
        if value > 0:
            groups.setdefault(obj, []).append((mode, int(np.ceil(cost / grid)), value))
    if not groups:
        return []
    keys = sorted(groups)
    neg = -np.inf
    best = np.full(cap + 1, neg)
    best[0] = 0.0
    choice = np.full((len(keys), cap + 1), -1, dtype=np.int32)
    for gi, obj in enumerate(keys):
        new = best.copy()
        for k, (_, w, v) in enumerate(groups[obj]):
            if w > cap:
                continue
            candidate = np.full(cap + 1, neg)
            candidate[w:] = best[: cap + 1 - w] + v
            better = candidate > new
            new[better] = candidate[better]
            choice[gi, better] = k
        best = new
    totals = np.arange(cap + 1) * grid + fixed_cost
    penalty = price * np.maximum(0.0, totals - free_bytes) if price > 0 else 0.0
    score = best - penalty
    end = int(np.argmax(score))
    if score[end] <= 0:
        return []
    selected = []
    rem = end
    for gi in range(len(keys) - 1, -1, -1):
        k = choice[gi, rem]
        if k >= 0:
            mode, w, _ = groups[keys[gi]][k]
            # Verify the transition is consistent with the stored table.
            selected.append((keys[gi], mode))
            rem -= w
    if rem != 0:  # the table was overwritten by a later column; recompute safely
        return _reconstruct(groups, keys, cap, end)
    return selected[::-1]


def _reconstruct(groups, keys, cap, end):
    """Backtracking reconstruction with full per-stage tables (rare fallback)."""
    neg = -np.inf
    tables = [np.full(cap + 1, neg)]
    tables[0][0] = 0.0
    for obj in keys:
        prev = tables[-1]
        new = prev.copy()
        for _, w, v in groups[obj]:
            if w <= cap:
                cand = np.full(cap + 1, neg)
                cand[w:] = prev[: cap + 1 - w] + v
                new = np.maximum(new, cand)
        tables.append(new)
    selected, rem = [], end
    for gi in range(len(keys), 0, -1):
        obj = keys[gi - 1]
        prev, cur = tables[gi - 1], tables[gi]
        if np.isclose(cur[rem], prev[rem]):
            continue
        for mode, w, v in groups[obj]:
            if w <= rem and np.isclose(cur[rem], prev[rem - w] + v):
                selected.append((obj, mode))
                rem -= w
                break
    return selected[::-1]


def solve_greedy(options: list[Option], budget: float) -> list[tuple[int, int]]:
    """LP-dominance greedy with the best-single-item safeguard (>= OPT/2)."""
    groups: dict[int, list[tuple[float, float, int]]] = {}
    for obj, mode, cost, value in options:
        if value > 0 and cost <= budget:
            groups.setdefault(obj, []).append((cost, value, mode))
    increments = []
    for obj, opts in groups.items():
        opts.sort()
        # Remove dominated options, then keep the upper convex hull from (0, 0).
        hull = [(0.0, 0.0, -1)]
        for cost, value, mode in opts:
            if value <= hull[-1][1]:
                continue
            while len(hull) >= 2:
                (c1, v1, _), (c2, v2, _) = hull[-2], hull[-1]
                if (v2 - v1) * (cost - c1) <= (value - v1) * (c2 - c1):
                    hull.pop()
                else:
                    break
            hull.append((cost, value, mode))
        for (c0, v0, _), (c1, v1, m1) in zip(hull, hull[1:]):
            increments.append(((v1 - v0) / (c1 - c0), c1 - c0, obj, m1))
    increments.sort(key=lambda item: (-item[0], item[2]))
    chosen: dict[int, int] = {}
    used = 0.0
    for _, dc, obj, mode in increments:
        # Upgrades of an object are consecutive on its hull, so order is preserved.
        if used + dc > budget:
            break
        chosen[obj] = mode
        used += dc
    value_of = {(obj, m): v for obj, opts in groups.items() for (_, v, m) in opts}
    greedy_value = sum(value_of[(o, m)] for o, m in chosen.items())
    single = max(((v, o, m) for o, opts in groups.items() for (_, v, m) in opts), default=None)
    if single and single[0] > greedy_value:
        return [(single[1], single[2])]
    return sorted(chosen.items())
