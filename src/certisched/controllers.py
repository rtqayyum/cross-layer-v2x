"""Scheduling policies.  Every policy sees the same causal context.

A decision is a list of (object, mode, priority) triples.  Priority classes map
to sidelink packet priorities (PPPP) and are served strictly in this order:
  1 certified critical objects (conformal budget, deadline guarantee),
  2 additional critical objects, priced by the DPP virtual queue,
  3 the RSU's background traffic (BSM/SPaT relay, not controlled here),
  4 best-effort objects (and baselines in their contract-compliant form).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .knapsack import solve_exact, solve_greedy
from .messages import CPM_HEADER_BYTES, MODES, Mode, mode_bytes
from .scene import SceneConfig, update_value

BOX = next(m for m in MODES if m.name == "box")


@dataclass
class Context:
    decision_ms: float
    objects: list[dict]          # annotated by scene.annotate
    env: dict                    # envelope values for this window (see envelope.py)
    loss_hat: float              # lagged loss estimate
    bg_bytes: float              # background bytes queued by the RSU this window
    queue_z: float               # DPP virtual queue
    beliefs: dict                # track_id -> last transmitted state
    last_tx: dict                # track_id -> state at last inclusion (ETSI rules)
    scene: SceneConfig = field(default_factory=SceneConfig)


@dataclass(frozen=True)
class Policy:
    name: str
    kind: str                     # "certisched" or a baseline id
    bound: str = "aci"            # envelope variant used for budgets
    tiers: bool = True
    ego_relevance: bool = True
    dpp: bool = True
    exact: bool = True
    opportunistic: bool = True    # class-3 best-effort fill with a point forecast
    priority: int = 4             # baselines: 4 = after RSU background (contract-compliant)
    modes: tuple = MODES
    v_dpp: float = 1.0e6         # DPP trade-off parameter V (swept in the paper)
    phy_max_bytes_per_ms: float = 3750.0  # 30 Mbit/s ceiling for infinite bounds


def _bound(ctx: Context, policy: Policy, tier: str, tau_ms: float) -> float:
    key = "S" if policy.bound == "oracle" else policy.bound
    value = float(ctx.env.get(f"{key}_{tier}", np.nan))
    if not np.isfinite(value):
        value = 0.0 if np.isnan(value) else policy.phy_max_bytes_per_ms * tau_ms
    return max(0.0, value)


def _options(objects, policy: Policy, ctx: Context):
    opts = []
    for j, obj in enumerate(objects):
        for k, mode in enumerate(policy.modes):
            value = update_value(obj, mode, ctx.loss_hat, ctx.scene)
            if not policy.ego_relevance:
                # Same estimation-error value without occlusion/conflict weighting.
                value = value / max(obj["relevance"], 1e-9)
            opts.append((j, k, float(mode_bytes(mode)), value))
    return opts


def _solve_with_headers(solve, options, modes, budget, **kwargs):
    """Solve once without repetition modes (one CPM header) and once with them
    (two headers: primary + repetition packet); keep the better feasible plan."""
    def value(sel):
        lookup = {(o, m): v for o, m, _, v in options}
        return sum(lookup[x] for x in sel)
    rep = {k for k, m in enumerate(modes) if m.copies > 1}
    plain = [o for o in options if o[1] not in rep]
    best = solve(plain, budget - CPM_HEADER_BYTES, **kwargs) if budget > CPM_HEADER_BYTES else []
    if rep and budget > 2 * CPM_HEADER_BYTES:
        alt = solve(options, budget - 2 * CPM_HEADER_BYTES, **kwargs)
        if value(alt) > value(best):
            best = alt
    return best


def certisched(ctx: Context, policy: Policy, taus: dict) -> list[tuple[dict, Mode, int]]:
    objs = ctx.objects
    if not objs:
        return []
    solve = solve_exact if policy.exact else (lambda o, b, **_: solve_greedy(o, b))
    crit_idx = [i for i, o in enumerate(objs) if policy.tiers and o["tier"] == "crit"]
    decision: list[tuple[dict, Mode, int]] = []
    used = 0.0
    if crit_idx:
        sub = [objs[i] for i in crit_idx]
        budget = _bound(ctx, policy, "crit", taus["crit"])
        price, free = 0.0, np.inf
        if policy.dpp:
            # Bytes beyond the predicted window service left after background
            # displace background traffic; they are priced at Z_t / V.
            price = ctx.queue_z / policy.v_dpp
            free = max(0.0, float(ctx.env.get("point_tot", 0.0)) - ctx.bg_bytes)
        chosen = _solve_with_headers(solve, _options(sub, policy, ctx), policy.modes, budget,
                                     price=price, free_bytes=free)
        for j, k in chosen:
            decision.append((sub[j], policy.modes[k], 1))
        used = sum(mode_bytes(m) for _, m, _ in decision)
        used += CPM_HEADER_BYTES * (1 + any(m.copies > 1 for _, m, _ in decision)) if decision else 0
    # Class 2: critical objects beyond the certified budget, sent ahead of the
    # background only while the DPP price Z_t / V makes it worthwhile.
    certified = {id(o) for o, _, _ in decision}
    if crit_idx and policy.opportunistic:
        extra = [objs[i] for i in crit_idx if id(objs[i]) not in certified]
        point = float(ctx.env.get("point_crit", 0.0))
        budget = (point if np.isfinite(point) else 0.0) - used
        if extra and budget > CPM_HEADER_BYTES:
            # Drift-plus-penalty: every class-2 byte can displace background when
            # the realized service falls short, so all of them carry Z_t / V.
            price = ctx.queue_z / policy.v_dpp if policy.dpp else 0.0
            chosen = _solve_with_headers(solve, _options(extra, policy, ctx), policy.modes, budget,
                                         price=price, free_bytes=0.0)
            for j, k in chosen:
                decision.append((extra[j], policy.modes[k], 2))
            used += sum(mode_bytes(policy.modes[k]) for _, k in chosen) + (CPM_HEADER_BYTES if chosen else 0)

    # Best-effort tier: everything not certified (normal objects and critical
    # objects that did not fit the certified budget).  It is queued at the
    # lowest priority, so an optimistic budget can only cost its own packets.
    certified = {id(o) for o, _, _ in decision}
    rest = [o for o in objs if id(o) not in certified]
    if rest:
        if policy.tiers and policy.opportunistic:
            point = float(ctx.env.get("point_norm", 0.0))
            budget = (point if np.isfinite(point) else 0.0) - used - ctx.bg_bytes
            prio = 4
        else:
            budget = _bound(ctx, policy, "norm", taus["norm"]) - used - (ctx.bg_bytes if policy.tiers else 0.0)
            prio = 4 if policy.tiers else 1
        options = _options(rest, policy, ctx)
        chosen = _solve_with_headers(solve, options, policy.modes, budget)
        values = {(o, m): v for o, m, _, v in options}
        planned = sorted(chosen, key=lambda jk: -values[jk] / mode_bytes(policy.modes[jk[1]]))
        for j, k in planned:
            decision.append((rest[j], policy.modes[k], prio))
        if policy.tiers and policy.opportunistic:
            # Overflow: remaining objects in their best value-per-byte mode, after
            # the planned set; strict priority lets them use only spare service.
            taken = {j for j, _ in chosen}
            extra = []
            for j, obj in enumerate(rest):
                if j in taken:
                    continue
                best = max(range(len(policy.modes)),
                           key=lambda k: values[(j, k)] / mode_bytes(policy.modes[k]))
                if values[(j, best)] > 0:
                    extra.append((values[(j, best)] / mode_bytes(policy.modes[best]), j, best))
            for _, j, k in sorted(extra, key=lambda e: -e[0]):
                decision.append((rest[j], policy.modes[k], prio))
    return decision


def _fill(ranked: list[dict], budget: float, mode: Mode = BOX, prio: int = 3):
    out, used = [], CPM_HEADER_BYTES
    for obj in ranked:
        b = mode_bytes(mode)
        if used + b > budget:
            continue
        out.append((obj, mode, prio))
        used += b
    return out


def _point_budget(ctx: Context) -> float:
    value = float(ctx.env.get("point_norm", 0.0))
    return value if np.isfinite(value) else 0.0


def send_all(ctx: Context, policy: Policy, taus: dict):
    return [(o, BOX, policy.priority) for o in ctx.objects]


def etsi_cpm(ctx: Context, policy: Policy, taus: dict):
    """ETSI TS 103 324 object inclusion rules + lower-layer byte budget."""
    included = []
    for o in ctx.objects:
        last = ctx.last_tx.get(o["track_id"])
        if last is None:
            included.append((np.inf, o))
            continue
        dt = ctx.decision_ms - last["capture_ms"]
        dpos = np.hypot(o["x"] - last["x"], o["y"] - last["y"])
        dspeed = abs(np.hypot(o["vx"], o["vy"]) - np.hypot(last["vx"], last["vy"]))
        dyaw = abs((o["yaw"] - last["yaw"] + np.pi) % (2 * np.pi) - np.pi)
        vru = str(o["object_class"]).lower() in {"pedestrian", "cyclist", "motorcyclist", "tricyclist"}
        if (dpos > 4.0 or dspeed > 0.5 or dyaw > np.deg2rad(4.0)
                or dt >= (500.0 if vru else 1000.0)):
            included.append((dt, o))
    included.sort(key=lambda item: -item[0])
    return _fill([o for _, o in included], _point_budget(ctx), prio=policy.priority)


def etsi_vrm(ctx: Context, policy: Policy, taus: dict):
    """Value-of-information redundancy mitigation (accuracy x relevance x novelty)."""
    eps = ctx.scene.error_scale_m
    def voi(o):
        novelty = 1.0 - np.exp(-o["receiver_error"] / eps)
        return o.get("confidence", 1.0) * np.exp(-o["ego_distance"] / 50.0) * novelty
    ranked = sorted(ctx.objects, key=lambda o: -voi(o))
    return _fill([o for o in ranked if voi(o) > 1e-6], _point_budget(ctx), prio=policy.priority)


def aoi_maxweight(ctx: Context, policy: Policy, taus: dict):
    """Max-weight age-of-information policy (Kadota et al.) with equal weights."""
    def age(o):
        belief = ctx.beliefs.get(o["track_id"])
        return 10_000.0 if belief is None else ctx.decision_ms - belief["capture_ms"]
    p = 1.0 - ctx.loss_hat
    ranked = sorted(ctx.objects, key=lambda o: -(p * age(o) * (age(o) + 100.0)))
    return _fill(ranked, _point_budget(ctx), prio=policy.priority)


def confidence_density(ctx: Context, policy: Policy, taus: dict):
    """Where2comm-style selection: confidence per transmitted byte."""
    ranked = sorted(ctx.objects, key=lambda o: -o.get("confidence", 1.0))
    return _fill(ranked, _point_budget(ctx), prio=policy.priority)


BASELINES = {
    "send_all": send_all,
    "etsi_cpm": etsi_cpm,
    "etsi_vrm": etsi_vrm,
    "aoi_maxweight": aoi_maxweight,
    "confidence_density": confidence_density,
}


def default_policies() -> list[Policy]:
    return [
        Policy("CertiSched", "certisched"),
        Policy("CertiSched-greedy", "certisched", exact=False),
        Policy("w/o conformal (point)", "certisched", bound="point"),
        Policy("w/o adaptation (split)", "certisched", bound="split"),
        Policy("fixed-level rolling", "certisched", bound="rolling"),
        Policy("last-value budget", "certisched", bound="last"),
        Policy("w/o ego relevance", "certisched", ego_relevance=False, tiers=False),
        Policy("w/o tiers", "certisched", tiers=False),
        Policy("w/o opportunistic tier", "certisched", opportunistic=False),
        Policy("w/o DPP", "certisched", dpp=False),
        Policy("box mode only", "certisched", modes=(BOX,)),
        Policy("Oracle service", "certisched", bound="oracle"),
        Policy("ETSI CPM rules", "etsi_cpm"),
        Policy("VoI redundancy mitigation", "etsi_vrm"),
        Policy("Max-weight AoI", "aoi_maxweight"),
        Policy("Confidence density", "confidence_density"),
        Policy("Send all", "send_all"),
        # Sensitivity: baselines ahead of the RSU background (not contract-compliant).
        Policy("Max-weight AoI (ahead of bg)", "aoi_maxweight", priority=2),
        Policy("VoI redundancy mitigation (ahead of bg)", "etsi_vrm", priority=2),
    ]


def decide(ctx: Context, policy: Policy, taus: dict):
    if policy.kind == "certisched":
        return certisched(ctx, policy, taus)
    return BASELINES[policy.kind](ctx, policy, taus)
