import itertools

import numpy as np
import pandas as pd
import pytest

from certisched import envelope, knapsack, geometry
from certisched.controllers import Policy, default_policies
from certisched.replay import ReplayConfig, run_episode, taus
from certisched.scene import SceneConfig
from certisched.synthetic import make_link, make_sequence


def brute_force(options, budget):
    groups = {}
    for o, m, c, v in options:
        groups.setdefault(o, []).append((m, c, v))
    best = 0.0
    keys = list(groups)
    for combo in itertools.product(*[[None] + g for g in groups.values()]):
        cost = sum(x[1] for x in combo if x)
        value = sum(x[2] for x in combo if x)
        if cost <= budget:
            best = max(best, value)
    return best


def value_of(options, sel):
    lookup = {(o, m): v for o, m, _, v in options}
    return sum(lookup[s] for s in sel)


def cost_of(options, sel):
    lookup = {(o, m): c for o, m, c, _ in options}
    return sum(lookup[s] for s in sel)


@pytest.mark.parametrize("seed", range(25))
def test_mckp_exact_matches_brute_force_and_greedy_half(seed):
    rng = np.random.default_rng(seed)
    options = [(o, m, float(rng.integers(1, 12) * 8), float(rng.uniform(0, 1)))
               for o in range(5) for m in range(3)]
    budget = float(rng.integers(2, 30) * 8)
    exact = knapsack.solve_exact(options, budget, grid=8.0)
    assert cost_of(options, exact) <= budget
    assert len({o for o, _ in exact}) == len(exact)
    opt = brute_force(options, budget)
    assert value_of(options, exact) == pytest.approx(opt)
    greedy = knapsack.solve_greedy(options, budget)
    assert cost_of(options, greedy) <= budget
    assert value_of(options, greedy) >= 0.5 * opt - 1e-12


def test_bev_iou_basic():
    a = (0.0, 0.0, 4.0, 2.0, 0.3)
    assert geometry.bev_iou(a, a) == pytest.approx(1.0)
    assert geometry.bev_iou(a, (20.0, 0.0, 4.0, 2.0, 0.0)) == 0.0
    # Half overlap of axis-aligned 2x2 squares -> IoU 1/3.
    assert geometry.bev_iou((0, 0, 2, 2, 0), (1, 0, 2, 2, 0)) == pytest.approx(1 / 3)
    assert geometry.bev_iou((0, 0, 2, 2, 0), (1, 0, 2, 2, np.pi)) == pytest.approx(1 / 3)


def test_aci_long_run_coverage_bound_holds_on_shifting_sequence():
    rng = np.random.default_rng(1)
    n = 6000
    target = np.r_[rng.normal(100, 5, n // 2), rng.normal(60, 20, n // 2)].clip(0)  # abrupt shift
    pred = np.full(n, 100.0)
    cfg = envelope.EnvelopeConfig(alpha=0.05, gamma=0.01)
    start = 200
    bounds, _ = envelope.aci_lower_bounds(pred, target, cfg, start)
    errs = (target[start:] < bounds[start:]).astype(float)
    gap = abs(errs.mean() - cfg.alpha)
    assert gap <= envelope.gibbs_candes_bound(cfg.alpha, cfg.gamma, n - start) + 1e-12
    # The non-adaptive split bound breaks after the shift.
    split = envelope.split_lower_bounds(pred, target, cfg, start)
    assert (target[n // 2:] < split[n // 2:]).mean() > 0.3


def _link_with_envelopes(seed=0, mean_rate=25.0):
    link = make_link(seed, mean_rate_bpms=mean_rate)
    cfg = envelope.EnvelopeConfig(alpha=0.05, gamma=0.005)
    rc = ReplayConfig()
    t = taus(rc)
    alphas = {"crit": 0.02, "norm": 0.10, "tot": 0.10}

    def model_for(tier):
        def predict(feats):
            return feats[f"target_mean{envelope.ROLL}"].fillna(0).to_numpy()
        return predict
    env = envelope.build_trace_envelopes(link, {k: model_for(k) for k in t}, t, alphas, cfg)
    return link, env, alphas


def _episode(policy, link, env, start=1000, seed=3):
    seq = make_sequence(seed)
    windows = link.iloc[start:start + 90].to_dict("records")
    envs = env.iloc[start:start + 90].to_dict("records")
    return run_episode(seq, {"windows": windows, "env": envs}, policy, ReplayConfig(),
                       SceneConfig(), f"test:{policy.name}:{start}")


def test_replay_runs_for_every_policy_and_metrics_are_bounded():
    link, env, _ = _link_with_envelopes()
    for policy in default_policies():
        out = _episode(policy, link, env)
        for key in ("critical_awareness", "hidden_awareness", "background_delivery"):
            if not np.isnan(out[key]):
                assert 0.0 <= out[key] <= 1.0, (policy.name, key, out[key])
        assert out["decisions"] > 0


def test_oracle_service_has_no_service_misses_and_send_all_over_admits():
    link, env, _ = _link_with_envelopes()
    pol = {p.name: p for p in default_policies()}
    oracle = _episode(pol["Oracle service"], link, env)
    assert oracle["crit_service_miss_rate"] in (0.0,) or np.isnan(oracle["crit_service_miss_rate"])
    scarce, scarce_env, _ = _link_with_envelopes(mean_rate=4.0)
    assert _episode(pol["Send all"], scarce, scarce_env)["over_admission_rate"] > 0


def test_certified_service_miss_rate_tracks_alpha():
    link, env, alphas = _link_with_envelopes()
    pol = {p.name: p for p in default_policies()}
    misses, total = 0, 0
    for start in range(400, 2800, 100):
        out = _episode(pol["CertiSched"], link, env, start=start, seed=start)
        if out["crit_tx"]:
            misses += out["crit_service_miss_rate"] * out["crit_tx"]
            total += out["crit_tx"]
    assert total > 50
    # Service-induced misses of certified critical objects stay near alpha_crit.
    assert misses / total <= alphas["crit"] + 0.05


def test_delay_grows_with_queued_bytes():
    from certisched.replay import latency_sample
    win = dict(lat_min_ms=1, lat_p10_ms=2, lat_p50_ms=5, lat_p90_ms=9, lat_max_ms=12)
    samples = [latency_sample(win, u) for u in np.linspace(0, 1, 11)]
    assert samples == sorted(samples) and samples[0] == 1 and samples[-1] == 12
