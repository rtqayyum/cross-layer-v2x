"""Causal service envelopes for deadline-constrained sidelink service.

For a relative deadline tau (ms) the deliverable bytes in window t are

    S_t(tau) = r_t * max(0, tau - L_t),

where r_t is the gross served rate of the window (bytes/ms) and L_t is the
largest packet access latency measured in that window.  If the bytes queued
at the head of a strict-priority queue do not exceed S_t(tau), every one of
them completes before tau (Lemma 1 in the paper).

The scheduler needs a lower bound \\hat S_t that is computed *before* window t
from lagged features only.  `aci_lower_bounds` implements adaptive conformal
inference (Gibbs & Candes, 2021) around a leave-one-scenario-out quantile
model, giving the deterministic long-run guarantee

    | (1/T) sum_t 1{S_t < \\hat S_t} - alpha | <= (max(alpha_1, 1-alpha_1) + gamma) / (gamma T)

for any sequence of service values.  Baselines in this module (point forecast,
last value, split conformal, fixed-level rolling conformal) carry no such
guarantee under distribution shift.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LAGS = (1, 2, 3)
ROLL = 10
BASE_COLUMNS = ("rate_bpms", "lat_max_ms", "lat_p50_ms", "loss", "cbr", "sci_count",
                "sci_mean_mcs", "sci_mean_prb", "sci_retx")


def deliverable_bytes(rate_bpms: np.ndarray, lat_max_ms: np.ndarray, tau_ms: float) -> np.ndarray:
    return np.asarray(rate_bpms, float) * np.maximum(0.0, tau_ms - np.asarray(lat_max_ms, float))


def lagged_features(windows: pd.DataFrame, tau_ms: float) -> pd.DataFrame:
    """Features for window t built only from windows t-1, t-2, ... of the same link."""
    w = windows.sort_values("window").copy()
    w["target"] = deliverable_bytes(w.rate_bpms, w.lat_max_ms, tau_ms)
    cols = [c for c in BASE_COLUMNS if c in w] + ["target"]
    feats = {}
    for c in cols:
        shifted = w[c].shift(1)
        for lag in LAGS:
            feats[f"{c}_lag{lag}"] = w[c].shift(lag)
        feats[f"{c}_mean{ROLL}"] = shifted.rolling(ROLL, min_periods=1).mean()
        feats[f"{c}_min{ROLL}"] = shifted.rolling(ROLL, min_periods=1).min()
    out = pd.DataFrame(feats, index=w.index)
    out["window"] = w["window"].to_numpy()
    out["target"] = w["target"].to_numpy()
    return out


@dataclass
class EnvelopeConfig:
    alpha: float = 0.05
    gamma: float = 0.005
    residual_window: int = 2000
    calibration_fraction: float = 0.2
    min_calibration: int = 100
    base_quantile: float = 0.5


def fit_base_model(train: pd.DataFrame, quantile: float, seed: int = 20261009):
    """Quantile gradient boosting on log1p(bytes); returns predict(features)->bytes."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    features = [c for c in train.columns if c not in ("window", "target", "trace_id", "scenario_id")]
    data = train.dropna(subset=["target"])
    model = HistGradientBoostingRegressor(loss="quantile", quantile=quantile, max_iter=200,
                                          learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=40, random_state=seed)
    model.fit(data[features], np.log1p(data["target"]))

    def predict(frame: pd.DataFrame) -> np.ndarray:
        return np.expm1(model.predict(frame[features])).clip(min=0.0)

    return predict


def _quantile(values: np.ndarray, level: float) -> float:
    """Finite-sample conformal quantile: ceil((n+1) level)-th order statistic."""
    n = len(values)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * level))
    if k > n:
        return np.inf
    if k < 1:
        return -np.inf
    return float(np.partition(values, k - 1)[k - 1])


def aci_lower_bounds(pred: np.ndarray, target: np.ndarray, cfg: EnvelopeConfig,
                     start: int) -> tuple[np.ndarray, np.ndarray]:
    """Adaptive conformal lower bounds for windows start..T-1.

    Nonconformity e_u = pred_u - target_u (overprediction).  The bound is
    pred_t - Q_{1-alpha_t}(e_{u<t}); alpha_t <= 0 gives the trivial bound 0
    (always covered) and alpha_t >= 1 gives +inf (never covered), which keeps
    alpha_t in [-gamma, 1+gamma] and yields the Gibbs-Candes bound.
    Returns (bounds, alpha_trajectory); entries before `start` are NaN.
    """
    n = len(pred)
    bounds = np.full(n, np.nan)
    alphas = np.full(n, np.nan)
    residuals = list(pred[:start] - target[:start])
    alpha_t = cfg.alpha
    for t in range(start, n):
        history = np.asarray(residuals[-cfg.residual_window:])
        if alpha_t <= 0:
            bound = 0.0
        elif alpha_t >= 1:
            bound = np.inf
        else:
            q = _quantile(history, 1.0 - alpha_t)
            # q = +inf (too few scores for the level): trivial bound 0;
            # q = -inf (level below the smallest order statistic): no bound.
            bound = 0.0 if q == np.inf else (np.inf if q == -np.inf else max(0.0, pred[t] - q))
        bounds[t], alphas[t] = bound, alpha_t
        err = float(target[t] < bound)
        alpha_t = alpha_t + cfg.gamma * (cfg.alpha - err)
        residuals.append(pred[t] - target[t])
    return bounds, alphas


def split_lower_bounds(pred, target, cfg: EnvelopeConfig, start: int) -> np.ndarray:
    """Static split conformal: residual quantile fixed on the calibration prefix."""
    q = _quantile(pred[:start] - target[:start], 1.0 - cfg.alpha)
    out = np.full(len(pred), np.nan)
    out[start:] = 0.0 if not np.isfinite(q) else np.maximum(0.0, pred[start:] - q)
    return out


def rolling_lower_bounds(pred, target, cfg: EnvelopeConfig, start: int) -> np.ndarray:
    """Fixed-level rolling conformal (no level adaptation)."""
    resid = pd.Series(pred - target)
    q = resid.shift(1).rolling(cfg.residual_window, min_periods=start).quantile(
        1.0 - cfg.alpha, interpolation="higher").to_numpy()
    out = np.full(len(pred), np.nan)
    out[start:] = np.maximum(0.0, pred[start:] - np.nan_to_num(q[start:], nan=np.inf))
    return out


def build_trace_envelopes(link_windows: pd.DataFrame, models: dict, taus: dict,
                          alphas: dict, cfg: EnvelopeConfig) -> pd.DataFrame:
    """All envelope variants for one link, every tier.

    `models[tier]` maps lagged features to a base prediction (fit without this
    link's scenario).  Returns one row per window with, for each tier k:
    S_k (truth), and lower bounds aci_k, split_k, rolling_k, point_k, last_k.
    """
    w = link_windows.sort_values("window").reset_index(drop=True)
    out = pd.DataFrame({"window": w["window"].to_numpy()})
    n = len(w)
    start = max(cfg.min_calibration, int(cfg.calibration_fraction * n))
    start = min(start, max(n - 1, 0))
    out["evaluable"] = np.arange(n) >= start
    for tier, tau in taus.items():
        feats = lagged_features(w, tau).reset_index(drop=True)
        target = feats["target"].to_numpy(float)
        pred = models[tier](feats)
        tier_cfg = EnvelopeConfig(**{**cfg.__dict__, "alpha": alphas[tier]})
        out[f"S_{tier}"] = target
        out[f"point_{tier}"] = pred
        out[f"last_{tier}"] = np.r_[np.nan, target[:-1]]
        out[f"aci_{tier}"], out[f"alpha_{tier}"] = aci_lower_bounds(pred, target, tier_cfg, start)
        out[f"split_{tier}"] = split_lower_bounds(pred, target, tier_cfg, start)
        out[f"rolling_{tier}"] = rolling_lower_bounds(pred, target, tier_cfg, start)
    return out


def gibbs_candes_bound(alpha_1: float, gamma: float, horizon: int) -> float:
    return (max(alpha_1, 1 - alpha_1) + gamma) / (gamma * max(horizon, 1))
