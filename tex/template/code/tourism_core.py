"""Core calculations used in Questions 1--3 (I/O and plotting omitted)."""

import numpy as np

ALPHA, SEED, B = 0.1, 20260817, 10_000


# ---------- preprocessing: training-only log-linear interpolation ----------
def log_interpolate(years, values):
    """Fill only interior gaps; the two boundaries must already be observed."""
    years, y = np.asarray(years), np.asarray(values, dtype=float).copy()
    known = np.flatnonzero(np.isfinite(y))
    for i in np.flatnonzero(~np.isfinite(y)):
        left, right = known[known < i][-1], known[known > i][0]
        weight = (years[i] - years[left]) / (years[right] - years[left])
        y[i] = np.exp((1 - weight) * np.log(y[left])
                      + weight * np.log(y[right]))
    return y


def features(years):
    """Time trend, administrative-intervention period, recovery period."""
    t = np.asarray(years, dtype=float)
    return np.column_stack(((t - 2010) / 10,
                            ((t >= 2020) & (t <= 2022)).astype(float),
                            (t >= 2023).astype(float)))


# ---------- Question 1: pre-COVID exponential model ----------
def exponential_fit(years, values):
    x = np.column_stack((np.ones(len(years)), np.asarray(years) - 2010))
    beta = np.linalg.lstsq(x, np.log(values), rcond=None)[0]
    fitted = np.exp(x @ beta)
    return beta, fitted, np.exp(beta[1]) - 1


def exponential_predict(beta, years):
    x = np.column_stack((np.ones(len(years)), np.asarray(years) - 2010))
    return np.exp(x @ beta)


# ---------- Question 2: standardized raw-scale Ridge ----------
def ridge_fit(years, values, alpha=ALPHA):
    x = features(years)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1.0
    z, y = (x - mean) / scale, np.asarray(values, dtype=float)
    coef = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]),
                           z.T @ (y - y.mean()))
    return {"mean": mean, "scale": scale, "intercept": y.mean(),
            "coef": coef, "alpha": alpha}


def ridge_predict(model, years):
    z = (features(years) - model["mean"]) / model["scale"]
    return np.maximum(0, model["intercept"] + z @ model["coef"])


def smape(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return np.mean(200 * np.abs(actual - predicted)
                   / (np.abs(actual) + np.abs(predicted)))


def rolling_predictions(years, values, test_years):
    """Expanding-window comparison; each fold sees years before its test year."""
    years, values = np.asarray(years), np.asarray(values, dtype=float)
    output = {name: [] for name in ("naive", "log_ols", "q1_exp", "ridge")}
    actual = []
    for test_year in test_years:
        train = years < test_year
        tx, ty = years[train], values[train]
        actual.append(values[years == test_year][0])
        output["naive"].append(ty[-1])
        beta_all, _, _ = exponential_fit(tx, ty)
        output["log_ols"].append(exponential_predict(beta_all, [test_year])[0])
        pre = tx <= 2019
        beta_pre, _, _ = exponential_fit(tx[pre], ty[pre])
        output["q1_exp"].append(exponential_predict(beta_pre, [test_year])[0])
        output["ridge"].append(ridge_predict(ridge_fit(tx, ty), [test_year])[0])
    return {name: smape(actual, prediction)
            for name, prediction in output.items()}


def ridge_bootstrap(years, values, future_years, repetitions=B, seed=SEED):
    """Fixed-design residual bootstrap for mean and single-observation intervals."""
    years, y = np.asarray(years), np.asarray(values, dtype=float)
    model = ridge_fit(years, y)
    z = (features(years) - model["mean"]) / model["scale"]
    zf = (features(future_years) - model["mean"]) / model["scale"]
    fitted = model["intercept"] + z @ model["coef"]
    residual = y - fitted
    residual -= residual.mean()
    inverse = np.linalg.inv(z.T @ z + model["alpha"] * np.eye(z.shape[1]))
    rng = np.random.default_rng(seed)
    sampled = residual[rng.integers(0, len(y), size=(repetitions, len(y)))]
    y_star = fitted + sampled
    intercept = y_star.mean(axis=1)
    coef = ((y_star - intercept[:, None]) @ z) @ inverse
    mean_draw = np.maximum(0, intercept[:, None] + coef @ zf.T)
    new_error = residual[rng.integers(0, len(y),
                                      size=(repetitions, len(future_years)))]
    prediction_draw = np.maximum(0, mean_draw + new_error)
    return (ridge_predict(model, future_years),
            np.quantile(mean_draw, [0.025, 0.975], axis=0),
            np.quantile(prediction_draw, [0.025, 0.975], axis=0))


# ---------- Question 3: accounting-consistent scenarios and OAT ----------
def scenario_path(income_growth, spend_growth, shock_2026=0.0):
    years = np.arange(2026, 2031)
    horizon = years - 2025
    anchor_visits, anchor_income = 2800.0, 231.0
    anchor_spend = 10_000 * anchor_income / anchor_visits
    if shock_2026:
        income = anchor_income * (1 + shock_2026) * (1 + income_growth) ** (horizon - 1)
    else:
        income = anchor_income * (1 + income_growth) ** horizon
    spend = anchor_spend * (1 + spend_growth) ** horizon
    visits = 10_000 * income / spend
    return np.column_stack((years, visits, income, spend))


def oat_2030(visitor_growth, spend_growth, level_multiplier=1.0):
    visits = 2800 * (1 + visitor_growth) ** 5 * level_multiplier
    spend = 825 * (1 + spend_growth) ** 5
    return visits, visits * spend / 10_000


SCENARIOS = {
    "baseline": scenario_path(0.08, 0.03),
    "optimistic": scenario_path(0.12, 0.04),
    "pessimistic": scenario_path(0.05, 0.02, -0.15),
}

# OAT settings around the baseline; each tuple is (visitor growth, spend growth,
# persistent level multiplier).  The shock setting uses multiplier 0.85.
BASE_GV = 1.08 / 1.03 - 1
OAT = {
    "source_low": oat_2030(BASE_GV - 0.02, 0.03),
    "source_high": oat_2030(BASE_GV + 0.02, 0.03),
    "spend_low": oat_2030(BASE_GV, 0.02),
    "spend_high": oat_2030(BASE_GV, 0.04),
    "coordination_low": oat_2030(BASE_GV, 0.03, 0.95),
    "coordination_high": oat_2030(BASE_GV, 0.03, 1.05),
    "shock_low": oat_2030(BASE_GV, 0.03, 0.85),
}
