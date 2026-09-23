import time

import numpy as np
import torch
from scipy.optimize import minimize

from .data_gen import surface_for_params
from .heston import HESTON_PARAM_NAMES, PARAM_RANGES
from .train_surrogate import load_model, normalise_input

BOUNDS = np.array([PARAM_RANGES[name] for name in HESTON_PARAM_NAMES])
MIDPOINT = BOUNDS.mean(axis=1)
WIDTHS = BOUNDS[:, 1] - BOUNDS[:, 0]

def synthetic_target(random_seed: int):
    ''' draw one parameter set from the uniform ranges and return its implied-vol surface as a DataFrame '''
    np.random.seed(random_seed)
    true_params = {name: np.random.uniform(low, high) for name, (low, high) in PARAM_RANGES.items()}
    target_surface = surface_for_params(**true_params)
    return true_params, target_surface


def classical_calibrate(target_surface, x0=None, max_iter: int = 200):
    '''Nelder-Mead straight at the COS pricer, one full re-price per objective call.'''
    target = np.asarray(target_surface, dtype=float).ravel()
    known = np.isfinite(target)          # market surfaces have gaps, synthetic ones do not
    x0 = MIDPOINT if x0 is None else np.asarray(x0, dtype=float)

    def objective(params):
        # the simplex is unconstrained, so push it back with a penalty rather than
        # clipping, which would leave flat patches for it to stall on
        below = np.clip(BOUNDS[:, 0] - params, 0, None)
        above = np.clip(params - BOUNDS[:, 1], 0, None)
        penalty = 1e3 * (below ** 2 + above ** 2).sum()

        surface = surface_for_params(*params)
        if np.isnan(surface[known]).any():
            return 1e3 + penalty
        return ((surface - target)[known] ** 2).sum() + penalty

    start = time.perf_counter()
    result = minimize(objective, x0, method="Nelder-Mead",
                      options={"maxiter": max_iter, "xatol": 1e-4, "fatol": 1e-8})
    elapsed = time.perf_counter() - start
    return result.x, elapsed, result.nit


def surrogate_calibrate(target_surface, model, y_scaler, x0=None,
                        steps: int = 500, lr: float = 0.05):
    '''Gradient descent on the 5 inputs of the frozen net. Parameters live behind a
    sigmoid so they stay inside PARAM_RANGES without clipping.'''
    target = np.asarray(target_surface, dtype=float).ravel()
    known = torch.tensor(np.isfinite(target))
    x0 = MIDPOINT if x0 is None else np.asarray(x0, dtype=float)

    lo = torch.tensor(BOUNDS[:, 0], dtype=torch.float32)
    hi = torch.tensor(BOUNDS[:, 1], dtype=torch.float32)
    start_params = torch.tensor(x0, dtype=torch.float32)
    z = torch.log((start_params - lo) / (hi - start_params)).requires_grad_(True)

    target_t = y_scaler.transform(torch.tensor(np.nan_to_num(target), dtype=torch.float32))
    optimiser = torch.optim.Adam([z], lr=lr)

    start = time.perf_counter()
    for _ in range(steps):
        optimiser.zero_grad()
        params = lo + (hi - lo) * torch.sigmoid(z)
        predicted = model(normalise_input(params.unsqueeze(0), PARAM_RANGES))
        loss = ((predicted.squeeze(0) - target_t)[known] ** 2).mean()
        loss.backward()
        optimiser.step()
    elapsed = time.perf_counter() - start

    with torch.no_grad():
        params = (lo + (hi - lo) * torch.sigmoid(z)).numpy()
    return params.astype(float), elapsed, steps


def run_comparison(n_trials: int = 10, seed: int = 100, verbose: bool = True):
    '''Both methods against the same synthetic targets, scored on time and on how
    close the recovered parameters are to the ones that made the surface.'''
    model, y_scaler = load_model()
    rows = []

    for trial in range(n_trials):
        true_params, target = synthetic_target(seed + trial)
        truth = np.array([true_params[name] for name in HESTON_PARAM_NAMES])

        classical_params, classical_time, iterations = classical_calibrate(target)
        surrogate_params, surrogate_time, _ = surrogate_calibrate(target, model, y_scaler)

        classical_err = np.abs(classical_params - truth) / WIDTHS
        surrogate_err = np.abs(surrogate_params - truth) / WIDTHS

        rows.append({
            "trial": trial,
            "true_params": truth,
            "classical_params": classical_params,
            "surrogate_params": surrogate_params,
            "classical_time": classical_time,
            "surrogate_time": surrogate_time,
            "classical_err": 100 * classical_err.mean(),
            "surrogate_err": 100 * surrogate_err.mean(),
            "iterations": iterations,
        })

        if verbose:
            print(f"trial {trial}: classical {classical_time:6.2f}s (err {100*classical_err.mean():5.2f}%)   "
                  f"surrogate {surrogate_time:.3f}s (err {100*surrogate_err.mean():5.2f}%)   "
                  f"speedup {classical_time/surrogate_time:.0f}x")

    return rows


def summarise(rows):
    classical_time = np.mean([r["classical_time"] for r in rows])
    surrogate_time = np.mean([r["surrogate_time"] for r in rows])
    return {
        "trials": len(rows),
        "classical_time_s": classical_time,
        "surrogate_time_s": surrogate_time,
        "speedup": classical_time / surrogate_time,
        "classical_err_pct": np.mean([r["classical_err"] for r in rows]),
        "surrogate_err_pct": np.mean([r["surrogate_err"] for r in rows]),
    }


def compare_on_market(surface=None, max_iter: int = 400, steps: int = 1000):
    '''Same two methods against a real SPX surface. No ground-truth parameters
    here, so the score is how well each one reproduces the quoted vols.'''
    from .market_data import market_surface

    info = {}
    if surface is None:
        surface, info = market_surface()
    surface = np.asarray(surface, dtype=float).ravel()
    known = np.isfinite(surface)

    results = {}
    for name, calibrate in [("classical", lambda: classical_calibrate(surface, max_iter=max_iter)),
                            ("surrogate", lambda: _surrogate_on(surface, steps))]:
        params, elapsed, _ = calibrate()
        fitted = surface_for_params(*params)
        residual = (fitted - surface)[known]
        results[name] = {
            "params": dict(zip(HESTON_PARAM_NAMES, params)),
            "seconds": elapsed,
            "fit_rmse": float(np.sqrt((residual ** 2).mean())),
            "at_bound": [n for n, p, (lo, hi) in zip(HESTON_PARAM_NAMES, params, BOUNDS)
                         if p < lo + 0.01 * (hi - lo) or p > hi - 0.01 * (hi - lo)],
        }

    results["info"] = info | {"points_quoted": int(known.sum()), "points_total": known.size}
    return results


def _surrogate_on(surface, steps):
    model, y_scaler = load_model()
    return surrogate_calibrate(surface, model, y_scaler, steps=steps)


if __name__ == "__main__":
    rows = run_comparison(n_trials=10)
    summary = summarise(rows)

    print(f"\n=== {summary['trials']} trials ===")
    print(f"classical  {summary['classical_time_s']:7.2f}s   {summary['classical_err_pct']:.2f}% mean param error")
    print(f"surrogate  {summary['surrogate_time_s']:7.3f}s   {summary['surrogate_err_pct']:.2f}% mean param error")
    print(f"speedup    {summary['speedup']:.0f}x")
