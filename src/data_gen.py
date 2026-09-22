import numpy as np
import pandas as pd
from .heston import heston_implied_vol, HESTON_PARAM_NAMES, PARAM_RANGES

S0, R = 100.0, 0.02

MONEYNESS_GRID = [0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3]
MATURITY_GRID = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
POINTS = [(m, t) for m in MONEYNESS_GRID for t in MATURITY_GRID]
COLUMNS = [f"{m:.2f}_{t:.2f}" for m, t in POINTS]


def sample_params(n: int, seed: int, param_ranges: dict = PARAM_RANGES) -> pd.DataFrame:
    '''Sample n random Heston parameter sets from uniform distributions over the ranges in param_ranges'''
    rng = np.random.default_rng(seed)
    params = {name: rng.uniform(low, high, size=n) for name, (low, high) in param_ranges.items()}
    return pd.DataFrame(params)

def surface_for_params(kappa, theta, xi, rho, v0) -> np.ndarray:
    '''Return the implied-vol surface for the standard money/time grid.'''
    return np.array([
        heston_implied_vol(S0, m * S0, t, R, kappa, theta, xi, rho, v0)
        for m, t in POINTS
    ], dtype=float)

def build_dataset(params: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''Price the implied vol surface for each parameter set. Rows where any grid
    point falls outside arbitrage-free bounds are dropped rather than filled.'''

    surfaces = pd.DataFrame(
        [surface_for_params(*row[HESTON_PARAM_NAMES]) for _, row in params.iterrows()],
        columns=COLUMNS,
    )
    keep = surfaces.notna().all(axis=1)
    dropped = (~keep).sum()
    if dropped:
        print(f"  dropped {dropped}/{len(params)} samples with out-of-bounds prices")
    return params[keep].reset_index(drop=True), surfaces[keep].reset_index(drop=True)


def generate_data(random_seed: int, trainN: int, testN: int, valN: int,
                  param_ranges: dict = PARAM_RANGES):
    '''Three independently seeded sets so they can be regenerated separately.'''
    sets = {}
    for offset, (name, n) in enumerate([("train", trainN), ("test", testN), ("val", valN)]):
        print(f"generating {name} set ({n} samples)...")
        params = sample_params(n, random_seed + offset, param_ranges)
        sets[name] = build_dataset(params)
    return sets["train"], sets["test"], sets["val"]


if __name__ == "__main__":
    import os

    (train_params, train_surfaces), (test_params, test_surfaces), (val_params, val_surfaces) = \
        generate_data(random_seed=100, trainN=6000, testN=800, valN=800)

    os.makedirs("data", exist_ok=True)
    np.savez("data/heston_surfaces.npz",
             X_train=train_params.values, Y_train=train_surfaces.values,
             X_test=test_params.values, Y_test=test_surfaces.values,
             X_val=val_params.values, Y_val=val_surfaces.values,
             moneyness_grid=MONEYNESS_GRID, maturity_grid=MATURITY_GRID)

    print(f"\ntrain={len(train_params)}  test={len(test_params)}  val={len(val_params)}  "
          f"surface points={train_surfaces.shape[1]}")
    print("saved to data/heston_surfaces.npz")
