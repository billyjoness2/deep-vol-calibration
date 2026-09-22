from torch import Tensor
import numpy as np
import pandas as pd
from .heston import cos_call_price, heston_implied_vol, HESTON_PARAM_NAMES, PARAM_RANGES

S0, R = 100.0, 0.02

MONEYNESS_GRID = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
MATURITY_GRID = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0]
POINTS = [(m, t) for m in MONEYNESS_GRID for t in MATURITY_GRID]

def sample_params(n: int, param_ranges: dict = PARAM_RANGES) -> pd.DataFrame:
    '''Sample n random Heston parameter sets from uniform distributions over the ranges in param_ranges'''
    params = {name: np.random.uniform(low, high, size=n) for name, (low, high) in param_ranges.items()}
    return pd.DataFrame(params)

def build_dataset(n: int, param_ranges: dict = PARAM_RANGES) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = sample_params(n, param_ranges)
    prices = []
    for _, row in params.iterrows():
        kappa, theta, xi, rho, v0 = row[HESTON_PARAM_NAMES]
        surface = [cos_call_price(S0, m * S0, t, R, kappa, theta, xi, rho, v0) for m, t in POINTS]
        prices.append(surface)
    return pd.DataFrame(prices, columns=[f"{m:.2f}_{t:.2f}" for m, t in POINTS])

def generate_data(random_seed: int, trainN: int, testN: int, valN: int, param_ranges: dict = PARAM_RANGES):

    np.random.seed(random_seed)
    train_params = sample_params(trainN, param_ranges)
    test_params = sample_params(testN, param_ranges)
    val_params = sample_params(valN, param_ranges)

    train_prices = build_dataset(trainN, param_ranges)
    test_prices = build_dataset(testN, param_ranges)
    val_prices = build_dataset(valN, param_ranges)

    return (train_params, train_prices), (test_params, test_prices), (val_params, val_prices)

if __name__ == "__main__":
    (train_params, train_prices), (test_params, test_prices), (val_params, val_prices) = generate_data(
        random_seed=100, trainN=90, testN=5, valN=5
    )
    print("Train params shape:", train_params.shape)
    print("Train prices shape:", train_prices.shape)
    print("Test params shape:", test_params.shape)
    print("Test prices shape:", test_prices.shape)
    print("Val params shape:", val_params.shape)
    print("Val prices shape:", val_prices.shape)
    print("Train params head:\n", train_params.head())