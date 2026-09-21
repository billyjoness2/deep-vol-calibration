import numpy as np
import pytest

from src.black_scholes import bs_call_price, implied_vol

S0, R = 100.0, 0.05


# --- bs_call_price ---

def test_known_value():
    assert bs_call_price(100, 100, 1.0, 0.05, 0.2) == pytest.approx(10.4506, abs=1e-4)


def test_monotonic_in_strike_and_vol():
    prices_k = [bs_call_price(S0, K, 1.0, R, 0.2) for K in range(60, 141, 10)]
    prices_v = [bs_call_price(S0, 100, 1.0, R, s) for s in np.linspace(0.05, 1.0, 10)]
    assert np.all(np.diff(prices_k) < 0)
    assert np.all(np.diff(prices_v) > 0)


# --- implied_vol ---

@pytest.mark.parametrize("K, T, sigma", [(80, 0.5, 0.2), (100, 1.0, 0.3), (130, 2.0, 0.5)])
def test_round_trip(K, T, sigma):
    price = bs_call_price(S0, K, T, R, sigma)
    assert implied_vol(price, S0, K, T, R) == pytest.approx(sigma, abs=1e-6)


def test_monotonic_in_price():
    prices = np.linspace(8, 30, 10)
    vols = [implied_vol(p, S0, 100, 1.0, R) for p in prices]
    assert np.all(np.diff(vols) > 0)


def test_nan_below_intrinsic():
    intrinsic = S0 - 80 * np.exp(-R * 1.0)
    assert np.isnan(implied_vol(intrinsic - 0.01, S0, 80, 1.0, R))
    assert np.isnan(implied_vol(0.0, S0, 80, 1.0, R))


def test_nan_above_spot():
    assert np.isnan(implied_vol(S0 + 1.0, S0, 100, 1.0, R))
    assert np.isnan(implied_vol(S0, S0, 100, 1.0, R))


def test_nan_not_exception_on_garbage():
    assert np.isnan(implied_vol(-5.0, S0, 100, 1.0, R))
