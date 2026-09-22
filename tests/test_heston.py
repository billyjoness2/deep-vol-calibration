import numpy as np
import pytest

from src.black_scholes import bs_call_price, implied_vol
from src.heston import cos_call_price, heston_implied_vol

S0, R = 100.0, 0.02
REALISTIC = dict(kappa=1.5, theta=0.06, xi=0.6, rho=-0.7, v0=0.05)


@pytest.mark.parametrize("K", [70, 85, 100, 115, 130])
@pytest.mark.parametrize("T", [0.25, 1.0, 3.0])
def test_collapses_to_black_scholes(K, T):
    # xi -> 0 and v0 = theta makes Heston a constant-vol model with sigma = sqrt(theta)
    theta = 0.04
    heston = cos_call_price(S0, K, T, R, kappa=1.0, theta=theta, xi=1e-4, rho=0.0, v0=theta)
    bs = bs_call_price(S0, K, T, R, np.sqrt(theta))
    assert heston == pytest.approx(bs, abs=1e-4)


def test_prices_decrease_in_strike():
    strikes = np.arange(60, 145, 5)
    prices = [cos_call_price(S0, K, 1.0, R, **REALISTIC) for K in strikes]
    assert np.all(np.diff(prices) < 0)


def test_negative_rho_gives_downward_skew():
    strikes = np.arange(70, 135, 5)
    vols = [heston_implied_vol(S0, K, 1.0, R, **REALISTIC) for K in strikes]
    assert not np.isnan(vols).any()
    assert np.all(np.diff(vols) < 0)


def test_price_within_arbitrage_bounds():
    for K in (60, 100, 150):
        for T in (0.1, 2.0):
            p = cos_call_price(S0, K, T, R, **REALISTIC)
            assert max(S0 - K * np.exp(-R * T), 0) <= p <= S0
