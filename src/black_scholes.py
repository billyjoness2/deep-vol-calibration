import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def bs_call_price(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """
       Black-Scholes European call price:
       S0: spot price
       K: strike price
       T: time to expiration (years)
       r: annualized risk-free interest rate
       sigma: annualized volatility of the stocks returns
    """

    if T <= 0 or sigma <= 0:
        return max(S0 - K, 0.0)

    d1 = (np.log(S0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_vol(price: float, S0: float, K: float, T: float, r: float,
                 lo: float = 1e-4, hi: float = 5.0) -> float:
    '''
    Solve Black-Scholes for IV.
    price:
    '''

    def f(sigma):
        return bs_call_price(S0, K, T, r, sigma) - price

    try:
        return brentq(f, lo, hi, xtol=1e-8, maxiter=200)
    except ValueError:
        return np.nan


if __name__ == "__main__":
    # Example usage
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    price = bs_call_price(S0, K, T, r, sigma)
    print(f"Call price: {price:.4f}")

    # iv = implied_vol(price, S0, K, T, r)
    # print(f"Implied volatility: {iv:.4f}")
