import numpy as np
from .black_scholes import implied_vol

HESTON_PARAM_NAMES = ["kappa", "theta", "xi", "rho", "v0"]

PARAM_RANGES = {
    "kappa": (0.3, 5.0),
    "theta": (0.01, 0.25),
    "xi": (0.1, 1.0),
    "rho": (-0.95, -0.1),
    "v0": (0.01, 0.25),
}


def _char_func(u, T, r, kappa, theta, xi, rho, v0):
    """
    Compute the Heston characteristic function of log(S_T).
    u: Fourier frequency / cosine series transform argument
    T: time to expiration (years)
    r: annualized risk-free interest rate
    kappa: mean-reversion speed of variance
    theta: long-run variance level
    xi: volatility-of-volatility
    rho: correlation between spot and variance processes
    v0: initial variance

    Returns: characteristic function value for log(S_T).
    """

    i = 1j
    a = kappa - rho * xi * i * u
    d = np.sqrt(a ** 2 + (xi ** 2) * (i * u + u ** 2))
    g = (a - d) / (a + d)

    exp_dT = np.exp(-d * T)
    C = (kappa * theta / xi ** 2) * (
        (a - d) * T - 2.0 * np.log((1 - g * exp_dT) / (1 - g))
    )
    D = ((a - d) / xi ** 2) * ((1 - exp_dT) / (1 - g * exp_dT))

    return np.exp(C + D * v0 + i * u * r * T)


def cos_call_price(S0, K, T, r, kappa, theta, xi, rho, v0, N=160, L=10):
    """
    Price a European call option under the Heston model with the COS method.
    S0: spot price
    K: strike price
    T: time to expiration (years)
    r: annualized risk-free interest rate
    kappa: mean-reversion speed of variance
    theta: long-run variance level
    xi: volatility-of-volatility
    rho: correlation between spot and variance processes
    v0: initial variance
    N: number of cosine terms
    L: truncation-range multiplier

    Returns: European call price.
    """
    x0 = np.log(S0 / K)

    # Truncation range from the first four cumulants.
    c1 = r * T
    c2 = theta * T + (v0 - theta) * (1 - np.exp(-kappa * T)) / kappa
    a = x0 + c1 - L * np.sqrt(abs(c2))
    b = x0 + c1 + L * np.sqrt(abs(c2))

    k = np.arange(N)
    u = k * np.pi / (b - a)

    cf = _char_func(u, T, r, kappa, theta, xi, rho, v0)
    Fk = np.real(cf * np.exp(1j * u * (x0 - a)))
    Fk[0] *= 0.5

    # Cosine-series coefficients of the call payoff (standard closed form).
    def chi_psi(a, b, c, d):
        psi = np.zeros(N)
        psi[1:] = (np.sin(k[1:] * np.pi * (d - a) / (b - a))
                   - np.sin(k[1:] * np.pi * (c - a) / (b - a))) * (b - a) / (k[1:] * np.pi)
        psi[0] = d - c

        arg = k * np.pi / (b - a)
        chi = (1.0 / (1 + arg ** 2)) * (
            np.cos(k * np.pi * (d - a) / (b - a)) * np.exp(d)
            - np.cos(k * np.pi * (c - a) / (b - a)) * np.exp(c)
            + arg * np.sin(k * np.pi * (d - a) / (b - a)) * np.exp(d)
            - arg * np.sin(k * np.pi * (c - a) / (b - a)) * np.exp(c)
        )
        return chi, psi

    c, d = 0.0, b
    chi, psi = chi_psi(a, b, c, d)
    Vk = 2.0 / (b - a) * K * (chi - psi)

    price = np.exp(-r * T) * np.dot(Fk, Vk)
    return max(price, 0.0)


def heston_implied_vol(S0, K, T, r, kappa, theta, xi, rho, v0):
    """
    Solve Heston for implied volatility.
    S0: spot price
    K: strike price
    T: time to expiration (years)
    r: annualized risk-free interest rate
    kappa: mean-reversion speed of variance
    theta: long-run variance level
    xi: volatility-of-volatility
    rho: correlation between spot and variance processes
    v0: initial variance

    Returns: implied volatility (annualized).
    """
    price = cos_call_price(S0, K, T, r, kappa, theta, xi, rho, v0)
    return implied_vol(price, S0, K, T, r)


def feller_condition(kappa, theta, xi):
    """
    Check the Feller condition for the Heston variance process.
    kappa: mean-reversion speed of variance
    theta: long-run variance level
    xi: volatility-of-volatility

    Returns: True if 2 * kappa * theta > xi^2, otherwise False.
    """
    return 2 * kappa * theta > xi ** 2

