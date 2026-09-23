'''SPX option chain from Yahoo, turned into an implied-vol surface on the
training grid. Forwards and discount factors come out of put-call parity rather
than being assumed, so no rate or dividend yield is needed anywhere.
'''
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from .data_gen import MONEYNESS_GRID, MATURITY_GRID, R

TICKER = "^SPX"
MIN_MATURITY = MATURITY_GRID[0] / 2
MAX_MATURITY = MATURITY_GRID[-1] * 1.5


def fetch_chain(ticker: str = TICKER) -> tuple[pd.DataFrame, float]:
    '''Calls and puts for every expiry in range.'''
    import yfinance as yf

    handle = yf.Ticker(ticker)
    spot = handle.fast_info["lastPrice"]
    today = pd.Timestamp.now("UTC").normalize().tz_localize(None)

    frames = []
    for expiry in handle.options:
        T = (pd.Timestamp(expiry) - today).days / 365.0
        if not MIN_MATURITY <= T <= MAX_MATURITY:
            continue
        chain = handle.option_chain(expiry)
        for side, quotes in (("call", chain.calls), ("put", chain.puts)):
            frame = quotes[["strike", "bid", "ask", "volume", "openInterest"]].copy()
            frame["side"] = side
            frame["expiry"] = expiry
            frame["T"] = T
            frames.append(frame)

    return pd.concat(frames, ignore_index=True), float(spot)


def clean_quotes(chain: pd.DataFrame, max_spread: float = 0.25) -> pd.DataFrame:
    '''Drop quotes we would not trust to invert.'''
    quotes = chain[(chain.bid > 0) & (chain.ask > chain.bid)].copy()
    quotes["mid"] = 0.5 * (quotes.bid + quotes.ask)
    quotes["spread"] = (quotes.ask - quotes.bid) / quotes["mid"]
    quotes = quotes[quotes.spread < max_spread]
    quotes = quotes[quotes.openInterest.fillna(0) > 0]
    return quotes.reset_index(drop=True)


def forward_and_discount(expiry_quotes: pd.DataFrame, spot: float,
                         n_strikes: int = 10) -> tuple[float, float]:
    '''C - P = D(F - K) is a straight line in K. Fit it near the money and read
    the forward and discount off the intercept and slope.'''
    calls = expiry_quotes[expiry_quotes.side == "call"].set_index("strike")["mid"]
    puts = expiry_quotes[expiry_quotes.side == "put"].set_index("strike")["mid"]
    both = calls.index.intersection(puts.index)
    if len(both) < 3:
        return np.nan, np.nan

    nearest = both[np.argsort(np.abs(both - spot))[:n_strikes]]
    K = nearest.to_numpy(dtype=float)
    basis = (calls[nearest] - puts[nearest]).to_numpy(dtype=float)

    slope, intercept = np.polyfit(K, basis, 1)
    discount = -slope
    if not 0.5 < discount <= 1.0001:
        return np.nan, np.nan
    return intercept / discount, discount


def black76_iv(price: float, F: float, K: float, T: float, discount: float,
               is_call: bool, lo: float = 1e-3, hi: float = 3.0) -> float:
    '''Implied vol quoted off the forward.'''
    intrinsic = discount * (F - K if is_call else K - F)
    if price <= max(intrinsic, 0.0):
        return np.nan

    def gap(sigma):
        d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if is_call:
            value = discount * (F * norm.cdf(d1) - K * norm.cdf(d2))
        else:
            value = discount * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
        return value - price

    try:
        sigma = brentq(gap, lo, hi, xtol=1e-8, maxiter=200)
    except ValueError:
        return np.nan
    return np.nan if sigma <= lo * 1.01 or sigma >= hi * 0.99 else sigma


def smile_by_expiry(quotes: pd.DataFrame, spot: float) -> pd.DataFrame:
    '''One row per usable quote. OTM only, where the spreads are tightest.'''
    rows = []
    for expiry, group in quotes.groupby("expiry"):
        T = group["T"].iloc[0]
        F, discount = forward_and_discount(group, spot)
        if np.isnan(F):
            continue

        otm = group[((group.side == "call") & (group.strike >= F)) |
                    ((group.side == "put") & (group.strike < F))]
        for _, row in otm.iterrows():
            iv = black76_iv(row.mid, F, row.strike, T, discount, row.side == "call")
            if np.isnan(iv):
                continue
            rows.append({"expiry": expiry, "T": T, "F": F, "discount": discount,
                         "strike": row.strike, "log_moneyness": np.log(row.strike / F),
                         "iv": iv})
    return pd.DataFrame(rows)


def interpolate_to_grid(smiles: pd.DataFrame) -> np.ndarray:
    '''Onto the training grid: strikes in log forward moneyness, maturities in
    total variance.'''
    expiries = smiles.groupby("T")
    maturities = np.array(sorted(expiries.groups))

    surface = np.full((len(MONEYNESS_GRID), len(MATURITY_GRID)), np.nan)
    for j, T in enumerate(MATURITY_GRID):
        if not maturities[0] <= T <= maturities[-1]:
            continue
        lower = maturities[maturities <= T].max()
        upper = maturities[maturities >= T].min()

        for i, m in enumerate(MONEYNESS_GRID):
            # training grid is K/S0 at r = R with no dividend, so the same point
            # in forward terms is K/F = m * exp(-R * T)
            k = np.log(m) - R * T
            variances = []
            for maturity in (lower, upper):
                smile = expiries.get_group(maturity).sort_values("log_moneyness")
                if not smile.log_moneyness.min() <= k <= smile.log_moneyness.max():
                    variances.append(np.nan)
                    continue
                iv = np.interp(k, smile.log_moneyness, smile.iv)
                variances.append(iv ** 2 * maturity)

            if np.isnan(variances).any():
                continue
            if upper == lower:
                total_variance = variances[0]
            else:
                weight = (T - lower) / (upper - lower)
                total_variance = variances[0] * (1 - weight) + variances[1] * weight
            surface[i, j] = np.sqrt(total_variance / T)

    return surface.ravel()


def market_surface(ticker: str = TICKER) -> tuple[np.ndarray, dict]:
    chain, spot = fetch_chain(ticker)
    quotes = clean_quotes(chain)
    smiles = smile_by_expiry(quotes, spot)
    surface = interpolate_to_grid(smiles)
    info = {
        "spot": spot,
        "quotes_raw": len(chain),
        "quotes_used": len(smiles),
        "expiries_used": smiles["T"].nunique(),
        "grid_points_filled": int(np.isfinite(surface).sum()),
    }
    return surface, info


if __name__ == "__main__":
    surface, info = market_surface()
    print(info)
    grid = surface.reshape(len(MONEYNESS_GRID), len(MATURITY_GRID))
    print(pd.DataFrame(grid, index=MONEYNESS_GRID, columns=MATURITY_GRID).round(4).to_string())
