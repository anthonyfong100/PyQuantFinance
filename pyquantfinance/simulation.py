import pandas as pd
import numpy as np
import math
from typing import Union
from .decorators import accepts
from .tvm import ann_to_inst, inst_to_ann
from .allocator import cppi_single_step


@accepts(pd.Series, pd.Series, (int, float), (int, float), float, float, float)
def run_cppi(risky_r: pd.Series,
             safe_r: pd.Series = None,
             m: Union[int, float] = 3,
             start: Union[int, float] = 1000,
             floor: float = 0.8,
             riskfree_rate: float = 0.03,
             drawdown: float = None):
    """
    Run a backtest of the CPPI strategy, given a set of returns for the risky asset
    Returns a dictionary containing: Asset Value History, Risk Budget History, Risky Weight History.
    """
    # set up the CPPI parameters
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start * floor
    peak = account_value
    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r, columns=["R"])

    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        # fast way to set all values to a number
        safe_r.values[:] = riskfree_rate / 12
    # set up some DataFrames for saving intermediate values
    account_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    floorval_history = pd.DataFrame().reindex_like(risky_r)
    peak_history = pd.DataFrame().reindex_like(risky_r)

    for step in range(n_steps):

        # if drawdown is specified the floor is constantly changing based on
        # peak of asset
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak * (1 - drawdown)

        asset_alloc_pct, cushion_pct = cppi_single_step(
            account_value, floor_value, m)

        risky_alloc = asset_alloc_pct[0] * account_value
        safe_alloc = asset_alloc_pct[1] * account_value
        cushion = cushion_pct * account_value

        # recompute the new account value at the end of this step
        account_value = risky_alloc * \
            (1 + risky_r.iloc[step]) + safe_alloc * (1 + safe_r.iloc[step])

        # save the histories for analysis and plotting
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = asset_alloc_pct[0]  # risky weight pct
        account_history.iloc[step] = account_value
        floorval_history.iloc[step] = floor_value
        peak_history.iloc[step] = peak

    risky_wealth = start * (1 + risky_r).cumprod()
    return {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth,
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "m": m,
        "start": start,
        "risky_r": risky_r,
        "safe_r": safe_r,
        "drawdown": drawdown,
        "peak": peak_history,
        "floor": floorval_history
    }


@accepts(int, int, float, float, int, float, bool)
def run_GBM(n_years: int = 10,
            n_scenarios: int = 1000,
            mu: float = 0.07,
            sigma: float = 0.15,
            steps_per_year: int = 12,
            s_0: float = 100.0,
            prices: bool = True) -> pd.DataFrame:
    """
    Evolution of Geometric Brownian Motion trajectories, such as for Stock Prices through Monte Carlo
    :param n_years:  The number of years to generate data for
    :param n_paths: The number of scenarios/trajectories
    :param mu: Annualized Drift, e.g. Market Return
    :param sigma: Annualized Volatility
    :param steps_per_year: granularity of the simulation
    :param s_0: initial value
    :param prices:if prices is true, return the resultant price assuming starting from s_0, else return the %change for each period
    :return: a numpy array of m x n,  m row (number of periods) x n col (number of scenarios)
    """
    # Derive per-step Model Parameters from User Specifications
    dt = 1 / steps_per_year
    n_steps = int(n_years * steps_per_year) + 1
    # the standard way ...
    # rets_plus_1 = np.random.normal(loc=mu*dt+1, scale=sigma*np.sqrt(dt), size=(n_steps, n_scenarios))
    # without discretization error ...
    rets_plus_1 = np.random.normal(
        loc=(
            1 + mu)**dt,  # mean, raise to the power of dt to account for compounding in a year
        scale=(
            sigma * np.sqrt(dt)),  # stdev used to add noise --> brownian motion term
        size=(
            n_steps,
            n_scenarios))
    rets_plus_1[0] = 1
    ret_val = s_0 * \
        pd.DataFrame(rets_plus_1).cumprod(
        ) if prices else pd.DataFrame(rets_plus_1 - 1)
    return ret_val


@accepts(int, int, float, float, float, int, float)
def run_CIR(n_years: int = 10, n_scenarios: int = 1, a: float = 0.05, b: float = 0.03,
            sigma: float = 0.05, steps_per_year: int = 12, r_0: float = None):
    """
    Generate random interest rate evolution over time using the CIR model
    b and r_0 are assumed to be the annualized rates, not the short rate
    and the returned values are the annualized rates as well. To read more:
    https://en.wikipedia.org/wiki/Cox%E2%80%93Ingersoll%E2%80%93Ross_model
    """
    if r_0 is None:
        r_0 = b

    # CIR model uses the instantaneous returns (short rate)
    r_0 = ann_to_inst(r_0)
    dt = 1 / steps_per_year
    num_steps = int(n_years * steps_per_year) + \
        1  # because n_years might be a float

    shock = np.random.normal(
        0, scale=np.sqrt(dt), size=(
            num_steps, n_scenarios))
    rates = np.empty_like(shock)
    rates[0] = r_0

    # For Price Generation of bonds
    h = math.sqrt(a**2 + 2 * sigma**2)
    prices = np.empty_like(shock)

    # calculate the price of a zero coupon bond using the CIR model
    def price(ttm, r):
      # ttm stands for time to maturity of bond
        _A = ((2 * h * math.exp((h + a) * ttm / 2)) / (2 * h + (h + a)
                                                       * (math.exp(h * ttm) - 1)))**(2 * a * b / sigma**2)
        _B = (2 * (math.exp(h * ttm) - 1)) / \
            (2 * h + (h + a) * (math.exp(h * ttm) - 1))
        _P = _A * np.exp(-_B * r)
        return _P
    prices[0] = price(n_years, r_0)

    for step in range(1, num_steps):
        r_t = rates[step - 1]
        d_r_t = a * (b - r_t) * dt + sigma * np.sqrt(r_t) * \
            shock[step]  # cir process

        rates[step] = abs(r_t + d_r_t)
        # generate prices at time t as well ...
        time_maturity = n_years - step * dt
        prices[step] = price(time_maturity, rates[step])

    rates = pd.DataFrame(data=inst_to_ann(rates), index=range(num_steps))
    # for prices
    prices = pd.DataFrame(data=prices, index=range(num_steps))
    ###
    return rates, prices
