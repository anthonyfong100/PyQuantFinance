import pandas as pd
import numpy as np
from typing import Union
from .allocator import cppi_single_step


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
