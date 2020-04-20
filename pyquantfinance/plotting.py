import numpy as np
import pandas as pd
from typing import Union
from .porfolio import portfolio_return, portfolio_vol
from .allocator import max_sharpe, minimize_vol


def plot_efficent_frontier(n_points: Union[int, float],
                           er: Union[pd.Series, pd.DataFrame],
                           cov: Union[pd.Series, pd.DataFrame],
                           show_cml: bool = False,
                           riskfree_rate: float = 0.01,
                           show_equal_weight: bool = False,
                           show_gmv: bool = False,
                           legend: bool = True,
                           style: str = '.-'
                           ):
    """
    Plots the multi-asset efficient frontier

    :params n_points: specifies the number of points to plot on the efficinet frontier
    :type n_points: int

    :params er: expected returns of the multiple assets
    :type er: pd.series, pd.DataFrame

    :params cov: covariance matrix for the multiple assets
    :type cov: pd.series, pd.Dataframe

    :params show_cml: decide whether or not to show the capital market line.
    The central market line is also known as the tangency line
    :type show_cml: bool, Optional

    :params riskfree_rate: the risk free rate to use, default value is 1%.
    This value must be present when show_cml is true
    :type riskfree_rate: bool, Optional

    :params show_equal_weight: equal weightage among all assets
    :type show_equal_weight: bool, Optional

    :params show_gmv: decide whether or not to show the global minimum volatility
    :type show_gmv: bool, Optional
    """

    # generate list of returns to plot efficient frontier and finding their
    # weight allocation
    target_expected_returns = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(cov, er, target_return)
               for target_return in target_expected_returns]

    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets,
        "Volatility": vols
    })
    ax = ef.plot.line(x="Volatility", y="Returns", style=style, legend=legend)
    if show_cml:
        ax.set_xlim(left=0)
        # get MSR
        w_msr = max_sharpe(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        # add CML
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(
            cml_x,
            cml_y,
            color='green',
            marker='o',
            linestyle='dashed',
            linewidth=2,
            markersize=10)
    if show_equal_weight:
        n = er.shape[0]
        w_ew = np.repeat(1 / n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        # add EW
        ax.plot([vol_ew], [r_ew], color='goldenrod', marker='o', markersize=10)
    if show_gmv:
        w_gmv = minimize_vol(cov)
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        # add EW
        ax.plot(
            [vol_gmv],
            [r_gmv],
            color='midnightblue',
            marker='o',
            markersize=10)

        return ax
