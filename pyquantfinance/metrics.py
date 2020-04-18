import pandas as pd
import numpy as np
import scipy
from typing import Union
from scipy.stats import norm
from .decorators import accepts
from .tvm import annualize_rets, annualize_vol


@accepts((list, pd.Series))
def drawdown(returns_pct: Union[list, pd.Series]) -> pd.DataFrame:
    """Takes a time series , or list of asset returns in decimal form (e.g 40 percent = 0.4).
       returns a DataFrame with columns for
       the Wealth assuming starting portfolio of 1,
       the previous peaks, and
       the percentage drawdown
    """
    if isinstance(returns_pct, list):
        returns_pct = pd.Series(data=returns_pct)

    wealth_index = (1 + returns_pct).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return pd.DataFrame({"Wealth": wealth_index,
                         "Previous Peak": previous_peaks,
                         "Drawdown": drawdowns})


@accepts((pd.Series, pd.DataFrame))
def max_drawdown(returns_pct: Union[list, pd.Series]) -> float:
    drawdown_df = drawdown(returns_pct)
    return drawdown_df["Drawdown"].max()


@accepts((pd.Series, pd.DataFrame))
def skewness(r: Union[pd.Series, pd.DataFrame]):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp / sigma_r**3


@accepts((pd.Series, pd.DataFrame))
def kurtosis(r: Union[pd.Series, pd.DataFrame]):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp / sigma_r**4


@accepts((pd.Series, pd.DataFrame), float, str)
def is_normal(r, level=0.01, method="jb"):
    """
    Applies the test to determine if a Series is normal or not
    Test is applied at the 1% confidence level level by default
    Test method default uses Jarque Bera test
    Returns True if the hypothesis of normality is accepted, False otherwise
    """
    if isinstance(r, pd.DataFrame):
        # apply function to ever column
        return r.aggregate(is_normal, axis=0, level=level, method=method)
    else:
        if method == "jb":
            _, p_value = scipy.stats.jarque_bera(r)
            return p_value > level
        else:
            raise ValueError(
                "Method not implemented, currently only support jb (Jarque Bera test)")


@accepts(pd.Series, float, int)
def sharpe_ratio(returns: pd.Series, riskfree_rate: float,
                 periods_per_year: int) -> float:
    """
    Computes the annualized sharpe ratio of a set of returns expressed as a series.
    riskfree_rate is expressed as a decimal (e.g 40% = 0.4)
    period_per_year is the number of times returns compounded per year
    """
    # convert the annual riskfree rate to per period
    rf_per_period = (1 + riskfree_rate)**(1 / periods_per_year) - 1
    excess_returns = returns - rf_per_period
    ann_ex_returns = annualize_rets(excess_returns, periods_per_year)
    ann_vol = annualize_vol(returns, periods_per_year)
    return ann_ex_returns / ann_vol


@accepts((pd.Series, pd.DataFrame))
def semi_deviation(returns: Union[pd.Series, pd.DataFrame]
                   ) -> Union[pd.Series, pd.DataFrame]:
    """
    Returns the semideviation aka negative semideviation of returns
    """
    if isinstance(returns, pd.Series):
        is_negative = returns < 0  # create a mask to select entries with return < 0
        return returns[is_negative].std(ddof=0)
    else:
        return returns.aggregate(semi_deviation)


@accepts((pd.Series, pd.DataFrame), int, str)
def value_at_risk(returns: Union[pd.Series, pd.DataFrame],
                  level: int = 5, method="historic") -> Union[float, pd.Series]:
    """
    Returns the Value at Risk (percent always positive) at a specified level
    level specifies the confidence level which we want to adopt
    method = historic , gaussian or gaussian_cf (cornish fisher)
    i.e. returns the number such that "level" percent of the returns
    fall below that number, and the (100-level) percent are above
    """
    if isinstance(returns, pd.DataFrame):
        return returns.aggregate(value_at_risk, level=level, method=method)
    else:
        if method == "historic":
            return abs(np.percentile(returns, level))
        elif method == "gaussian":
            z = norm.ppf(level / 100)
            return -(returns.mean() + z * returns.std(ddof=0))
        elif method == "gaussian_cf":
            s = skewness(returns)
            k = kurtosis(returns)
            z = (z +
                 (z**2 - 1) * s / 6 +
                 (z**3 - 3 * z) * (k - 3) / 24 -
                 (2 * z**3 - 5 * z) * (s**2) / 36
                 )
            return -(returns.mean() + z * returns.std(ddof=0))
        else:
            raise ValueError(
                "Method supports only include historic, gaussian or gaussian_cf")


@accepts((pd.Series, pd.DataFrame), int)
def cond_value_at_risk(returns: Union[pd.Series, pd.DataFrame],
                       level: int = 5) -> Union[float, pd.Series]:
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(returns, pd.Series):
        # only take those entries with returns < value_at_risk
        is_beyond = returns <= value_at_risk(returns, level=level)
        return -returns[is_beyond].mean()
    else:
        return returns.aggregate(cond_value_at_risk, level=level)
