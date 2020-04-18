import pandas as pd
import scipy
from typing import Union
from .decorators import accepts


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
