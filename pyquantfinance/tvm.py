# code used for time value of money calculations
import pandas as pd
from typing import Union
from .decorators import accepts


@accepts((pd.Series, pd.DataFrame), int)
def annualize_rets(r: Union[pd.Series, pd.DataFrame],
                   periods_per_year: int) -> Union[pd.Series, pd.DataFrame]:
    """
    Annualizes a set of returns from an pd.series or pd.dataframe object and return the annualized growth
    """
    compounded_growth = (1 + r).prod()
    n_periods = r.shape[0]
    num_years = n_periods / periods_per_year
    return compounded_growth**(1 / num_years) - 1


@accepts((pd.Series, pd.DataFrame), int)
def annualize_vol(r: Union[pd.Series, pd.DataFrame],
                  periods_per_year: int) -> Union[pd.Series, pd.DataFrame]:
    """
    Annualizes the vol of a set of returns, assuming that the variance is proportional to the duration.
    Hence the volatility (std) is proportional to sqrt(duration)
    """
    return r.std() * (periods_per_year**0.5)
