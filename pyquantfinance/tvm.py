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


@accepts((list, pd.Index), (float, pd.Series))
def _discount_table(t: Union[list, pd.Index],
                    discount_rate: Union[float, pd.Series]) -> pd.Series:
    """
    Compute the price of a pure discount bond that pays a dollar at time period t. The discount rate
    used can be either a float (fixed_rate), or a pd.Series with index being num of periods and value being
    corresponding rate for that duration (Considering 2 year loan and 20 year loan have different rates).

    Returns a series with index being time period and value being the factor to discount back to get PV
    """
    if isinstance(discount_rate, float):
        discount_rates = [(1 + discount_rate)**-i for i in t]
    else:
        discount_rates = [(1 + discount_rate.loc[i])**-i for i in t]
    return pd.Series(discount_rates, index=t)


@accepts((pd.Series, pd.DataFrame), (float, pd.Series))
def pv(flows: Union[pd.Series, pd.DataFrame],
       r: Union[float, pd.Series]):
    """
    Compute the present value of a sequence of cash flows given by the time period (as an index) and amounts
    r can be a scalar, or a Series with the index (time period) matching that of flows

    :param flows: The subsequennt cash flows with the index being a int specfying number of periods from today.
    Takes in a m,n dataframe with m being number of periods from today, and n being the number of assets
    :type flows: pd.Series, pd.DataFrame
    :param r: discount rate
    :type r: float
    """
    dates = flows.index
    # generate a discount table of values used to discount back
    discounts_table = _discount_table(dates, r)
    return discounts_table.T @ flows
