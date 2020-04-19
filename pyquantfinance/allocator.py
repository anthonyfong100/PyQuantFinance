'''
This module contains different methods to calculate the optimal weights of combination of assets
'''
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Union
from .porfolio import portfolio_vol, portfolio_return
from.decorators import accepts


@accepts((pd.DataFrame, np.array), (pd.DataFrame, np.array), float)
def minimize_vol(covariance_matrix: Union[pd.DataFrame, np.array],
                 expected_returns: Union[pd.DataFrame, np.array] = None,
                 target_return: float = None) -> np.array:
    """
    Returns the optimal weights that achieve the target return
    given a set of expected returns and a covariance matrix. If a target is specified,
    user need to include the expected_returns. Else if no target_return is specified,
    the function finds the global minimum volatility

    :param covariance_matrix: covariance matrix for the assets
    :type covariance_matrix: pd.DataFrame or np.array
    :param expected_returns: array of expected_returns in decimal form (e.g. 40% is 0.4)
    :type expected_returns: pd.DataFrame , np.array or None, defaults to None
    :return: An array of weights specifying the fractional weight of each asset (same order as covariance matrix column)
    :rtype: np.array
    """
    if target_return and expected_returns is None:
        raise ValueError(
            "In order to calculate the minimum volatility at a specified return, expected returns cannot be None")

    n = covariance_matrix.shape[0]  # number of assets
    init_guess = np.repeat(1 / n, n)  # initial guess
    bounds = ((0.0, 1.0),) * n  # an N-tuple of 2-tuples!
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
                        }

    # find minimum volatlity at a specified target return level
    if target_return:
        return_is_target = {'type': 'eq',
                            'args': (expected_returns,),
                            'fun': lambda weights, expected_returns: target_return - portfolio_return(weights, expected_returns)
                            }
        weights = minimize(portfolio_vol, init_guess,
                           args=(covariance_matrix,), method='SLSQP',
                           options={'disp': False},
                           constraints=(weights_sum_to_1, return_is_target),
                           bounds=bounds)

    # return the minimum volatility
    else:
        weights = minimize(portfolio_vol, init_guess,
                           args=(covariance_matrix,), method='SLSQP',
                           options={'disp': False},
                           constraints=(weights_sum_to_1),
                           bounds=bounds)

    return weights.x


@accepts(float, (pd.DataFrame, np.array), (pd.DataFrame, np.array))
def max_sharpe(riskfree_rate: float,
               expected_returns: Union[pd.DataFrame, np.array],
               covariance_matrix: Union[pd.DataFrame, np.array]) -> np.array:
    """
    Returns the weights of the portfolio that gives you the maximum sharpe ratio
    given the riskfree rate and expected returns and a covariance matrix
    """
    n = expected_returns.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n  # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
                        }

    def neg_sharpe(weights, riskfree_rate, expected_returns,
                   covariance_matrix):
        """
        Returns the negative of the sharpe ratio
        of the given portfolio
        """
        r = portfolio_return(weights, expected_returns)
        vol = portfolio_vol(weights, covariance_matrix)
        return -(r - riskfree_rate) / vol

    weights = minimize(neg_sharpe, init_guess,
                       args=(
                           riskfree_rate, expected_returns, covariance_matrix), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x


@accepts(float, float, float)
def cppi_single_step(account_value: Union[float, int],
                     floor_value: Union[float, int],
                     m: Union[float, int] = 3) -> np.array:
    '''
    Run CPPI (Constant Proportional Portfolio Insurance) on a single time step,
    returning the weights allocation of risky and riskless asset.

    :param account_value: Current account value of portfolio at this time step
    :type account_value: float, int
    :param floor_value: Floor value of portfolio that you want to insure
    :param floor_value: float,int
    :param m: Multipler of risky asset to allocate. If cushion percentage is 20% and m =3,
    will allocatte 60% (20%*3) in risky asset.Typical m ranges from 3 to 5, default to 3.
    :type m: float, int
    '''
    cushion_pct = (account_value - floor_value) / account_value
    risky_w = m * cushion_pct
    risky_w = np.clip(risky_w, 0, 1)
    safe_w = 1 - risky_w
    return np.array([risky_w, safe_w])
