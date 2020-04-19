'''
This module contains different methods to calculate the optimal weights of combination of assets
'''
import numpy as np
from scipy.optimize import minimize
from .porfolio import portfolio_vol, portfolio_return


def minimize_vol(covariance_matrix, expected_returns=None, target_return=None):
    """
    Returns the optimal weights that achieve the target return
    given a set of expected returns and a covariance matrix
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
