import numpy as np
from .decorators import accepts
from typing import Union


@accepts(np.array, np.array)
def portfolio_return(weights: np.array, returns: np.array) -> np.array:
    """
    Computes the return on a portfolio from constituent returns and weights
    weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix
    """
    return weights.T @ returns


@accepts(np.array, np.array)
def portfolio_vol(weights: np.array, covmat: np.array) -> np.array:
    """
    Computes the vol of a portfolio from a covariance matrix and constituent weights
    weights are a numpy array or N x 1 maxtrix and covmat is an N x N matrix
    """
    return (weights.T @ covmat @ weights)**0.5  # sqrt half for stdev (volatility)
