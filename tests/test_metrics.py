import pandas as pd
import numpy as np
from pyquantfinance.metrics import drawdown, max_drawdown, skewness, kurtosis, is_normal


def test_drawdown():
    # test 100% percent returns
    drawdown_test1 = drawdown([1, 1, 1, 1, 1])
    assert drawdown_test1["Wealth"].equals(pd.Series([2, 4, 8, 16, 32]))
    assert drawdown_test1["Previous Peak"].equals(pd.Series([2, 4, 8, 16, 32]))
    assert drawdown_test1["Drawdown"].equals(
        pd.Series([0.0, 0.0, 0.0, 0.0, 0.0]))

    # test all zero returns
    drawdown_test2 = drawdown([0, 0, 0, 0, 0])
    assert drawdown_test2["Wealth"].equals(pd.Series([1, 1, 1, 1, 1]))
    assert drawdown_test2["Previous Peak"].equals(pd.Series([1, 1, 1, 1, 1]))
    assert drawdown_test2["Drawdown"].equals(
        pd.Series([0.0, 0.0, 0.0, 0.0, 0.0]))

    # test random returns, use numpy allclose to test is if within reasonable
    # error
    drawdown_test2 = drawdown([0.43, -0.12, 1.0, 0.720, -.3])
    assert np.allclose(
        drawdown_test2["Wealth"], [
            1.43, 1.2584, 2.5168, 4.328896, 3.0302272])
    assert np.allclose(
        drawdown_test2["Previous Peak"], [
            1.43, 1.43, 2.5168, 4.328896, 4.328896])
    assert np.allclose(
        drawdown_test2["Drawdown"], [
            0.0, -0.12, 0.0, 0.0, -0.3])
