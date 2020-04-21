import pandas as pd
from typing import Union
from .tvm import annualize_rets, annualize_vol
from .metrics import sharpe_ratio, drawdown, skewness, kurtosis, value_at_risk, cond_value_at_risk


def summary_stats(r: Union[pd.Series, pd.DataFrame],
                  riskfree_rate: float = 0.03) -> pd.DataFrame:
    """
    Return a DataFrame that contains aggregated summary stats for the returns in every single column
    """
    ann_r = r.aggregate(annualize_rets, periods_per_year=12)
    ann_vol = r.aggregate(annualize_vol, periods_per_year=12)
    ann_sr = r.aggregate(
        sharpe_ratio,
        riskfree_rate=riskfree_rate,
        periods_per_year=12)
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    gaus_var5 = r.aggregate(
        value_at_risk,
        method="gaussian"
    )
    cf_var5 = r.aggregate(
        value_at_risk,
        method="gaussian_cf")  # 5% significance level
    hist_cvar5 = r.aggregate(cond_value_at_risk)  # 5% significance level
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Gaussian VaR (5%)": gaus_var5,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    })
