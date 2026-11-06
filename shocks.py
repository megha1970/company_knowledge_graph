"""Event study: detect company-specific price shocks and measure spillovers across the graph.

Method
1. Abnormal return = stock return - beta * market return, with beta estimated on a
   trailing window that ends the day BEFORE (no look-ahead).
2. A shock is a day where a company's abnormal return is > z_threshold trailing standard
   deviations, the market itself was calm, and only a few names moved that much on the same
   day (so the move is plausibly company-specific news, not a sector-wide event).
3. For every shock, record each other company's abnormal return on day 0 and its cumulative
   abnormal return over days +1..+horizon, multiplied by the shock's sign. A positive number
   means "moved in the same direction as the shocked company".
4. Average within each relationship type per event, then test across events. Averaging per
   event first avoids treating 20 stocks reacting to one shock as 20 independent observations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .graph import RELATION_ORDER, relation_between


def abnormal_returns(returns: pd.DataFrame, market: pd.Series, window: int = 120,
                     min_periods: int = 60) -> pd.DataFrame:
    cov = returns.rolling(window, min_periods=min_periods).cov(market)
    var = market.rolling(window, min_periods=min_periods).var()
    beta = cov.div(var, axis=0).shift(1)
    return returns.sub(beta.mul(market, axis=0)).dropna(how="all")


def detect_shocks(ar: pd.DataFrame, market: pd.Series, z_threshold: float = 3.0,
                  vol_window: int = 60, market_z_max: float = 2.0,
                  max_simultaneous: int = 2, cooldown: int = 5) -> pd.DataFrame:
    sigma = ar.rolling(vol_window, min_periods=40).std().shift(1)
    z = ar / sigma

    m = market.reindex(ar.index)
    m_z = (m / m.rolling(vol_window, min_periods=40).std().shift(1)).abs()

    big = z.abs() > z_threshold
    n_big = big.sum(axis=1)
    calm_market = m_z < market_z_max
    isolated = n_big <= max_simultaneous

    events = []
    for ticker in ar.columns:
        last = None
        dates = big.index[big[ticker] & calm_market & isolated]
        for d in dates:
            pos = ar.index.get_loc(d)
            if last is not None and pos - last <= cooldown:
                continue  # skip aftershocks of the same story
            last = pos
            events.append({"date": d, "ticker": ticker, "z": float(z.at[d, ticker]),
                           "abn_return": float(ar.at[d, ticker])})
    return pd.DataFrame(events).sort_values("date").reset_index(drop=True)


def measure_spillovers(events: pd.DataFrame, ar: pd.DataFrame, G, horizon: int = 5) -> pd.DataFrame:
    rows = []
    for ev in events.itertuples(index=False):
        t = ar.index.get_loc(ev.date)
        if t + horizon >= len(ar.index):
            continue
        sign = np.sign(ev.abn_return)
        window = ar.iloc[t + 1: t + 1 + horizon]
        for other in ar.columns:
            if other == ev.ticker:
                continue
            day0 = ar.iat[t, ar.columns.get_loc(other)]
            post = window[other].sum(min_count=horizon)
            if np.isnan(day0) or np.isnan(post):
                continue
            rows.append({
                "date": ev.date, "shocked": ev.ticker, "other": other,
                "relation": relation_between(G, ev.ticker, other),
                "signed_day0": sign * day0, "signed_post": sign * post,
            })
    return pd.DataFrame(rows)


def summarize(spill: pd.DataFrame) -> pd.DataFrame:
    """Per-relation means in basis points, with t-stats computed across events."""
    per_event = (spill.groupby(["date", "shocked", "relation"])[["signed_day0", "signed_post"]]
                 .mean().reset_index())
    out = []
    for rel in RELATION_ORDER:
        sub = per_event[per_event["relation"] == rel]
        if len(sub) < 3:
            continue
        row = {"relation": rel, "n_events": len(sub)}
        for col, label in [("signed_day0", "day0"), ("signed_post", "days_1_to_h")]:
            x = sub[col].to_numpy()
            row[f"{label}_bps"] = round(x.mean() * 1e4, 1)
            row[f"{label}_t"] = round(float(stats.ttest_1samp(x, 0.0).statistic), 2)
        out.append(row)
    return pd.DataFrame(out)
