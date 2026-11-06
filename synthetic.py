"""Synthetic returns with KNOWN spillovers, used to check the pipeline recovers them."""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate(G, n_days: int = 1500, spill_day0: float = 0.4, spill_lag1: float = 0.15,
             seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Market + industry factors + idiosyncratic noise with occasional news jumps.

    A customer absorbs `spill_day0` of each supplier's idiosyncratic move the same day and
    `spill_lag1` of it the next day. Nothing else is linked, so unconnected stocks should
    show ~0 spillover.
    """
    rng = np.random.default_rng(seed)
    tickers = list(G.nodes)
    dates = pd.bdate_range("2018-01-01", periods=n_days)

    market = rng.normal(0.0003, 0.010, n_days)
    industries = sorted({G.nodes[t]["industry"] for t in tickers})
    ind_factor = {i: rng.normal(0, 0.006, n_days) for i in industries}

    idio = rng.normal(0, 0.012, (n_days, len(tickers)))
    jumps = rng.random((n_days, len(tickers))) < 0.01
    idio += jumps * rng.choice([-1, 1], size=idio.shape) * rng.uniform(0.05, 0.10, idio.shape)

    col = {t: i for i, t in enumerate(tickers)}
    rets = np.zeros_like(idio)
    for t in tickers:
        j = col[t]
        rets[:, j] = rng.uniform(0.8, 1.3) * market + ind_factor[G.nodes[t]["industry"]] + idio[:, j]
    for s, c, d in G.edges(data=True):
        if "supplier" in d["relations"]:
            rets[:, col[c]] += spill_day0 * idio[:, col[s]]
            rets[1:, col[c]] += spill_lag1 * idio[:-1, col[s]]

    return pd.DataFrame(rets, index=dates, columns=tickers), pd.Series(market, index=dates, name="SPY")
