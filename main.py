"""Run the full pipeline: graph -> prices -> shocks -> spillovers -> figures.

    python main.py                     # real data from Yahoo Finance
    python main.py --synthetic         # planted spillovers, no network needed
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from src.graph import graph_summary, load_graph
from src.plots import plot_graph, plot_spillovers
from src.prices import fetch_prices, log_returns
from src.shocks import abnormal_returns, detect_shocks, measure_spillovers, summarize
from src.synthetic import simulate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default=date.today().isoformat())
    p.add_argument("--z", type=float, default=3.0, help="shock threshold in trailing std devs")
    p.add_argument("--horizon", type=int, default=5, help="days after the shock to measure")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    out = Path(args.out) / ("synthetic" if args.synthetic else "real")
    out.mkdir(parents=True, exist_ok=True)

    G = load_graph("data/companies.csv", "data/relationships.csv")
    tickers = list(G.nodes)
    print(f"Graph: {G.number_of_nodes()} companies, {G.number_of_edges()} directed edges")

    if args.synthetic:
        returns, market = simulate(G)
    else:
        prices = fetch_prices(tickers + ["SPY"], args.start, args.end)
        rets = log_returns(prices)
        returns, market = rets[tickers], rets["SPY"]

    ar = abnormal_returns(returns, market)
    events = detect_shocks(ar, market, z_threshold=args.z)
    spill = measure_spillovers(events, ar, G, horizon=args.horizon)
    summary = summarize(spill)

    graph_summary(G).to_csv(out / "graph_summary.csv", index=False)
    events.to_csv(out / "shock_events.csv", index=False)
    summary.to_csv(out / "spillover_summary.csv", index=False)
    plot_graph(G, str(Path(args.out) / "knowledge_graph.png"))
    plot_spillovers(summary, args.horizon, str(out / "spillovers.png"))

    print(f"Shock events detected: {len(events)}")
    with pd.option_context("display.width", 120):
        print(summary.to_string(index=False))
    print(f"\nOutputs written to {out}/")


if __name__ == "__main__":
    main()
