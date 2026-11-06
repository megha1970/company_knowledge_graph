# Company Knowledge Graph: Shock Spillovers Across Supply Chains

When one company's stock jumps or crashes on its own news, what happens to its suppliers, customers and competitors? This project builds a knowledge graph of 24 semiconductor and big-tech companies and runs an event study on 2019–2026 daily prices to measure how company-specific shocks spread along graph links.

![Knowledge graph](results/knowledge_graph.png)

## How it works

1. **Graph** (`src/graph.py`). Companies are nodes. Edges are typed: `supplier` (directed: TSMC → NVIDIA), `competitor`, and shared industry. A pair can carry several relations at once (TSMC both supplies and competes with Intel). Links are curated by hand from public information in `data/relationships.csv`.
2. **Abnormal returns** (`src/shocks.py`). Each stock's daily return minus beta × S&P 500 return, with beta estimated on a trailing 120-day window that ends the day before. A test checks there is no look-ahead.
3. **Shock detection.** A shock is a day when a company's abnormal return exceeds 3 trailing standard deviations, the market is calm, and at most two names in the universe move that much, so the move is likely company-specific rather than sector-wide. Repeated shocks within 5 days are treated as one story.
4. **Spillover measurement.** For each shock, every other company's abnormal return is multiplied by the shock's sign (positive = moved the same way) on day 0 and summed over days +1 to +5. Results are averaged per relation type within each event first, then tested across events, so one shock with 20 reacting stocks isn't counted as 20 independent observations.

## Validation on synthetic data

Before trusting real-data results, the pipeline is run on simulated returns where spillovers are planted along supplier edges only (40% same day, 15% next day). It recovers them:

| Relation to shocked company | Day 0 (bps) | t | Days +1..+5 (bps) | t |
|---|---|---|---|---|
| Customer | 233 | 17.2 | 138 | 4.0 |
| Supplier | 93 | 6.9 | 33 | 1.2 |
| Competitor | 46 | 4.4 | 3 | 0.1 |
| Same industry only | 45 | 5.0 | 10 | 0.5 |
| Unconnected | 3 | 0.8 | −11 | −1.0 |

Customers show the planted effect, including the lag. Smaller effects for suppliers and peers are expected: a "shock" to NVIDIA is partly TSMC's shock passed through, so TSMC and other TSMC customers co-move with it. Unconnected stocks show nothing, which is the null holding.

## Results on real data (Jan 2019 – Nov 2026, 452 shocks)

![Spillovers](results/real/spillovers.png)

| Relation to shocked company | Day 0 (bps) | t | Days +1..+5 (bps) | t |
|---|---|---|---|---|
| Customer | 83 | 7.3 | −2 | −0.1 |
| Supplier | 84 | 7.6 | 30 | 1.3 |
| Competitor | 168 | 10.3 | 22 | 0.7 |
| Same industry only | 102 | 8.9 | 38 | 1.7 |
| Unconnected | 27 | 5.2 | −4 | −0.3 |

**What this suggests**

- Every linked group moves with the shocked company on the same day, well above the unconnected baseline.
- Competitors move the most, and in the same direction. For these companies, a big move usually signals news about the whole product market (AI demand, chip cycle) rather than one firm taking share from another, so investors "read through" to rivals.
- Unconnected companies still co-move slightly (27 bps, t = 5.2). A market-only model doesn't remove the shared tech factor in this universe, so the fair comparison is each group versus the unconnected baseline, not versus zero.
- No significant drift after day 0. For heavily traded mega-caps, news appears to be priced across the network within the same day. Cohen & Frazzini (2008, *Journal of Finance*) found slower, monthly-horizon customer-to-supplier momentum in a broader, less-followed universe, which is a natural next test.

## Limitations

- **The graph is static and curated today.** Applying 2026 relationships to 2019 prices is a form of look-ahead. A production version would use time-stamped edges, for example from customer disclosures in 10-K filings.
- Small, highly correlated universe; t-stats ignore overlap between events close in time.
- Market-model abnormal returns leave sector effects in. A sector or multi-factor model would sharpen the comparison.
- The edges are illustrative; verify them before using them for anything real.

## Next steps

- Add **news co-mention edges** (two companies named in the same articles, e.g. from GDELT) and test whether they predict spillovers beyond supply-chain links.
- Time-stamped edges and a broader small/mid-cap universe to retest lagged spillovers.
- Turn spillover estimates into a signal and backtest it with transaction costs.
- Learn edge weights from returns (e.g. a graph neural network) and compare against the curated graph.

## Run it

```bash
pip install -r requirements.txt
python main.py --synthetic          # validation, no network needed
python main.py                      # real data via Yahoo Finance (cached to data/cache/)
python main.py --z 4 --horizon 10   # stricter shocks, longer window
pytest -q
```

Outputs go to `results/`: the graph figure, per-company centrality (`graph_summary.csv`), detected shocks (`shock_events.csv`) and the spillover table.

## Project layout

```
data/companies.csv        tickers, names, industries
data/relationships.csv    supplier and competitor links with notes
src/graph.py              graph construction and relation lookup
src/prices.py             price download, caching, returns
src/shocks.py             abnormal returns, shock detection, spillovers
src/synthetic.py          simulated returns with known spillovers
src/plots.py              figures
main.py                   pipeline entry point
tests/                    relation logic, planted-effect recovery, look-ahead check
```
