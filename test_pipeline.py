import pandas as pd

from src.graph import CUSTOMER, COMPETITOR, SUPPLIER, UNCONNECTED, load_graph, relation_between
from src.shocks import abnormal_returns, detect_shocks, measure_spillovers, summarize
from src.synthetic import simulate

G = load_graph("data/companies.csv", "data/relationships.csv")


def test_relation_labels():
    assert relation_between(G, "TSM", "NVDA") == CUSTOMER    # TSMC supplies NVIDIA
    assert relation_between(G, "NVDA", "TSM") == SUPPLIER
    assert relation_between(G, "NVDA", "AMD") == COMPETITOR
    assert relation_between(G, "TSLA", "META") == UNCONNECTED
    # TSMC both supplies and competes with Intel: supplier link takes priority, competitor kept
    assert relation_between(G, "TSM", "INTC") == CUSTOMER
    assert G["TSM"]["INTC"]["relations"] == {"supplier", "competitor"}


def test_every_csv_relationship_is_in_graph():
    rels = pd.read_csv("data/relationships.csv")
    for r in rels.itertuples(index=False):
        assert r.relation in G[r.source][r.target]["relations"]


def test_recovers_planted_spillover():
    returns, market = simulate(G, seed=1)
    ar = abnormal_returns(returns, market)
    events = detect_shocks(ar, market)
    summary = summarize(measure_spillovers(events, ar, G)).set_index("relation")

    assert len(events) > 50
    # customers should move with their shocked supplier, clearly and significantly
    assert summary.loc[CUSTOMER, "day0_bps"] > 100
    assert summary.loc[CUSTOMER, "day0_t"] > 5
    assert summary.loc[CUSTOMER, "days_1_to_h_t"] > 2   # planted one-day lag
    # nothing was planted for unconnected pairs
    assert abs(summary.loc[UNCONNECTED, "day0_t"]) < 3


def test_no_lookahead_in_beta():
    returns, market = simulate(G, n_days=300, seed=2)
    ar_full = abnormal_returns(returns, market)
    ar_cut = abnormal_returns(returns.iloc[:200], market.iloc[:200])
    common = ar_cut.index
    pd.testing.assert_frame_equal(ar_full.loc[common], ar_cut, check_freq=False)
