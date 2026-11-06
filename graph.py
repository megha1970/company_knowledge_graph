"""Build the company knowledge graph and classify how two companies are linked."""
from __future__ import annotations

import networkx as nx
import pandas as pd

# Relation labels, from the point of view of the company that was NOT shocked.
CUSTOMER = "customer of shocked co."
SUPPLIER = "supplier to shocked co."
COMPETITOR = "competitor"
SAME_INDUSTRY = "same industry only"
UNCONNECTED = "unconnected"

RELATION_ORDER = [CUSTOMER, SUPPLIER, COMPETITOR, SAME_INDUSTRY, UNCONNECTED]


def load_graph(companies_path: str, relationships_path: str) -> nx.DiGraph:
    """Nodes are companies (with name/industry); edges are typed relationships.

    supplier edges are directed  source -> target  ("source supplies target").
    competitor edges are stored in both directions.
    """
    companies = pd.read_csv(companies_path)
    rels = pd.read_csv(relationships_path)

    G = nx.DiGraph()
    for row in companies.itertuples(index=False):
        G.add_node(row.ticker, name=row.name, industry=row.industry)

    unknown = set(rels["source"]).union(rels["target"]) - set(G.nodes)
    if unknown:
        raise ValueError(f"Relationships reference unknown tickers: {sorted(unknown)}")

    def add(u: str, v: str, relation: str, note: str) -> None:
        # A pair can have several links (e.g. TSMC both supplies and competes with Intel),
        # so each edge holds a set of relation types instead of a single label.
        if not G.has_edge(u, v):
            G.add_edge(u, v, relations=set(), notes=[])
        G[u][v]["relations"].add(relation)
        G[u][v]["notes"].append(note)

    for row in rels.itertuples(index=False):
        add(row.source, row.target, row.relation, row.note)
        if row.relation == "competitor":
            add(row.target, row.source, row.relation, row.note)
    return G


def has_relation(G: nx.DiGraph, u: str, v: str, relation: str) -> bool:
    return G.has_edge(u, v) and relation in G[u][v]["relations"]


def relation_between(G: nx.DiGraph, shocked: str, other: str) -> str:
    """Strongest link from `shocked` to `other`, in a fixed priority order."""
    if has_relation(G, shocked, other, "supplier"):
        return CUSTOMER  # shocked company supplies `other`
    if has_relation(G, other, shocked, "supplier"):
        return SUPPLIER  # `other` supplies the shocked company
    if has_relation(G, shocked, other, "competitor"):
        return COMPETITOR
    if G.nodes[shocked]["industry"] == G.nodes[other]["industry"]:
        return SAME_INDUSTRY
    return UNCONNECTED


def graph_summary(G: nx.DiGraph) -> pd.DataFrame:
    """Per-company degree counts and centrality, useful for the README / sanity checks."""
    supply = nx.DiGraph([(u, v) for u, v in G.edges if has_relation(G, u, v, "supplier")])
    supply.add_nodes_from(G.nodes)
    centrality = nx.pagerank(supply.reverse())  # high = many important customers depend on you
    rows = []
    for n in G.nodes:
        rows.append({
            "ticker": n,
            "industry": G.nodes[n]["industry"],
            "customers": supply.out_degree(n),
            "suppliers": supply.in_degree(n),
            "competitors": sum(1 for _, v in G.out_edges(n) if has_relation(G, n, v, "competitor")),
            "upstream_importance": round(centrality[n], 3),
        })
    return pd.DataFrame(rows).sort_values("upstream_importance", ascending=False).reset_index(drop=True)
