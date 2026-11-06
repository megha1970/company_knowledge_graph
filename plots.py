"""Figures for the README."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def plot_graph(G: nx.DiGraph, path: str) -> None:
    industries = sorted({d["industry"] for _, d in G.nodes(data=True)})
    color = {ind: PALETTE[i % len(PALETTE)] for i, ind in enumerate(industries)}

    pos = nx.kamada_kawai_layout(G.to_undirected())
    fig, ax = plt.subplots(figsize=(13, 9))

    supplier = [(u, v) for u, v, d in G.edges(data=True) if "supplier" in d["relations"]]
    competitor = {tuple(sorted((u, v))) for u, v, d in G.edges(data=True) if "competitor" in d["relations"]}

    nx.draw_networkx_edges(G, pos, edgelist=supplier, ax=ax, edge_color="#555", alpha=0.6,
                           arrows=True, arrowsize=12, node_size=1300, width=1.2)
    nx.draw_networkx_edges(G, pos, edgelist=list(competitor), ax=ax, edge_color="#C44E52",
                           style="dashed", alpha=0.7, arrows=False, width=1.2)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1300,
                           node_color=[color[G.nodes[n]["industry"]] for n in G.nodes])
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_color="white", font_weight="bold")

    handles = [plt.Line2D([], [], marker="o", ls="", color=color[i], markersize=10, label=i) for i in industries]
    handles += [plt.Line2D([], [], color="#555", label="supplies →"),
                plt.Line2D([], [], color="#C44E52", ls="--", label="competes with")]
    ax.legend(handles=handles, loc="lower left", fontsize=9, frameon=False)
    ax.set_title("Company knowledge graph: supply-chain and competitor links", fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_spillovers(summary: pd.DataFrame, horizon: int, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, (col, title) in zip(axes, [("day0_bps", "Same day (day 0)"),
                                       ("days_1_to_h_bps", f"Following days (+1 to +{horizon})")]):
        ax.barh(summary["relation"], summary[col], color=PALETTE[0])
        ax.axvline(0, color="black", lw=0.8)
        tcol = col.replace("_bps", "_t")
        for y, (v, t) in enumerate(zip(summary[col], summary[tcol])):
            ax.text(max(v, 0), y, f"  t={t}", va="center", ha="left", fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("Signed abnormal return (bps)")
    axes[0].invert_yaxis()
    fig.suptitle("Do connected companies move with a shocked company?", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
