"""US-GAAP 2026 income-statement calculation tree (FASB UGT).

The seed is the full taxonomy graph. Each concept carries calculation children ``cc``
as ``{"c": child, "w": weight, "net": network}``. The generator uses the
``StatementOfIncome`` network, which holds **531 arcs** with +/-1 weights and reaches
depth 15 — real GAAP structure, including the negative-weight arcs that make naive
``SUM(amount)`` roll-ups wrong.

Weights matter: the group P&L view must be summed as ``SUM(amount * weight_product)``,
which is why :func:`build_gaap_bridge` pre-multiplies weights down each path.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import polars as pl

from .nodes import GROUP, POSTING, AccountNode

COA_ID = "USGAAP"
DEFAULT_NETWORK = "StatementOfIncome"


def _load_taxonomy(seeds_dir: Path) -> dict:
    path = Path(seeds_dir) / "usgaap2026_taxonomy.json"
    if not path.exists():
        raise FileNotFoundError(
            f"missing US-GAAP taxonomy seed at {path}; run seeds/fetch_seeds.sh "
            "(see seeds/SOURCES.md)"
        )
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_gaap_arcs(seeds_dir: Path, network: str = DEFAULT_NETWORK) -> pl.DataFrame:
    """Every calculation arc of one network, as parent -> child with a signed weight."""
    data = _load_taxonomy(seeds_dir)
    concepts: dict = data["concepts"]

    rows = []
    for parent, concept in concepts.items():
        for arc in concept.get("cc") or []:
            if arc.get("net") != network:
                continue
            rows.append(
                {
                    "network": network,
                    "parent": parent,
                    "child": arc["c"],
                    "weight": float(arc.get("w", 1.0)),
                    "order": float(arc.get("o", 0.0) or 0.0),
                }
            )
    if not rows:
        raise ValueError(
            f"no calculation arcs found for network {network!r} in the US-GAAP seed"
        )
    return pl.DataFrame(rows).sort(["parent", "order", "child"])


def load_gaap_nodes(seeds_dir: Path, network: str = DEFAULT_NETWORK) -> list[AccountNode]:
    """The calc network as a parent-child tree of concepts.

    A concept can legitimately appear under more than one parent in a taxonomy. The
    canonical tree keeps the first (lowest-order) parent so it stays a tree; every arc
    — including the extra ones — survives in :func:`load_gaap_arcs` and in the bridge,
    which is what the P&L query actually reads.
    """
    data = _load_taxonomy(seeds_dir)
    concepts: dict = data["concepts"]
    arcs = load_gaap_arcs(seeds_dir, network)

    chosen_parent: dict[str, tuple[str, float]] = {}
    for row in arcs.iter_rows(named=True):
        if row["child"] not in chosen_parent:
            chosen_parent[row["child"]] = (row["parent"], row["weight"])

    involved: set[str] = set(arcs["parent"].to_list()) | set(arcs["child"].to_list())
    roots = sorted(c for c in involved if c not in chosen_parent)

    def label(name: str) -> str:
        concept = concepts.get(name) or {}
        return concept.get("l") or name

    def is_abstract(name: str) -> bool:
        concept = concepts.get(name) or {}
        return bool(concept.get("a"))

    has_children = set(arcs["parent"].to_list())

    nodes: list[AccountNode] = []
    root_id = f"{COA_ID}:ROOT:{network}"
    nodes.append(
        AccountNode(
            coa_id=COA_ID,
            node_id=root_id,
            name=f"US-GAAP 2026 — {network}",
            node_type=GROUP,
            income_balance="PL",
            sort_order=-1,
            attrs={"network": network, "structural": True},
        )
    )

    order_by_child = {
        row["child"]: row["order"] for row in arcs.iter_rows(named=True)
    }

    for order, concept_name in enumerate(sorted(involved)):
        parent_info = chosen_parent.get(concept_name)
        if parent_info is None:
            parent_id = root_id
            weight = 1.0
        else:
            parent_id = f"{COA_ID}:{parent_info[0]}"
            weight = parent_info[1]
        concept = concepts.get(concept_name) or {}
        nodes.append(
            AccountNode(
                coa_id=COA_ID,
                node_id=f"{COA_ID}:{concept_name}",
                parent_id=parent_id,
                code=concept_name,
                name=label(concept_name),
                node_type=GROUP if concept_name in has_children else POSTING,
                income_balance="PL",
                category=str(concept.get("k") or ""),
                subcategory=str(concept.get("b") or ""),  # debit / credit balance
                sort_order=int(order_by_child.get(concept_name, order)),
                attrs={
                    "weight_to_parent": weight,
                    "abstract": is_abstract(concept_name),
                    "period_type": concept.get("p"),
                    "balance": concept.get("b"),
                    "is_root": parent_info is None,
                },
            )
        )

    if not roots:
        raise ValueError(f"US-GAAP network {network!r} has no root concept")
    return nodes


def build_gaap_bridge(
    arcs: pl.DataFrame, network: str = DEFAULT_NETWORK
) -> pl.DataFrame:
    """Ancestor bridge with weights pre-multiplied down each path.

    Emits ``(ancestor, node, depth_diff, weight_product)`` including the self-row at
    depth 0, so a group P&L is exactly::

        SELECT b.ancestor, SUM(f.amount * b.weight_product)
        FROM fact_gl f JOIN bridge_usgaap b ON b.node = f.gaap_concept
        GROUP BY b.ancestor

    Multi-parent concepts contribute one path per parent — that is the taxonomy's real
    behaviour, not a bug.
    """
    children: dict[str, list[tuple[str, float]]] = {}
    for row in arcs.iter_rows(named=True):
        children.setdefault(row["parent"], []).append((row["child"], row["weight"]))

    nodes: set[str] = set(arcs["parent"].to_list()) | set(arcs["child"].to_list())

    rows = [
        {
            "network": network,
            "ancestor": node,
            "node": node,
            "depth_diff": 0,
            "weight_product": 1.0,
        }
        for node in sorted(nodes)
    ]

    for ancestor in sorted(nodes):
        queue: deque[tuple[str, int, float]] = deque([(ancestor, 0, 1.0)])
        seen_on_path: dict[str, int] = {}
        while queue:
            current, depth, weight = queue.popleft()
            if depth > 32:  # taxonomy depth is ~15; guard against pathological cycles
                continue
            for child, arc_weight in children.get(current, []):
                if seen_on_path.get(child, 99) <= depth + 1:
                    continue
                seen_on_path[child] = depth + 1
                product = weight * arc_weight
                rows.append(
                    {
                        "network": network,
                        "ancestor": ancestor,
                        "node": child,
                        "depth_diff": depth + 1,
                        "weight_product": product,
                    }
                )
                queue.append((child, depth + 1, product))

    return pl.DataFrame(rows).unique(
        subset=["network", "ancestor", "node", "depth_diff"], keep="first"
    )
