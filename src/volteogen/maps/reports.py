"""Report layouts as data: ``dim_report_line`` + ``bridge_report_line``.

The point of this module is that a BI tool should never contain statement logic. A
financial statement is a hierarchy with signs — the same shape as the US-GAAP
calculation tree — so it gets the same treatment: flatten it into a bridge with the
weight pre-multiplied down each path, and every report becomes one unchanging query::

    SELECT l.sort_order, l.label,
           sum(f.<measure> * b.weight)
    FROM fact_gl f
    JOIN bridge_report_line b ON b.account_class = f.account_class
    JOIN dim_report_line    l ON l.report_id = b.report_id AND l.line_id = b.line_id
    WHERE l.report_id = 'IS_MGMT'
    GROUP BY 1, 2
    ORDER BY 1

Adding a line, reordering the statement or building a second layout (statutory HGB,
by-function P&L) is a YAML edit. The dashboard is untouched.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import yaml


def load_report_definitions(reports_dir: Path) -> list[dict]:
    reports_dir = Path(reports_dir)
    if not reports_dir.is_dir():
        return []
    definitions = []
    for path in sorted(reports_dir.glob("*.yaml")):
        spec = yaml.safe_load(path.read_text()) or {}
        if not spec.get("report_id") or not spec.get("lines"):
            raise ValueError(f"report definition {path.name} needs report_id and lines")
        spec["_source"] = path.name
        definitions.append(spec)
    return definitions


def _resolve(spec: dict) -> tuple[list[dict], list[dict]]:
    """Expand a definition into line rows and (line -> account_class) bridge rows."""
    report_id = spec["report_id"]
    lines = spec["lines"]
    by_id = {line["id"]: line for line in lines}

    for line in lines:
        for child in line.get("children", []) or []:
            if child not in by_id:
                raise ValueError(
                    f"report {report_id}: line {line['id']!r} references unknown child {child!r}"
                )

    depth_cache: dict[str, int] = {}

    def depth_of(line_id: str, guard: int = 0) -> int:
        """Indent level: how deep a line sits under other subtotals."""
        if guard > 32:
            raise ValueError(f"report {report_id}: cycle at {line_id!r}")
        if line_id in depth_cache:
            return depth_cache[line_id]
        parents = [l for l in lines if line_id in (l.get("children") or [])]
        value = 0 if not parents else min(depth_of(p["id"], guard + 1) for p in parents) + 1
        depth_cache[line_id] = value
        return value

    def leaves(line_id: str, weight: float, guard: int = 0) -> list[tuple[str, float]]:
        """Walk to account_class leaves, multiplying signs down the path."""
        if guard > 32:
            raise ValueError(f"report {report_id}: cycle at {line_id!r}")
        line = by_id[line_id]
        own = float(line.get("sign", 1)) * weight
        out: list[tuple[str, float]] = []
        for account_class in line.get("maps") or []:
            out.append((account_class, own))
        for child in line.get("children") or []:
            out.extend(leaves(child, own, guard + 1))
        return out

    line_rows = []
    bridge_rows = []
    for order, line in enumerate(lines):
        is_subtotal = bool(line.get("children"))
        line_rows.append(
            {
                "report_id": report_id,
                "report_name": spec.get("name", report_id),
                "line_id": line["id"],
                "label": line["label"],
                "sort_order": order,
                "indent": depth_of(line["id"]),
                "is_subtotal": is_subtotal,
                "measure": spec.get("measure", "amount_gc_actual_rates"),
                "unit": spec.get("unit", "units"),
            }
        )
        # Collapse duplicate (line, class) pairs by summing weights, so a class reached
        # by two paths nets correctly instead of double counting.
        collapsed: dict[str, float] = {}
        for account_class, weight in leaves(line["id"], 1.0):
            collapsed[account_class] = collapsed.get(account_class, 0.0) + weight
        for account_class, weight in sorted(collapsed.items()):
            if weight == 0:
                continue
            bridge_rows.append(
                {
                    "report_id": report_id,
                    "line_id": line["id"],
                    "account_class": account_class,
                    "weight": weight,
                }
            )

    return line_rows, bridge_rows


def build_report_tables(reports_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    definitions = load_report_definitions(reports_dir)
    if not definitions:
        return pl.DataFrame(), pl.DataFrame()

    all_lines: list[dict] = []
    all_bridge: list[dict] = []
    for spec in definitions:
        lines, bridge = _resolve(spec)
        all_lines.extend(lines)
        all_bridge.extend(bridge)
    return pl.DataFrame(all_lines), pl.DataFrame(all_bridge)


def render(
    gl: pl.DataFrame,
    lines: pl.DataFrame,
    bridge: pl.DataFrame,
    report_id: str,
    pivot: str | None = "entity",
    measure: str | None = None,
    **filters,
) -> pl.DataFrame:
    """The reference implementation of the one query a dashboard runs.

    Kept here so the Python path and the SQL path provably agree; a BI tool runs the
    SQL form against ClickHouse and gets the same numbers.
    """
    report_lines = lines.filter(pl.col("report_id") == report_id)
    if report_lines.height == 0:
        raise KeyError(f"unknown report_id {report_id!r}")
    measure = measure or str(report_lines["measure"][0])

    frame = gl
    for column, value in filters.items():
        frame = frame.filter(pl.col(column) == value)

    joined = (
        frame.join(
            bridge.filter(pl.col("report_id") == report_id),
            on="account_class",
            how="inner",
        )
        .with_columns((pl.col(measure) * pl.col("weight")).alias("value"))
    )

    group_keys = ["line_id"] + ([pivot] if pivot else [])
    aggregated = joined.group_by(group_keys).agg(pl.col("value").sum().alias("value"))

    out = report_lines.join(aggregated, on="line_id", how="left")
    if pivot:
        out = out.pivot(on=pivot, index=["sort_order", "label", "indent", "is_subtotal"],
                        values="value").fill_null(0.0)
    else:
        out = out.select("sort_order", "label", "indent", "is_subtotal", "value").fill_null(0.0)
    return out.sort("sort_order")
