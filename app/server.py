"""Drill-down income statement over the 400M-row fact table.

Three things carry the performance, all measured in `clickhouse/HIERARCHY_BENCHMARK.md`:

1. Hierarchy functions need a UInt64 key and a HASHED layout, so each tree has a
   surrogate id (`clickhouse/drilldown_setup.sql`).
2. `dictIsIn` in a WHERE clause reads every row (200M vs 17M for the same filter). The
   dictionary is used to EXPAND an ancestor into its descendant set instead.
3. The aggregate CTE is deliberately UNFILTERED: a `WHERE … IN (subquery)` stops the
   optimizer using the aggregate projection, and the projection collapses the table to
   ~1,500 rows. That took the drill query from 1,050 ms to ~20 ms.

Reporting bases are swapped at runtime, every hop a dictGet — so no mapping logic lives
in this file:

    mgmt      account_class -> report line               (bridge_report_line)
    function  (account_class, cc function) -> mgmt line  (dict_class_function)
    gaap      account_class -> concept -> signed tree     (dict_class_gaap + bridge_usgaap)

    python app/server.py            # http://localhost:8811
"""
from __future__ import annotations

import http.server
import json
import socketserver
import time
import urllib.parse
import urllib.request
from pathlib import Path

CLICKHOUSE = "http://127.0.0.1:8123/"
PORT = 8811
HERE = Path(__file__).parent

DIMS = {
    "cc": {"dict": "volteo.dict_cc", "table": "volteo.hier_cc",
           "column": "cc_id", "label": "Cost centre"},
    "product": {"dict": "volteo.dict_product", "table": "volteo.hier_product",
                "column": "product_id", "label": "Product"},
    "market": {"dict": "volteo.dict_market", "table": "volteo.hier_market",
               "column": "market_id", "label": "Market"},
    # The chart of accounts is a hierarchy like any other: four local charts of
    # different depth (Sage 2, SKR03/PGC 4, Business Central 6) unioned under one root.
    "account": {"dict": "volteo.dict_account", "table": "volteo.hier_account",
                "column": "account_id", "label": "Account"},
}

# Two fact tables, deliberately different in character:
#   journal  ONE company at document-line grain, 18 years. Meaningful figures, and NO
#            projection — so every query is an honest aggregation over the full table.
#   ensemble 295 stacked companies with an aggregate projection. Figures are a sum of
#            295 books (trillions), but it shows what a pre-aggregate buys you.
SOURCES = {
    "journal": {"table": "volteo.fact_journal", "label": "Journal lines (one company)"},
    "ensemble": {"table": "volteo.fact_gl_400m", "label": "Ensemble (295 runs, projected)"},
}

VIEWS = {
    "mgmt": "Management (by nature)",
    "function": "Management (by function)",
    "gaap": "US-GAAP StatementOfIncome",
}


def query(sql: str) -> list[dict]:
    body = (sql + "\nFORMAT JSONEachRow").encode()
    request = urllib.request.Request(CLICKHOUSE, data=body)
    with urllib.request.urlopen(request, timeout=180) as response:
        text = response.read().decode()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def root_id(dim: str) -> int:
    spec = DIMS[dim]
    rows = query(f"SELECT id FROM {spec['table']} WHERE parent_id = 0 ORDER BY id LIMIT 1")
    return int(rows[0]["id"]) if rows else 1


def _measure_sql(view: str, column: str, node: int, dictionary: str, na_clause: str,
                 table: str) -> str:
    """SQL for one reporting basis. The CTEs are shared; only the final SELECT differs."""
    common = f"""
        WITH
            kids AS (
                SELECT arrayJoin(dictGetChildren('{dictionary}', toUInt64({node}))) AS kid
            ),
            map AS (
                SELECT kid,
                       arrayJoin(arrayPushFront(
                           dictGetDescendants('{dictionary}', kid, 0), kid)) AS nid
                FROM kids{na_clause}
            ),
            agg AS (
                SELECT account_class AS ac, {column} AS nid, sum(amount) AS amt
                FROM {table}
                GROUP BY ac, nid
            )"""

    if view == "gaap":
        # Roll up the FASB tree with its SIGNED weights: 61 of the 531 arcs carry -1,
        # so a plain SUM(amount) over this tree is quietly wrong.
        return common + """
        SELECT
            br.ancestor                      AS label,
            toInt32(min(o.depth))            AS sort_order,
            toUInt8(1)                       AS is_subtotal,
            m.kid                            AS child,
            sum(agg.amt * br.weight_product) AS value
        FROM agg
        INNER JOIN map AS m ON m.nid = agg.nid
        INNER JOIN volteo.bridge_usgaap AS br
                ON br.node = dictGet('volteo.dict_class_gaap', 'gaap_concept', tuple(agg.ac))
        LEFT JOIN volteo.gaap_order AS o ON o.concept = br.ancestor
        GROUP BY label, child"""

    if view == "function":
        # The by-nature -> by-function pivot. Classes that apply regardless of function
        # carry a '~ANY~' row, so fall back to it when the pair is absent.
        return common + """
        SELECT
            if(dictHas('volteo.dict_class_function',
                       tuple(agg.ac, dictGet('volteo.dict_cc_function', 'function',
                                             toUInt64(agg.nid)))),
               dictGet('volteo.dict_class_function', 'management_line',
                       tuple(agg.ac, dictGet('volteo.dict_cc_function', 'function',
                                             toUInt64(agg.nid)))),
               dictGet('volteo.dict_class_function', 'management_line',
                       tuple(agg.ac, '~ANY~')))  AS label,
            toInt32(0)                           AS sort_order,
            toUInt8(0)                           AS is_subtotal,
            m.kid                                AS child,
            sum(agg.amt * -1)                    AS value
        FROM agg
        INNER JOIN map AS m ON m.nid = agg.nid
        GROUP BY label, child"""

    return common + """
        SELECT
            l.label                 AS label,
            l.sort_order            AS sort_order,
            l.is_subtotal           AS is_subtotal,
            m.kid                   AS child,
            sum(agg.amt * b.weight) AS value
        FROM agg
        INNER JOIN map AS m ON m.nid = agg.nid
        INNER JOIN volteo.bridge_report_line AS b ON b.account_class = agg.ac
        INNER JOIN volteo.dim_report_line    AS l
                ON l.report_id = b.report_id AND l.line_id = b.line_id
        WHERE l.report_id = 'IS_MGMT'
        GROUP BY label, sort_order, is_subtotal, child
        ORDER BY sort_order"""


def _subledger_drill(dim: str, node: int, current: dict, breadcrumb: list, spec: dict) -> dict:
    """Below FAMILY the ledger has nothing — spec 5 keeps SKU detail in the sub-ledger."""
    dictionary = spec["dict"]
    started = time.time()
    rows = query(f"""
        WITH
            kids AS (
                SELECT arrayJoin(dictGetChildren('{dictionary}', toUInt64({node}))) AS kid
            ),
            map AS (
                SELECT kid, arrayJoin(arrayPushFront(
                           dictGetDescendants('{dictionary}', kid, 0), kid)) AS nid
                FROM kids
            ),
            agg AS (
                SELECT product_id AS nid, sum(revenue) AS rev, sum(cogs) AS cg
                FROM volteo.sub_keyed GROUP BY nid
            )
        SELECT m.kid AS child, sum(agg.rev) AS rev, sum(agg.cg) AS cg
        FROM agg INNER JOIN map AS m ON m.nid = agg.nid
        GROUP BY child""")
    elapsed_ms = (time.time() - started) * 1000

    by_child = {int(r["child"]): r for r in rows}
    lines = []
    for order, label, key, sign in (
        (0, "Gross revenue", "rev", 1.0),
        (1, "COGS - standard cost", "cg", -1.0),
        (2, "Gross profit", None, 1.0),
    ):
        values, total = {}, 0.0
        for cid, row in by_child.items():
            value = (float(row["rev"]) - float(row["cg"])) if key is None \
                else float(row[key]) * sign
            values[str(cid)] = value
            total += value
        lines.append({"sort_order": order, "label": label, "is_subtotal": key is None,
                      "values": values, "total": total})

    ids = ",".join(str(i) for i in sorted(by_child)) or "0"
    meta = query(
        f"SELECT id, name, node_type, dictGetChildren('{dictionary}', toUInt64(id)) != [] "
        f"AS has_children FROM {spec['table']} WHERE id IN ({ids}) ORDER BY name")
    return {
        "dim": dim, "source": "subledger", "view": "mgmt", "views": VIEWS,
        "node": {"id": int(current["id"]), "name": current["name"],
                 "node_type": current["node_type"]},
        "breadcrumb": breadcrumb,
        "children": [{"id": int(c["id"]), "name": c["name"], "node_type": c["node_type"],
                      "has_children": bool(int(c["has_children"]))} for c in meta],
        "lines": lines,
        "elapsed_ms": round(elapsed_ms, 1),
    }


def drill(dim: str, node: int, view: str = "mgmt", source: str = "journal") -> dict:
    spec = DIMS[dim]
    dictionary, column = spec["dict"], spec["column"]

    info = query(f"""
        SELECT id, node_id, name, node_type, parent_id,
               arrayReverse(dictGetHierarchy('{dictionary}', toUInt64(id))) AS chain
        FROM {spec['table']} WHERE id = {node}""")
    if not info:
        return {"error": f"unknown node {node}"}
    current = info[0]

    crumb_ids = [int(x) for x in current["chain"]]
    crumbs = query(
        f"SELECT id, name FROM {spec['table']} "
        f"WHERE id IN ({','.join(str(i) for i in crumb_ids) or '0'})") if crumb_ids else []
    names = {int(r["id"]): r["name"] for r in crumbs}
    breadcrumb = [{"id": i, "name": names.get(i, str(i))} for i in crumb_ids]

    if dim == "product" and current["node_type"] in ("family", "model"):
        return _subledger_drill(dim, node, current, breadcrumb, spec)

    # The ~NA~ bucket (dimension does not apply) belongs at the ROOT, where the statement
    # must tie to the group total. Carried deeper it would show all group revenue inside
    # a single department.
    at_root = node == root_id(dim)
    na_clause = ("""
                UNION ALL
                SELECT toUInt64(0) AS kid, toUInt64(0) AS nid""" if at_root else "")

    started = time.time()
    table = SOURCES[source]["table"]
    rows = query(_measure_sql(view, column, node, dictionary, na_clause, table))
    elapsed_ms = (time.time() - started) * 1000

    lines: dict = {}
    for row in rows:
        key = row["label"] if view != "mgmt" else int(row["sort_order"])
        entry = lines.setdefault(key, {
            "sort_order": int(row["sort_order"]), "label": row["label"],
            "is_subtotal": bool(int(row["is_subtotal"])), "values": {}, "total": 0.0,
        })
        entry["values"][str(int(row["child"]))] = float(row["value"])
        entry["total"] += float(row["value"])

    if view == "mgmt":
        ordered = [lines[k] for k in sorted(lines)]
    else:
        # These bases have no fixed line order — rank by magnitude and cap the list.
        ordered = sorted(lines.values(), key=lambda e: -abs(e["total"]))[:20]
        for index, entry in enumerate(ordered):
            entry["sort_order"] = index

    child_ids = sorted({int(r["child"]) for r in rows if int(r["child"]) != 0})
    has_na = any(int(r["child"]) == 0 for r in rows)
    meta = query(
        f"SELECT id, name, node_type, dictGetChildren('{dictionary}', toUInt64(id)) != [] "
        f"AS has_children FROM {spec['table']} "
        f"WHERE id IN ({','.join(str(i) for i in child_ids) or '0'}) ORDER BY name"
    ) if child_ids else []

    children = ([{"id": 0, "name": "~NA~ not applicable", "node_type": "structural",
                  "has_children": False}] if has_na else []) + [
        {"id": int(c["id"]), "name": c["name"], "node_type": c["node_type"],
         "has_children": bool(int(c["has_children"]))} for c in meta]

    return {
        "dim": dim, "source": "ledger", "view": view, "views": VIEWS,
        "fact_source": source, "sources": SOURCES, "table": table,
        "node": {"id": int(current["id"]), "name": current["name"],
                 "node_type": current["node_type"]},
        "breadcrumb": breadcrumb,
        "children": children,
        "lines": ordered,
        "elapsed_ms": round(elapsed_ms, 1),
    }


def stats(source: str = "journal") -> dict:
    table = SOURCES[source]["table"]
    short = table.split(".")[-1]
    counts = query(f"SELECT count() AS rows FROM {table}")
    size = query(f"""
        SELECT formatReadableSize(sum(bytes_on_disk)) AS on_disk
        FROM system.parts WHERE database = 'volteo' AND table = '{short}' AND active""")
    out = dict(counts[0]) if counts else {}
    out.update(size[0] if size else {})
    out["table"] = short
    return out


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, payload: bytes, content_type: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        try:
            if parsed.path in ("/", "/index.html"):
                self._send((HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif parsed.path == "/api/stats":
                source = params.get("source", ["journal"])[0]
                self._send(json.dumps(stats(source)).encode(), "application/json")
            elif parsed.path == "/api/drill":
                dim = params.get("dim", ["cc"])[0]
                view = params.get("view", ["mgmt"])[0]
                if dim not in DIMS:
                    raise ValueError(f"unknown dimension {dim}")
                if view not in VIEWS:
                    raise ValueError(f"unknown view {view}")
                source = params.get("source", ["journal"])[0]
                if source not in SOURCES:
                    raise ValueError(f"unknown source {source}")
                node = int(params.get("node", [0])[0]) or root_id(dim)
                self._send(json.dumps(drill(dim, node, view, source)).encode(),
                           "application/json")
            else:
                self._send(b"not found", "text/plain", 404)
        except Exception as exc:
            self._send(json.dumps({"error": str(exc)}).encode(), "application/json", 500)


if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("", PORT), Handler) as httpd:
        print(f"drill-down app on http://localhost:{PORT}")
        httpd.serve_forever()
