"""Management P&L pack: actual vs budget / forecast / prior year, by entity and period.

Reads the generator's parquet output and serves a single-page app. No database needed:

    volteogen --set period.months=96          # writes out/
    python app/pnl/serve.py                   # http://127.0.0.1:8812
    python app/pnl/serve.py --export app/pnl/pnl.json   # static copy of the data

All statement logic lives in the dataset (rpt_income_statement is already built from
fact_pnl through the report bridge), so this file only reshapes rows into a compact,
column-oriented JSON payload the page can slice in the browser.
"""
from __future__ import annotations

import argparse
import http.server
import json
import socketserver
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
REPORT_ID = "IS_MGMT"


def build_payload(parquet_dir: Path) -> dict:
    rpt = pl.read_parquet(parquet_dir / "rpt_income_statement.parquet").filter(
        pl.col("report_id") == REPORT_ID
    )
    lines = (
        pl.read_parquet(parquet_dir / "dim_report_line.parquet")
        .filter(pl.col("report_id") == REPORT_ID)
        .sort("sort_order")
    )
    entities = pl.read_parquet(parquet_dir / "dim_entity.parquet")
    scenarios = pl.read_parquet(parquet_dir / "dim_scenario.parquet")
    fx = pl.read_parquet(parquet_dir / "dim_fx_rate.parquet")
    calendar = pl.read_parquet(parquet_dir / "dim_calendar.parquet").sort("month_index")

    line_ids = lines["line_id"].to_list()
    entity_ids = sorted(rpt["entity"].unique().to_list())
    scenario_ids = sorted(rpt["scenario_key"].unique().to_list())
    line_ix = {v: i for i, v in enumerate(line_ids)}
    entity_ix = {v: i for i, v in enumerate(entity_ids)}
    scenario_ix = {v: i for i, v in enumerate(scenario_ids)}

    rpt = rpt.filter(pl.col("line_id").is_in(line_ids)).sort(
        ["scenario_key", "entity", "fiscal_year", "period_no", "sort_order"]
    )

    closing = (
        fx.filter(pl.col("from_currency") != pl.col("to_currency"))
        .group_by("from_currency", "fiscal_year")
        .agg(pl.col("rate_budget").first())
    )

    return {
        "report": {
            "id": REPORT_ID,
            "name": str(lines["report_name"][0]),
        },
        "group_currency": str(fx["to_currency"][0]),
        "lines": [
            {"id": r["line_id"], "label": r["label"], "indent": r["indent"], "subtotal": r["is_subtotal"]}
            for r in lines.to_dicts()
        ],
        "entities": [
            {
                "id": r["entity"],
                "name": r["name"],
                "currency": r["currency"],
                "consolidated": bool(r["is_consolidated"]),
                "nci_pct": float(r.get("nci_pct") or 0.0),
                "notes": r.get("notes") or "",
            }
            for r in entities.to_dicts()
            if r["entity"] in entity_ix
        ],
        "entity_ids": entity_ids,
        "scenarios": [
            {
                "key": r["scenario_key"],
                "type": r["scenario_type"],
                "version": r["version"],
                "fiscal_year": r["fiscal_year"],
                "closed_months": r["closed_months"],
                "description": r["description"],
            }
            for r in scenarios.to_dicts()
            if r["scenario_key"] in scenario_ix
        ],
        "scenario_ids": scenario_ids,
        "periods": [
            {"fy": r["fiscal_year"], "p": r["period_no"], "key": r["period_key"], "label": r["period_label"]}
            for r in calendar.to_dicts()
        ],
        "budget_rates": closing.sort("from_currency", "fiscal_year").to_dicts(),
        # Column-oriented rows: s, e, fy, p, l index into the lists above.
        "rows": {
            "s": [scenario_ix[v] for v in rpt["scenario_key"].to_list()],
            "e": [entity_ix[v] for v in rpt["entity"].to_list()],
            "fy": rpt["fiscal_year"].to_list(),
            "p": rpt["period_no"].to_list(),
            "l": [line_ix[v] for v in rpt["line_id"].to_list()],
            "gc": rpt["amount_gc"].round(0).cast(pl.Int64).to_list(),
            "cc": rpt["amount_gc_cc"].round(0).cast(pl.Int64).to_list(),
            "lc": rpt["amount_lc"].round(0).cast(pl.Int64).to_list(),
        },
    }


def serve(parquet_dir: Path, port: int) -> None:
    payload = json.dumps(build_payload(parquet_dir), separators=(",", ":")).encode()
    # The page is authored as a fragment (it is also published as a hosted artifact,
    # which supplies the document skeleton), so add the doctype here.
    index = b'<!doctype html><html lang="en"><meta charset="utf-8">' + (HERE / "index.html").read_bytes()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - stdlib naming
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                body, kind = index, "text/html; charset=utf-8"
            elif path == "/pnl.json":
                body, kind = payload, "application/json"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    with socketserver.ThreadingTCPServer(("127.0.0.1", port), Handler) as httpd:
        print(f"P&L pack on http://127.0.0.1:{port}  (data: {parquet_dir})")
        httpd.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=str(REPO / "out" / "parquet"), help="parquet directory")
    parser.add_argument("--port", type=int, default=8812)
    parser.add_argument("--export", metavar="PATH", help="write the JSON payload and exit")
    args = parser.parse_args()

    parquet_dir = Path(args.data)
    if not (parquet_dir / "rpt_income_statement.parquet").exists():
        raise SystemExit(
            f"{parquet_dir} has no rpt_income_statement.parquet - run `volteogen` first"
        )
    if args.export:
        Path(args.export).write_text(json.dumps(build_payload(parquet_dir), separators=(",", ":")))
        print(f"wrote {args.export}")
        return
    serve(parquet_dir, args.port)


if __name__ == "__main__":
    main()
