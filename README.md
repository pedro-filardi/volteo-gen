# volteogen

Seeded, config-driven generator for a synthetic multi-entity enterprise finance dataset
(Volteo Electronics Group): GL actuals, budget and forecast scenarios, multiple local
charts of accounts (US ERP, SKR03, PGC, UK/Sage), intercompany, consolidation, FX and
injected data-quality defects.

Generated data is not committed — it is fully reproducible from the seed and config.

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . pytest
```

## Generate

```bash
volteogen                                   # seed 42, preset M, 36 months -> out/
volteogen --set period.months=96            # 8 fiscal years (FY2024-FY2031)
volteogen --preset S --seed 7 --out out_s   # smaller, different seed
volteogen --formats parquet,csv             # choose export formats
```

Presets live in `config/presets/` (S, M, L, XL); any config path can be overridden with
`--set path=value`. The output folder includes `README_dataset.md` describing every
table, the validation results and the injected defects.

Main tables for P&L work:

- `fact_pnl` — the P&L cube: actuals, budgets and forecasts on one grain
  (scenario × entity × fiscal period × group account × product family), in local
  currency plus two group-currency measures:
  - `amount_gc` (reported): actuals at monthly average rates, budget at the budget rate,
    forecast at actual rates for closed months and the budget rate for open months
  - `amount_gc_cc` (constant currency): everything at the fiscal year's budget rate
  - plans carry their own intercompany eliminations (ELIM), and non-controlling
    interests are booked on GRP, so summing all entities gives the consolidated group
- `rpt_income_statement` — `fact_pnl` rolled up to income statement lines for every
  entity/scenario/period, in presentation sign (revenue positive, costs negative)
- `fact_gl` — actuals at trial-balance grain (the books, local charts of accounts)
- `fact_budget`, `fact_forecast` — BUD_V1/BUD_V2 and FC3+9/FC6+6/FC9+3 per fiscal year
  (planning submissions in local currency; forecasts re-phase the board budget by
  year-to-date actual/budget)
- `dim_report_line` + `bridge_report_line` — income statement layout, defined in
  `config/reports/income_statement.yaml`
- `dim_scenario`, `dim_calendar` (July–June fiscal year, P13 close period), `dim_entity`,
  `dim_fx_rate`, other `dim_*` hierarchies

## P&L app

```bash
volteogen --set period.months=96
python app/pnl/serve.py                 # http://127.0.0.1:8812
```

Actual vs budget, forecast or prior year for the group or any entity; month, QTD, YTD
or full year; reported, constant-currency or local-currency figures; monthly trend and
forecast convergence.

## Tests

```bash
pytest
```

## Other folders

- `seeds/` — pinned real-world CoA/taxonomy skeletons (see `seeds/SOURCES.md`, refresh with `seeds/fetch_seeds.sh`)
- `clickhouse/`, `deploy/` — ClickHouse views and docker-compose for serving the dataset
- `evidence/` — Evidence.dev dashboards (`npm install` inside that folder)
- `app/pnl/` — management P&L app (above)
- `app/server.py` — drill-down income statement over ClickHouse
