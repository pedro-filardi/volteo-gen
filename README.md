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

- `fact_gl` — actuals at trial-balance grain (entity, ledger, account, period, dims, LC and group-currency amounts)
- `fact_budget`, `fact_forecast` — BUD_V1/BUD_V2 and FC3+9/FC6+6/FC9+3 per fiscal year
- `dim_report_line` + `bridge_report_line` — income statement layout (Net revenue → Net income)
- `dim_scenario`, `dim_calendar`, `dim_entity`, other `dim_*` hierarchies

## Tests

```bash
pytest
```

## Other folders

- `seeds/` — pinned real-world CoA/taxonomy skeletons (see `seeds/SOURCES.md`, refresh with `seeds/fetch_seeds.sh`)
- `clickhouse/`, `deploy/` — ClickHouse views and docker-compose for serving the dataset
- `evidence/` — Evidence.dev dashboards (`npm install` inside that folder)
- `app/` — small local explorer
