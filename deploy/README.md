# Serving the dataset from ClickHouse in an open-source BI tool

The goal is that **no statement logic lives in the dashboard**. Line order, subtotal
composition, sign convention and hierarchy roll-ups are all data; the BI tool only picks
rows, columns and filters.

## The layering

```
config/reports/*.yaml          report layout, authored once
   |  volteogen build
   v
dim_report_line                label, sort_order, indent, is_subtotal
bridge_report_line             (line -> account_class) with sign pre-multiplied
   |  load.sh
   v
ClickHouse: fact_gl + dims + bridges
   |  views.sql
   v
v_income_statement             one row per (line x entity x period)
   |
   v
Metabase / Superset / Evidence — a pivot table, no custom SQL
```

Adding a P&L line, reordering the statement, or building a second layout (statutory HGB,
by-function) is a YAML edit plus a rebuild. Nothing downstream changes.

## Why a bridge instead of CASE WHEN

The naive approach puts the statement in the query:

```sql
-- DON'T: every new line is a code change, and subtotals get hand-maintained
SELECT sum(CASE WHEN account_class = 'revenue_trade' THEN -amount END) AS revenue,
       sum(CASE WHEN account_class IN ('cogs_standard','cogs_variance') THEN -amount END) AS cogs,
       ...
```

A subtotal in the bridge maps to **every** `account_class` beneath it, with the sign
already multiplied in. So `Gross profit` is not computed — it is a `GROUP BY` over four
mapped classes. This is the same mechanism as `bridge_usgaap`, where 61 of the 531 FASB
calculation arcs carry weight −1 and a plain `SUM(amount)` is silently wrong.

## Run it

```bash
cd deploy
mkdir -p plugins && curl -L -o plugins/clickhouse.metabase-driver.jar \
  https://github.com/ClickHouse/metabase-clickhouse-driver/releases/latest/download/clickhouse.metabase-driver.jar
docker compose up -d

# load: schema -> parquet -> views
cd ../out/clickhouse
./load.sh ../parquet --host 127.0.0.1 --user volteo --password volteo
```

Then in Metabase: add the ClickHouse database (`clickhouse:8123`, db `volteo`), and build
the income statement as a plain question against `v_income_statement`:

| Metabase setting | Value |
|---|---|
| Table | `v_income_statement` |
| Filter | `report_id = IS_MGMT`, `fiscal_year = 2025` |
| Summarize | `Sum of value` |
| Group by | Rows `label`, Columns `entity` |
| Sort | `sort_order` |

That is the whole configuration. Consolidation is included automatically because
eliminations and topside journals are rows in `fact_gl` under the `ELIM` and `GRP`
entities — the group total is just every entity summed.

## Choosing the tool

| Tool | Use it when | Watch out for |
|---|---|---|
| **Metabase** | fastest path; non-technical users self-serve | driver is a manual jar drop; statement formatting (indent, parentheses) is limited |
| **Superset** | you want a curated, versioned dashboard set and row-level security | steeper setup; pivot formatting still basic |
| **Evidence** | the statement must *look* like a statement — indent, ruled subtotals, negatives in parentheses; reports as code in git | it is a static site generator, not ad-hoc exploration |
| **Rill** | fast interactive exploration directly on ClickHouse | dashboards are metric-centric, less suited to statement layout |
| **Cube** | several consumers (BI + API + spreadsheet) must share one metric definition | another service to run; overkill for one dashboard |

For financial statements specifically the honest split is: **Metabase** if people need to
explore, **Evidence** if the output must read like a real statement. The views work
unchanged for both.

## What is deliberately not solved here

- **Indentation and subtotal styling.** `dim_report_line.indent` and `is_subtotal` are
  exposed, but Metabase and Superset will not render a ruled, indented statement. If that
  fidelity matters, Evidence (or a small custom front end) is the answer.
- **Balance sheet and cash flow.** Only the income statement layout ships; the generator
  models balance-sheet accounts as plausible balances, not a full articulation.
- **Budget vs actual.** `scenario_key` is in the view and the grain is right, but the
  generator does not yet produce budget or forecast facts, so today every row is `ACT`.
