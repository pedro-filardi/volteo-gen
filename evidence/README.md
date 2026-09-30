# Volteo financial statements — Evidence on ClickHouse

Statement-grade reporting over the generated dataset: indented lines, ruled subtotals, a
double rule at net income, negatives in parentheses. Evidence was chosen over Metabase or
Superset precisely because those render a pivot table, not a statement.

**No statement logic lives in this project.** Line order, subtotal composition and the
sign convention come from `config/reports/income_statement.yaml`, compiled into
`dim_report_line` + `bridge_report_line` and served by the `v_income_statement` view.
Adding a P&L line is a YAML edit followed by a rebuild — this page never changes.

## Run it

```bash
# 1. Generate the dataset (from volteo-gen/)
PYTHONPATH=src python3 -m volteogen.cli --preset S

# 2. Start ClickHouse. Either the native binary...
clickhouse server --config-file=/path/to/config.xml
#    ...or Docker: cd ../deploy && docker compose up -d

# 3. Load schema, data and views
cd ../out/clickhouse
./load.sh ../parquet --host 127.0.0.1

# 4. Pull from ClickHouse and serve
cd ../../evidence
npm install
npm run sources     # executes sources/volteo/*.sql against ClickHouse
npm run dev         # http://localhost:3000
```

`npm run sources` is the only step that touches ClickHouse. It materialises each query to
Parquet under `build/data/`, and the browser then queries those with DuckDB-WASM. That is
why the site is a static build and why the page is fast regardless of `fact_gl` size —
but also why **you must re-run `npm run sources` after regenerating the dataset**.

## The ClickHouse connector

Evidence ships **no first-party ClickHouse connector**. This project uses the community
plugin `evidence-connector-clickhouse` (v0.0.2, MIT). It works, with two sharp edges worth
knowing:

1. **No `database` option** — every query in `sources/volteo/*.sql` fully qualifies
   `volteo.<table>`.
2. **Types are inferred from the first row**, and ClickHouse returns `Int64` as a *quoted
   string* in `JSONEachRow`. Left alone, every integer arrives as text and arithmetic in
   the page silently breaks. Every numeric column is therefore cast explicitly:

   ```sql
   toInt32(sort_order)  as sort_order,
   toFloat64(value)     as value,
   toUInt8(is_subtotal) as is_subtotal   -- Bool would infer as a string too
   ```

If you would rather not depend on a v0.0.x plugin, the alternative is to point the
first-party `@evidence-dev/duckdb` source at the generated `out/duckdb/volteo.duckdb` (or
at ClickHouse-exported Parquet). Same queries, same page — only `connection.yaml` changes.

## Layout

```
sources/volteo/
  connection.yaml        ClickHouse HTTP endpoint
  income_statement.sql   the report-line join — the query that drives the statement
  margins.sql            calibration targets vs realised
  monthly_result.sql     monthly result by entity
  data_quality.sql       injected defects, surfaced
  entities.sql, nci.sql
pages/
  index.md               the statement, charts and notes
```

The statement table is hand-rolled with a Svelte `{#each}` rather than `<DataTable>`,
because Evidence's table component cannot express indentation, subtotal rules or
accounting parentheses. Formatting is done inline with `toLocaleString` so the page needs
no `<script>` block.

## Verified

Built against ClickHouse 26.6 with the S preset. FY2025 consolidated figures in the
shipped Parquet: net revenue 276,375k, gross profit 91,757k, EBITDA 17,383k, net income
10,213k — identical to the generator's own reference implementation and to the SQL in
`out/clickhouse/example_queries.sql`.

Not verified: how the page looks in a browser. The build and the data were checked; the
visual rendering was not.
