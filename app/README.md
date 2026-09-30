# Drill-down income statement over 400M rows

An income statement you can navigate up and down any hierarchy — cost centre, product or
market — served from ClickHouse in ~20 ms regardless of table size.

```bash
clickhouse/start.sh                          # ClickHouse on :8123
python app/server.py                         # app on :8811
```

## Where the 400M rows come from

Every row is **real generator output**. Nothing is fabricated with `rand()`.

A single generator run tops out near 1.4M GL rows — `fact_gl` is trial-balance grain, so
its size is bounded by families × markets × customers × months, and inflating that with
fake dimensions is what spec §6 forbids. 400M rows in one polars frame would also need
far more RAM than the machine has.

So the table is an **ensemble**: ~300 independent runs, each with its own seed, each
individually calibrated to its margin narrative and each individually reconciling. They
are unioned under a `run_id` column, built by `clickhouse/build_400m.py` at ~290k
rows/sec. Any single `run_id` still ties out on its own; the union is a benchmark-scale
table of genuine generated data.

## The three things that make it fast

**1. Hierarchical dictionaries need integer keys.** `dictGetChildren` /
`dictGetDescendants` / `dictGetHierarchy` require a **UInt64** key and a `HASHED` layout —
`COMPLEX_KEY_HASHED` supports lookups but no traversal at all. The business keys here are
strings (`CC:US01:SLS:FIELDSALES`), so `clickhouse/drilldown_setup.sql` derives a
surrogate id per tree, with root = `parent_id 0`.

**2. Never `dictIsIn` in a WHERE clause.** Measured on 200M rows
(`clickhouse/HIERARCHY_BENCHMARK.md`):

| predicate | rows read | marks |
|---|---:|---:|
| `dictIsIn(dict, key, ancestor)` | 200,000,000 | 24,419 |
| `key IN (dictGetDescendants(…))` | 17,498,112 | 2,136 |

`dictIsIn` is an opaque per-row function, so the primary-key index cannot be used. An
`IN` against a literal set can. The dictionary's job here is to **expand** an ancestor
into its descendant set; the set then does the filtering.

**3. Aggregate before the bridge fans out — then let a projection do even that.** The
report bridge turns each fact row into ~4 rows (one per subtotal it belongs to), so
joining first fans out tens of millions of rows. Aggregating to
`(child × account_class)` first was 8× faster in the benchmark.

Better still, an aggregate projection over the drillable dimensions collapses the scan:

```sql
ALTER TABLE volteo.fact_gl_400m ADD PROJECTION p_dims
(SELECT account_class, account_id, entity, fiscal_year, period_no,
        cc_id, product_id, market_id, sum(amount)
 GROUP BY account_class, account_id, entity, fiscal_year, period_no,
          cc_id, product_id, market_id);
```

**The grain matters for realism.** An earlier version keyed only on
`(account_class, cc_id, product_id, market_id)` and had just **1,540** distinct
combinations across the whole table — because the applicability matrix means each
account class touches only the dimensions that apply to it (revenue has no cost centre,
payroll has no product). That made the projection unrealistically effective. Adding
**account, entity and period** to the grain takes it to **~140k+ combinations**, which
is a believable pre-aggregate ratio rather than a flattering one.

`EXPLAIN indexes=1` then shows `ReadFromMergeTree (p_dims)` reading **12 granules**.

The catch worth knowing: a `WHERE … IN (subquery)` **stops the optimizer using the
projection**. So the app's aggregate CTE is deliberately *unfiltered* — the projection
returns ~1,500 rows instantly and the descendant filter is applied to that tiny result
instead of to the fact table. That single change took the drill query from **1,050 ms to
20 ms**.

## Measured

| step | time |
|---|---:|
| join-first, filtered scan | 3,200 ms |
| pre-aggregate before bridge | 1,050 ms |
| **unfiltered aggregate + projection** | **20–32 ms** |

Flat as the table grows, because the projection is what gets read.

## Full hierarchy, and where it changes source

The app walks the **whole** dictionary — it never caps depth. But the data does not sit
at every level, and that is deliberate:

| dimension | hierarchy | fact_gl posts at | drills to |
|---|---|---|---|
| Cost centre | root → entity → function → department → team | department + team (leaves) | full depth |
| Market | root → region → country → subterritory | country + subterritory (leaves) | full depth |
| Product | root → division → category → family → model → **sku** | **family only** | full depth, via drill-through |

Spec §5 puts product on the GL at *family* grain and keeps SKU and customer detail in
the sub-ledger. So the model and SKU levels are genuinely empty in the ledger. Rather
than show zeros, the app **switches source table at the family boundary**: below family
it reads `volteo.sub_keyed` (the keyed sub-ledger) and the UI says so.

Only revenue, COGS and gross profit exist below family — the sub-ledger has no payroll or
overhead, because those never had a product in the first place.

## The `~NA~` column

At the root you get a `~NA~ not applicable` column. That is not a data gap — it is the
applicability matrix showing through: revenue has no cost centre, so it cannot be
attributed to one. Keeping it visible is what lets the root statement reconcile to the
group total (the app asserts children sum to the total).

Drill one level down and the bucket disappears, because carrying all group revenue into a
single department would make a G&A team look like it earns the group's income.

## Layout

```
app/server.py     stdlib HTTP server; /api/drill?dim=cc|product|market&node=<id>
app/index.html    the UI — click a column heading to drill, breadcrumb to go up
```

Not verified: how the page renders in a browser. The API, the timings and the
reconciliation were checked; the visual layout was not.
