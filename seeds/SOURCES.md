# Seed sources (real-world skeletons)

These files are **real** chart-of-accounts / taxonomy skeletons pulled from public,
permissively-licensed sources. The generator loads them and never invents their
structure. If a file is missing, `volteogen.seedload` fails loudly with the URL below
(see `seeds/fetch_seeds.sh`); it never silently substitutes fake structure.

| file | content | upstream | license |
|---|---|---|---|
| `us_gl_us.al` | US ERP CoA: 346 `InsertGLAccount` calls with US nominal codes, Begin/End-Total ranges, categories, indentation | `microsoft/ALAppExtensions` @ `main` — `Apps/US/ContosoCoffeeDemoDatasetUS/app/DemoData/Finance/1.Setup Data/CreateUSGLAccounts.Codeunit.al` | MIT |
| `us_gl_finance.al`, `us_gl_common.al`, `us_gl_fa.al`, `us_gl_hr.al`, `us_gl_job.al`, `us_gl_mfg.al`, `us_gl_svc.al` | W1 (industry-neutral) account **name labels** referenced by the US codeunit | `microsoft/ALAppExtensions` @ `main` — `Apps/W1/ContosoCoffeeDemoDataset/app/DemoData/**/Create*GLAccount*.Codeunit.al` | MIT |
| `skr.csv` | SKR03 German CoA, 1,274 accounts, German + English names, `tag_ids` = real HGB statutory tags | `odoo/odoo` @ `17.0` — `addons/l10n_de/data/template/account.account-de_skr03.csv` | LGPL-3.0 |
| `pgc_odoo.csv` | Spanish PGC CoA, ES/CA/EN names | `odoo/odoo` @ `17.0` — `addons/l10n_es/data/template/account.account-es_common.csv` | LGPL-3.0 |
| `uk_coa_data.py` | 166 UK nominal codes + category layer + VAT treatment + HMRC box references (Sage-standard numbering) | `billkhiz-bit/uk-chart-of-accounts` @ `master` — `src/uk_coa/data.py` | MIT |
| `usgaap2026_taxonomy.json` | FASB US-GAAP 2026 taxonomy graph. Network `StatementOfIncome` carries **531 calculation arcs** with ±1 weights | `hfreeman-factualiq/us-gaap-2026-explorer` @ `master` — `data/taxonomy-data.json` | see upstream repo |

## Why the US CoA needs two files

The Business Central demo data splits the US chart across two layers, and the
generator joins them on the **AL procedure name**:

- `Apps/W1/.../Create*GLAccount*.Codeunit.al` declares `<Base>Lbl: Label '<text>'`
  — the industry-neutral account *names*.
- `Apps/US/.../CreateUSGLAccounts.Codeunit.al` declares the US *structure*
  (`InsertGLAccount(<Base>(), <Base>Name(), <income/balance>, <category>,
  <subcategory>, <account type>, …, <indentation>, <totaling range>, …)`) and binds
  names to US nominal codes via `AddAccountForLocalization(<Base>Name(), '<code>')`.

Account numbers are never hardcoded in the generator: they come from
`AddAccountForLocalization` in the upstream MIT source.

## Refresh

```bash
bash seeds/fetch_seeds.sh
```

Checksums of the pinned copies are in `seeds/CHECKSUMS.txt`; `volteogen.seedload`
warns (not fails) when a local file drifts from the pinned digest, so an intentional
upstream refresh is easy but an accidental edit is visible.
