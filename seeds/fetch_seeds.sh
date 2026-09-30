#!/usr/bin/env bash
# Re-download the real seed skeletons. See SOURCES.md for provenance/licensing.
set -euo pipefail
cd "$(dirname "$0")"

AL="https://raw.githubusercontent.com/microsoft/ALAppExtensions/main/Apps"
W1="$AL/W1/ContosoCoffeeDemoDataset/app/DemoData"
ODOO="https://raw.githubusercontent.com/odoo/odoo/17.0/addons"
UK="https://raw.githubusercontent.com/billkhiz-bit/uk-chart-of-accounts/master"
GAAP="https://raw.githubusercontent.com/hfreeman-factualiq/us-gaap-2026-explorer/master"

get() { echo "  -> $2"; curl -sSLf --max-time 300 -o "$2" "$1"; }

echo "US ERP CoA (Business Central, MIT)"
get "$AL/US/ContosoCoffeeDemoDatasetUS/app/DemoData/Finance/1.Setup%20Data/CreateUSGLAccounts.Codeunit.al" us_gl_us.al
get "$W1/Finance/1.Setup%20data/CreateGLAccount.Codeunit.al"            us_gl_finance.al
get "$W1/Common/1.Setup%20Data/CreateCommonGLAccount.Codeunit.al"       us_gl_common.al
get "$W1/FixedAsset/1.Setup%20Data/CreateFAGLAccount.Codeunit.al"       us_gl_fa.al
get "$W1/HumanResources/1.%20SetupData/CreateHRGLAccount.Codeunit.al"   us_gl_hr.al
get "$W1/Jobs/1.Setup%20Data/CreateJobGLAccount.Codeunit.al"            us_gl_job.al
get "$W1/Manufacturing/1.Setup%20data/CreateMfgGLAccount.Codeunit.al"   us_gl_mfg.al
get "$W1/Service/1.Setup%20Data/CreateSvcGLAccount.Codeunit.al"         us_gl_svc.al

echo "SKR03 / PGC (Odoo, LGPL-3.0)"
get "$ODOO/l10n_de/data/template/account.account-de_skr03.csv" skr.csv
get "$ODOO/l10n_es/data/template/account.account-es_common.csv" pgc_odoo.csv

echo "UK nominals (MIT)"
get "$UK/src/uk_coa/data.py" uk_coa_data.py

echo "US-GAAP 2026 taxonomy (FASB via explorer)"
get "$GAAP/data/taxonomy-data.json" usgaap2026_taxonomy.json

echo
echo "Verifying against pinned checksums:"
shasum -a 256 -c CHECKSUMS.txt || echo "  (drift vs pinned digests — review before committing)"
