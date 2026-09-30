-- ===========================================================================
-- Runtime report VIEWS, all resolved through dictionaries.
--
-- The statement layout is already data (dim_report_line / bridge_report_line).
-- These add the other two reporting bases so a viewer can switch at runtime:
--
--   MGMT     management P&L by nature      account_class -> report line
--   US_GAAP  FASB StatementOfIncome        account_class -> concept -> signed tree
--   FUNCTION by-nature -> by-function      (account_class, cc function) -> mgmt line
--
-- Every hop is a dictGet, so no mapping logic lives in the application.
-- ===========================================================================

-- Cost-centre function must be reachable from the fact table's cc_id, so it joins
-- the hierarchy table and the dictionary that fronts it.
DROP TABLE IF EXISTS volteo.hier_cc_fn SYNC;
CREATE TABLE volteo.hier_cc_fn ENGINE = MergeTree ORDER BY id AS
SELECT h.id AS id, h.node_id AS node_id, h.name AS name,
       ifNull(d.function, '~NA~') AS function
FROM volteo.hier_cc AS h
LEFT JOIN volteo.dim_cost_center_node AS d ON d.node_id = h.node_id;

DROP DICTIONARY IF EXISTS volteo.dict_cc_function SYNC;
CREATE DICTIONARY volteo.dict_cc_function
(id UInt64, function String DEFAULT '~NA~', name String)
PRIMARY KEY id SOURCE(CLICKHOUSE(DB 'volteo' TABLE 'hier_cc_fn'))
LAYOUT(HASHED()) LIFETIME(MIN 0 MAX 0);

-- account_class -> US-GAAP concept. Verified 1:1 (each class resolves to exactly one
-- concept), so a plain lookup dictionary is sufficient.
DROP TABLE IF EXISTS volteo.map_class_gaap SYNC;
CREATE TABLE volteo.map_class_gaap ENGINE = MergeTree ORDER BY account_class AS
SELECT g.account_class AS account_class, any(gg.gaap_concept) AS gaap_concept
FROM volteo.dim_group_account AS g
INNER JOIN volteo.map_group_to_gaap AS gg ON gg.group_account = g.code
GROUP BY g.account_class;

DROP DICTIONARY IF EXISTS volteo.dict_class_gaap SYNC;
CREATE DICTIONARY volteo.dict_class_gaap
(account_class String, gaap_concept String DEFAULT '')
PRIMARY KEY account_class SOURCE(CLICKHOUSE(DB 'volteo' TABLE 'map_class_gaap'))
LAYOUT(COMPLEX_KEY_HASHED()) LIFETIME(MIN 0 MAX 0);

-- (account_class, cost-centre function) -> management line. THE by-nature to
-- by-function pivot: the same salary account becomes "COGS - labour", "R&D" or "G&A"
-- depending only on the cost centre it was posted to.
DROP TABLE IF EXISTS volteo.map_class_function SYNC;
CREATE TABLE volteo.map_class_function
ENGINE = MergeTree ORDER BY (account_class, cc_function) AS
SELECT g.account_class AS account_class,
       m.cc_function   AS cc_function,
       any(m.management_line) AS management_line
FROM volteo.map_management AS m
INNER JOIN volteo.dim_group_account AS g ON g.code = m.group_account
WHERE m.management_line != '' AND m.is_excluded = false
GROUP BY g.account_class, m.cc_function;

DROP DICTIONARY IF EXISTS volteo.dict_class_function SYNC;
CREATE DICTIONARY volteo.dict_class_function
(account_class String, cc_function String, management_line String DEFAULT 'Other')
PRIMARY KEY account_class, cc_function
SOURCE(CLICKHOUSE(DB 'volteo' TABLE 'map_class_function'))
LAYOUT(COMPLEX_KEY_HASHED()) LIFETIME(MIN 0 MAX 0);

-- Display order for the GAAP view: depth from the statement root, so concepts print
-- in tree order rather than alphabetically.
DROP TABLE IF EXISTS volteo.gaap_order SYNC;
CREATE TABLE volteo.gaap_order ENGINE = MergeTree ORDER BY concept AS
SELECT node AS concept, min(depth_diff) AS depth, count() AS reachable
FROM volteo.bridge_usgaap
GROUP BY node;
