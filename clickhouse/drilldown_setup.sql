-- ===========================================================================
-- Surrogate integer keys + hierarchical dictionaries for the drill-down app.
--
-- ClickHouse hierarchy functions (dictGetChildren / dictGetDescendants /
-- dictGetHierarchy) require a UInt64 key and a HASHED/FLAT layout. The business
-- keys here are strings, so each tree gets a surrogate id. Root = parent_id 0.
-- ===========================================================================

DROP TABLE IF EXISTS volteo.hier_product SYNC;
CREATE TABLE volteo.hier_product ENGINE = MergeTree ORDER BY id AS
WITH numbered AS (
    SELECT rowNumberInAllBlocks() + 1 AS id, node_id, parent_id AS pnode, name, node_type, is_leaf
    FROM (SELECT node_id, parent_id, name, node_type, is_leaf
          FROM volteo.dim_product_node ORDER BY node_id)
)
SELECT n.id AS id, p.id AS parent_id, n.node_id AS node_id, n.name AS name,
       n.node_type AS node_type, n.is_leaf AS is_leaf
FROM numbered AS n LEFT JOIN numbered AS p ON p.node_id = n.pnode
SETTINGS max_threads = 1;

DROP TABLE IF EXISTS volteo.hier_market SYNC;
CREATE TABLE volteo.hier_market ENGINE = MergeTree ORDER BY id AS
WITH numbered AS (
    SELECT rowNumberInAllBlocks() + 1 AS id, node_id, parent_id AS pnode, name, node_type, is_leaf
    FROM (SELECT node_id, parent_id, name, node_type, is_leaf
          FROM volteo.dim_market_node ORDER BY node_id)
)
SELECT n.id AS id, p.id AS parent_id, n.node_id AS node_id, n.name AS name,
       n.node_type AS node_type, n.is_leaf AS is_leaf
FROM numbered AS n LEFT JOIN numbered AS p ON p.node_id = n.pnode
SETTINGS max_threads = 1;

DROP TABLE IF EXISTS volteo.hier_cc SYNC;
CREATE TABLE volteo.hier_cc ENGINE = MergeTree ORDER BY id AS
WITH numbered AS (
    SELECT rowNumberInAllBlocks() + 1 AS id, node_id, parent_id AS pnode, name, node_type, is_leaf
    FROM (SELECT node_id, parent_id, name, node_type, is_leaf
          FROM volteo.dim_cost_center_node ORDER BY node_id)
)
SELECT n.id AS id, p.id AS parent_id, n.node_id AS node_id, n.name AS name,
       n.node_type AS node_type, n.is_leaf AS is_leaf
FROM numbered AS n LEFT JOIN numbered AS p ON p.node_id = n.pnode
SETTINGS max_threads = 1;

DROP DICTIONARY IF EXISTS volteo.dict_product SYNC;
CREATE DICTIONARY volteo.dict_product
(id UInt64, parent_id UInt64 DEFAULT 0 HIERARCHICAL, node_id String, name String, node_type String)
PRIMARY KEY id SOURCE(CLICKHOUSE(DB 'volteo' TABLE 'hier_product'))
LAYOUT(HASHED()) LIFETIME(MIN 0 MAX 0);

DROP DICTIONARY IF EXISTS volteo.dict_market SYNC;
CREATE DICTIONARY volteo.dict_market
(id UInt64, parent_id UInt64 DEFAULT 0 HIERARCHICAL, node_id String, name String, node_type String)
PRIMARY KEY id SOURCE(CLICKHOUSE(DB 'volteo' TABLE 'hier_market'))
LAYOUT(HASHED()) LIFETIME(MIN 0 MAX 0);

DROP DICTIONARY IF EXISTS volteo.dict_cc SYNC;
CREATE DICTIONARY volteo.dict_cc
(id UInt64, parent_id UInt64 DEFAULT 0 HIERARCHICAL, node_id String, name String, node_type String)
PRIMARY KEY id SOURCE(CLICKHOUSE(DB 'volteo' TABLE 'hier_cc'))
LAYOUT(HASHED()) LIFETIME(MIN 0 MAX 0);

-- The 400M fact table is built by clickhouse/build_400m.py from real generator
-- output (an ensemble of independently calibrated runs), not defined here.
