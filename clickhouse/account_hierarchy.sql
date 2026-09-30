-- ===========================================================================
-- One account tree spanning all four local charts, so the dashboard can drill
-- accounts the same way it drills cost centres.
--
-- The four charts have genuinely different depth (Sage 2, SKR03/PGC 4, Business
-- Central 6) and overlapping code numbers, so they are unioned under a synthetic
-- root with the CoA as the first real level. node_id is already CoA-qualified
-- ("SKR03:8400"), which is what keeps the codes from colliding.
-- ===========================================================================

DROP TABLE IF EXISTS volteo.hier_account SYNC;
CREATE TABLE volteo.hier_account ENGINE = MergeTree ORDER BY id AS
WITH
    -- Synthetic root + one node per chart, then every real account node beneath.
    src AS (
        SELECT 'ALL'                              AS node_id,
               ''                                 AS pnode,
               'All accounts'                     AS name,
               'root'                             AS node_type,
               ''                                 AS coa_id,
               ''                                 AS code
        UNION ALL
        SELECT concat('COA:', coa_id), 'ALL', coa_id, 'chart', coa_id, ''
        FROM (SELECT DISTINCT coa_id FROM volteo.dim_account_node)
        UNION ALL
        SELECT node_id,
               if(empty(parent_id), concat('COA:', coa_id), parent_id),
               concat(if(empty(code), '', concat(code, '  ')), name),
               node_type,
               coa_id,
               code
        FROM volteo.dim_account_node
    ),
    numbered AS (
        SELECT rowNumberInAllBlocks() + 1 AS id, node_id, pnode, name, node_type, coa_id, code
        FROM (SELECT * FROM src ORDER BY node_id)
    )
SELECT n.id AS id, p.id AS parent_id, n.node_id AS node_id, n.name AS name,
       n.node_type AS node_type, n.coa_id AS coa_id, n.code AS code
FROM numbered AS n LEFT JOIN numbered AS p ON p.node_id = n.pnode
SETTINGS max_threads = 1;

DROP DICTIONARY IF EXISTS volteo.dict_account SYNC;
CREATE DICTIONARY volteo.dict_account
(id UInt64, parent_id UInt64 DEFAULT 0 HIERARCHICAL, node_id String, name String,
 node_type String, coa_id String, code String)
PRIMARY KEY id SOURCE(CLICKHOUSE(DB 'volteo' TABLE 'hier_account'))
LAYOUT(HASHED()) LIFETIME(MIN 0 MAX 0);

-- (entity, local account code) -> surrogate id, so the generator's fact rows can be
-- keyed without carrying the CoA around.
DROP TABLE IF EXISTS volteo.map_account_key SYNC;
CREATE TABLE volteo.map_account_key ENGINE = MergeTree ORDER BY (entity, code) AS
SELECT e.entity AS entity, h.code AS code, h.id AS id
FROM volteo.hier_account AS h
INNER JOIN volteo.dim_entity AS e ON e.coa_id = h.coa_id
WHERE h.code != '';
