select
    entity,
    name                        as entity_name,
    currency,
    country,
    gaap,
    role,
    toFloat64(nci_pct)          as nci_pct,
    toInt32(consolidated_from_month) as consolidated_from_month,
    notes
from volteo.dim_entity
order by entity
