select
    entity,
    toInt32(fiscal_year)                        as fiscal_year,
    toFloat64(nci_pct)                          as nci_pct,
    toFloat64(entity_net_income)                as entity_net_income,
    toFloat64(nci_before_adjustment)            as nci_before_adjustment,
    toFloat64(nci_unrealised_profit_adjustment) as nci_unrealised_adjustment,
    toFloat64(nci_net_income)                   as nci_net_income
from volteo.fact_nci
order by entity, fiscal_year
