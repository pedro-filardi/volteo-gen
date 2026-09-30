select
    entity,
    toInt32(fiscal_year)    as fiscal_year,
    toInt32(period_no)      as period_no,
    period_key,
    toFloat64(result)       as result
from volteo.rpt_monthly_result
order by entity, period_key
