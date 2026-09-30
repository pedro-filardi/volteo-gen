select
    issue,
    entity,
    toInt32(fiscal_year)        as fiscal_year,
    period_key,
    toInt32(rows_affected)      as rows_affected,
    toFloat64(amount)           as amount
from volteo.v_data_quality
order by issue, period_key
