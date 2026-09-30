select
    entity,
    toInt32(fiscal_year)    as fiscal_year,
    arc,
    toFloat64(gm_target)    as gm_target,
    toFloat64(gm_actual)    as gm_actual,
    toFloat64(ni_target)    as ni_target,
    toFloat64(ni_actual)    as ni_actual,
    toFloat64(net_revenue)  as net_revenue
from volteo.meta_calibration
order by entity, fiscal_year
