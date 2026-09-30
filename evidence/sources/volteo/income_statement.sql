select
    report_id,
    toInt32(sort_order)             as sort_order,
    label,
    toInt32(indent)                 as indent,
    toUInt8(is_subtotal)            as is_subtotal,
    entity,
    scenario_key,
    toInt32(fiscal_year)            as fiscal_year,
    period_key,
    toInt32(period_no)              as period_no,
    toFloat64(value)                as value,
    toFloat64(value_constant_fx)    as value_constant_fx
from volteo.v_income_statement
