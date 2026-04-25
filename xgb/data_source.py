"""
Data source layer: fetch and aggregate blood data from MySQL.
Returns DataFrames compatible with the existing XGBoost feature pipeline.
"""

import logging
from typing import Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from product_category import classify_product

log = logging.getLogger(__name__)

# --------------- Collection (采血) ---------------

_COLLECTION_SQL = """
SELECT
    DATE(blood_collection_time)  AS `date`,
    collection_department        AS `血站`,
    COALESCE(precheck_blood_type, archive_blood_type) AS `血型`,
    SUM(base_unit_value)         AS `总血量`
FROM blood_collection_fact
WHERE blood_collection_time IS NOT NULL
    {station_filter}
    {date_filter}
GROUP BY DATE(blood_collection_time),
         collection_department,
         COALESCE(precheck_blood_type, archive_blood_type)
ORDER BY `date`
"""


def load_collection_daily(
    engine: Engine,
    station: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load daily collection volume from DB.

    Returns DataFrame with columns: date, 血站, 血型, 总血量
    Compatible with existing remove_group_data.csv schema.
    """
    params = {}
    station_filter = ""
    date_filter = ""

    if station:
        station_filter = "AND collection_department = :station"
        params["station"] = station
    if start_date:
        date_filter += " AND blood_collection_time >= :start_date"
        params["start_date"] = start_date
    if end_date:
        date_filter += (
            " AND blood_collection_time < DATE_ADD(:end_date, INTERVAL 1 DAY)"
        )
        params["end_date"] = end_date

    sql = _COLLECTION_SQL.format(station_filter=station_filter, date_filter=date_filter)
    df = pd.read_sql(text(sql), engine, params=params)
    df["date"] = pd.to_datetime(df["date"])
    df["血型"] = df["血型"].astype(str).str.strip().str.upper()
    df["总血量"] = pd.to_numeric(df["总血量"], errors="coerce").fillna(0.0)
    log.info("Loaded collection data: %d rows", len(df))
    return df


# --------------- Supply (供血) ---------------

_SUPPLY_SQL = """
SELECT
    DATE(issue_time)       AS `date`,
    issuing_org            AS `发血机构`,
    blood_product_name,
    SUM(base_unit_value)   AS `raw_volume`
FROM blood_supply_fact
WHERE issue_time IS NOT NULL
    {org_filter}
    {date_filter}
GROUP BY DATE(issue_time), issuing_org, blood_product_name
ORDER BY `date`
"""


def load_supply_daily(
    engine: Engine,
    scope: str = "global",
    org: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load daily supply volume from DB.

    scope='global': group by (date, 供血类型)         -> columns: date, 供血类型, 总供血量
    scope='org':    group by (date, 发血机构, 供血类型) -> columns: date, 发血机构, 供血类型, 总供血量
    """
    params = {}
    org_filter = ""
    date_filter = ""

    if org:
        org_filter = "AND issuing_org = :org"
        params["org"] = org
    if start_date:
        date_filter += " AND issue_time >= :start_date"
        params["start_date"] = start_date
    if end_date:
        date_filter += " AND issue_time < DATE_ADD(:end_date, INTERVAL 1 DAY)"
        params["end_date"] = end_date

    sql = _SUPPLY_SQL.format(org_filter=org_filter, date_filter=date_filter)
    df = pd.read_sql(text(sql), engine, params=params)
    df["date"] = pd.to_datetime(df["date"])

    # Classify product names
    df["供血类型"] = df["blood_product_name"].apply(classify_product)

    # Drop '其他'
    df = df[df["供血类型"] != "其他"].copy()

    # Re-aggregate by scope
    if scope == "global":
        result = (
            df.groupby(["date", "供血类型"], as_index=False)["raw_volume"]
            .sum()
            .rename(columns={"raw_volume": "总供血量"})
        )
    else:  # scope == 'org'
        result = (
            df.groupby(["date", "发血机构", "供血类型"], as_index=False)["raw_volume"]
            .sum()
            .rename(columns={"raw_volume": "总供血量"})
        )

    log.info("Loaded supply data (scope=%s): %d rows", scope, len(result))
    return result
