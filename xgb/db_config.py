"""
Database configuration for blood prediction service.
Reads credentials from environment variables.
"""

import os
from functools import lru_cache

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine import URL

DB_HOST = os.environ.get("BLOOD_DB_HOST", "101.37.104.90")
DB_PORT = os.environ.get("BLOOD_DB_PORT", "13306")
DB_NAME = os.environ.get("BLOOD_DB_NAME", "blood")
DB_USER = os.environ.get("BLOOD_DB_USER", "root")
DB_PASSWORD = os.environ.get("BLOOD_DB_PASSWORD", "yx@)25")
DB_QUERY = {
    "charset": "utf8mb4",
}


def get_db_url() -> URL:
    if not DB_USER or not DB_PASSWORD:
        raise RuntimeError(
            "BLOOD_DB_USER and BLOOD_DB_PASSWORD environment variables are required"
        )
    return URL.create(
        drivername="mysql+pymysql",
        username=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=int(DB_PORT),
        database=DB_NAME,
        query=DB_QUERY,
    )


def get_masked_url() -> str:
    query = "&".join(f"{key}={value}" for key, value in DB_QUERY.items())
    return f"mysql+pymysql://{DB_USER}:****@{DB_HOST}:{DB_PORT}/{DB_NAME}?{query}"


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_engine(
        get_db_url(),
        pool_pre_ping=True,
        connect_args={"ssl_disabled": True},
    )


def check_connection() -> bool:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
