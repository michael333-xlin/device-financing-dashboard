"""Data access: one query() call that runs on Snowflake (Snowpark) or locally (DuckDB).

All SQL in the app is written to run unchanged on both engines, so the offline
fallback shows exactly the same numbers as the Snowflake deployment.
"""
from __future__ import annotations

import decimal
import os

import pandas as pd
import streamlit as st

import config as cfg

TABLES = [
    "APPLICATIONS", "APPLICATION_SCORES", "MODEL_METRICS", "PAYOFFS", "SUPPORT_BOUNDS",
    "POLICY_THRESHOLDS", "TIER_DEFAULTS", "REFERENCE_VALUES", "MODEL_INFO",
]


class SnowflakeBackend:
    name = "Snowflake"

    def __init__(self, session):
        self.session = session

    def table(self, name: str) -> str:
        return f"{cfg.DATABASE}.{cfg.SCHEMA}.{name}"

    def query(self, sql: str, params: list | None = None) -> pd.DataFrame:
        return self.session.sql(sql, params=params or None).to_pandas()


class LocalBackend:
    name = "Local demo (DuckDB)"

    def __init__(self, folder: str):
        import duckdb

        self.con = duckdb.connect()
        missing = [t for t in TABLES if not os.path.exists(os.path.join(folder, f"{t}.parquet"))]
        if missing:
            raise FileNotFoundError(
                f"Local artifacts missing in {folder}: {missing}. "
                "Run: python model/train_register.py --local --csv financing_data.csv"
            )
        for t in TABLES:
            path = os.path.join(folder, f"{t}.parquet").replace("'", "''")
            self.con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{path}')")

    def table(self, name: str) -> str:
        return name

    def query(self, sql: str, params: list | None = None) -> pd.DataFrame:
        return self.con.execute(sql, params or []).df()


@st.cache_resource(show_spinner=False)
def get_backend():
    try:
        from snowflake.snowpark.context import get_active_session

        return SnowflakeBackend(get_active_session())
    except Exception:
        return LocalBackend(cfg.LOCAL_ARTIFACTS)


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).upper() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            sample = df[c].dropna()
            if len(sample) and isinstance(sample.iloc[0], decimal.Decimal):
                df[c] = df[c].astype(float)
    return df


def T(name: str) -> str:
    """Fully qualified table name for the active backend."""
    return get_backend().table(name)


@st.cache_data(ttl=3600, show_spinner=False)
def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    return _normalise(get_backend().query(sql, list(params)))


def regime_case(col: str = "APPLICATION_DATE") -> str:
    b1, b2 = cfg.REGIME_BREAKS
    return (
        f"CASE WHEN {col} < DATE '{b1}' THEN 'Regime 1' "
        f"WHEN {col} < DATE '{b2}' THEN 'Regime 2' ELSE 'Regime 3' END"
    )


def base_cte() -> str:
    """Applications joined to model scores, with regime and funding flags."""
    return f"""
    WITH A AS (
        SELECT a.*,
               {regime_case('a.APPLICATION_DATE')} AS REGIME,
               CASE WHEN a.FINAL_STATUS <> 'Not_Funded' THEN 1 ELSE 0 END AS FUNDED,
               s.SPLIT, s.PD_HGB, s.PD_INTERNAL, s.PD_CREDIT, s.IN_SUPPORT
        FROM {T('APPLICATIONS')} a
        JOIN {T('APPLICATION_SCORES')} s ON a.APPLICATION_ID = s.APPLICATION_ID
    )
    """


# ----------------------------------------------------------------------------- small lookups
def lookup_tables() -> dict:
    return dict(
        payoffs=query(f"SELECT * FROM {T('PAYOFFS')}"),
        policy=query(f"SELECT * FROM {T('POLICY_THRESHOLDS')}"),
        bounds=query(f"SELECT * FROM {T('SUPPORT_BOUNDS')}"),
        tiers=query(f"SELECT * FROM {T('TIER_DEFAULTS')}"),
        reference=query(f"SELECT * FROM {T('REFERENCE_VALUES')}"),
        metrics=query(f"SELECT * FROM {T('MODEL_METRICS')}"),
        model_info=query(f"SELECT * FROM {T('MODEL_INFO')}"),
    )


def sql_in(values: list[str], allowed: list[str]) -> str:
    """Whitelisted IN-list (values come from fixed UI options only)."""
    safe = [v for v in values if v in allowed]
    return "(" + ", ".join("'" + v.replace("'", "''") + "'" for v in safe) + ")" if safe else "(NULL)"