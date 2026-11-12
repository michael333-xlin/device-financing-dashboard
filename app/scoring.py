"""Model scoring for the dashboard.

Order of preference:
  1. Snowflake Model Registry (warehouse inference, governed versions)
  2. joblib bundle on @MODEL_STAGE, loaded into the app (fast, same artifact)
  3. local joblib bundle (offline demo)
Every result records which source produced it, so the UI can label it.
"""
from __future__ import annotations

import io
import os

import numpy as np
import pandas as pd
import streamlit as st

import config as cfg
from data_layer import SnowflakeBackend, get_backend

NUMERIC = ["CREDIT_SCORE", "INTERNAL_RISK_SCORE", "AGE", "INCOME"]
CATEGORICAL = ["EXISTING_CUSTOMER", "DEVICE_REQUESTED", "REGIME"]
FEATURES = NUMERIC + CATEGORICAL
INT_FEATURES = ["CREDIT_SCORE", "AGE", "EXISTING_CUSTOMER"]
FLOAT_FEATURES = ["INTERNAL_RISK_SCORE", "INCOME"]
STR_FEATURES = ["DEVICE_REQUESTED", "REGIME"]


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Identical to model/train_register.py::prepare_features (training schema)."""
    x = frame[FEATURES].copy()
    for c in INT_FEATURES:
        x[c] = pd.to_numeric(x[c], errors="raise").astype("int64")
    for c in FLOAT_FEATURES:
        x[c] = pd.to_numeric(x[c], errors="raise").astype("float64")
    for c in STR_FEATURES:
        x[c] = x[c].astype(str).astype(object)
    return x


class Scorer:
    def __init__(self):
        self.source = None
        self.detail = ""
        self.models = None  # in-app sklearn pipelines
        self.registry_versions = None
        backend = get_backend()
        errors = []
        if isinstance(backend, SnowflakeBackend):
            try:
                self._init_registry(backend.session)
                return
            except Exception as exc:  # pragma: no cover - needs Snowflake
                errors.append(f"Registry: {exc}")
            try:
                self._init_stage(backend.session)
                return
            except Exception as exc:  # pragma: no cover - needs Snowflake
                errors.append(f"Stage: {exc}")
        try:
            self._init_local()
        except Exception as exc:
            errors.append(f"Local: {exc}")
            self.source = "Unavailable"
            self.detail = " | ".join(errors)

    # -- initialisers ---------------------------------------------------------------
    def _init_registry(self, session):
        from snowflake.ml.registry import Registry

        reg = Registry(session=session, database_name=cfg.DATABASE, schema_name=cfg.SCHEMA)
        self.registry_versions = {
            key: reg.get_model(name).version(cfg.MODEL_VERSION)
            for key, name in cfg.REGISTRY_MODELS.items()
        }
        self.source = "Snowflake Model Registry"
        self.detail = f"{cfg.REGISTRY_MODELS['HGB']} / {cfg.MODEL_VERSION} (warehouse inference)"

    def _init_stage(self, session):
        import joblib

        path = f"@{cfg.DATABASE}.{cfg.SCHEMA}.{cfg.MODEL_STAGE}/{cfg.BUNDLE_FILE}"
        with session.file.get_stream(path) as fh:
            bundle = joblib.load(fh)
        self.models = bundle["models"]
        self.source = "Snowflake stage artifact"
        self.detail = f"{path} · {bundle['version']} · sklearn {bundle['sklearn_version']}"

    def _init_local(self):
        import joblib

        path = os.path.join(cfg.LOCAL_ARTIFACTS, cfg.BUNDLE_FILE)
        bundle = joblib.load(path)
        self.models = bundle["models"]
        self.source = "Local model file"
        self.detail = f"{cfg.BUNDLE_FILE} · {bundle['version']} · sklearn {bundle['sklearn_version']}"

    # -- scoring --------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self.source not in (None, "Unavailable")

    def predict(self, records: pd.DataFrame, keys=("HGB", "INTERNAL", "CREDIT")) -> pd.DataFrame:
        """Return a frame with PD_<KEY> columns, one row per input record (same order)."""
        x = prepare_features(records).reset_index(drop=True)
        out = pd.DataFrame(index=x.index)
        for key in keys:
            if self.registry_versions is not None:
                res = self.registry_versions[key].run(x, function_name="predict_proba")
                new_cols = [c for c in res.columns if c.upper() not in {f.upper() for f in FEATURES}]
                # predict_proba returns one column per class; the last one is P(default = 1).
                out[f"PD_{key}"] = pd.to_numeric(res[new_cols[-1]]).to_numpy()
            else:
                out[f"PD_{key}"] = self.models[key].predict_proba(x)[:, 1]
        return out


@st.cache_resource(show_spinner="Loading risk models…")
def get_scorer() -> Scorer:
    return Scorer()


@st.cache_data(show_spinner=False, max_entries=256)
def score_cached(records_json: str) -> pd.DataFrame:
    """Cache predictions by input so re-renders do not re-run warehouse inference."""
    records = pd.read_json(io.StringIO(records_json), orient="records", dtype=False)
    return get_scorer().predict(records)


def score(records: pd.DataFrame) -> pd.DataFrame:
    return score_cached(records[FEATURES].to_json(orient="records"))