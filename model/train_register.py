"""
Train, evaluate and publish the device-financing default-risk models.

The modelling choices reproduce eda_and_modeling.ipynb (Q7 and Q8):
  * funded plans with an observed default_flag only
  * stratified 75/25 split, random_state = 42, in original CSV row order
  * baselines: cubic spline + logistic regression on credit_score / internal_risk_score
  * candidate: HistGradientBoostingClassifier with the Q8 settings
The only change is packaging: one-hot encoding now lives inside a scikit-learn
Pipeline, so the dashboard applies exactly the same feature transformation.

Where to run
------------
1. Snowflake Notebook (recommended). Add packages scikit-learn, snowflake-ml-python,
   pandas, joblib. Paste this file into a cell (or upload it) and run. It writes the
   evidence tables, registers three models in the Model Registry and saves a joblib
   fallback bundle to @MODEL_STAGE.
2. Locally, to build an offline demo / presentation fallback:
       python model/train_register.py --local --csv financing_data.csv --out local_artifacts
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

# ----------------------------------------------------------------------------- config
DATABASE = "DEVICE_FINANCING_DB"
SCHEMA = "FINANCING"
MODEL_STAGE = "MODEL_STAGE"
MODEL_VERSION = "V1"
BUNDLE_FILE = "default_models.joblib"

REGIME_BREAKS = (pd.Timestamp("2023-08-01"), pd.Timestamp("2024-05-01"))
OOT_CUTOFF = pd.Timestamp("2024-10-01")  # out-of-time check: train before, test Oct-Dec 2024
RANDOM_STATE = 42

NUMERIC = ["CREDIT_SCORE", "INTERNAL_RISK_SCORE", "AGE", "INCOME"]
CATEGORICAL = ["EXISTING_CUSTOMER", "DEVICE_REQUESTED", "REGIME"]
FEATURES = NUMERIC + CATEGORICAL
INT_FEATURES = ["CREDIT_SCORE", "AGE", "EXISTING_CUSTOMER"]
FLOAT_FEATURES = ["INTERNAL_RISK_SCORE", "INCOME"]
STR_FEATURES = ["DEVICE_REQUESTED", "REGIME"]

HGB_PARAMS = dict(
    loss="log_loss", learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
    min_samples_leaf=100, l2_regularization=10, early_stopping=True,
    random_state=RANDOM_STATE,
)

MODEL_NAMES = {  # registry name -> short key used in tables
    "DEFAULT_MODEL_HGB": "HGB",
    "DEFAULT_MODEL_INTERNAL": "INTERNAL",
    "DEFAULT_MODEL_CREDIT": "CREDIT",
}
MODEL_LABELS = {
    "HGB": "Gradient boosting (proposed)",
    "INTERNAL": "Internal risk score only",
    "CREDIT": "Credit score only",
}


# ----------------------------------------------------------------------------- features
def assign_regime(dates: pd.Series) -> np.ndarray:
    dates = pd.to_datetime(dates)
    return np.select(
        [dates < REGIME_BREAKS[0], dates < REGIME_BREAKS[1]],
        ["Regime 1", "Regime 2"],
        default="Regime 3",
    )


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the model input frame with the exact dtypes used in training.

    The dashboard has an identical copy of this function (app/scoring.py).
    """
    x = frame[FEATURES].copy()
    for c in INT_FEATURES:
        x[c] = pd.to_numeric(x[c], errors="raise").astype("int64")
    for c in FLOAT_FEATURES:
        x[c] = pd.to_numeric(x[c], errors="raise").astype("float64")
    for c in STR_FEATURES:
        x[c] = x[c].astype(str).astype(object)
    return x


# ----------------------------------------------------------------------------- models
def build_hgb() -> Pipeline:
    prep = ColumnTransformer(
        [
            ("num", "passthrough", NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )
    return Pipeline([("prep", prep), ("hgb", HistGradientBoostingClassifier(**HGB_PARAMS))])


def build_spline(feature: str) -> Pipeline:
    """Single-score baseline that accepts the full feature frame (shared schema)."""
    spline = SplineTransformer(n_knots=5, degree=3, knots="quantile", include_bias=False)
    return Pipeline(
        [
            ("select", ColumnTransformer([("spline", spline, [feature])])),
            ("scale", StandardScaler()),
            ("logit", LogisticRegression(max_iter=2000)),
        ]
    )


def build_models() -> dict[str, Pipeline]:
    return {
        "HGB": build_hgb(),
        "INTERNAL": build_spline("INTERNAL_RISK_SCORE"),
        "CREDIT": build_spline("CREDIT_SCORE"),
    }


def metric_rows(sample_label: str, y: pd.Series, preds: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for key, p in preds.items():
        rows.append(
            dict(
                EVAL_SAMPLE=sample_label,
                MODEL=key,
                MODEL_LABEL=MODEL_LABELS[key],
                N=int(len(y)),
                LOG_LOSS=float(log_loss(y, p, labels=[0, 1])),
                AUC=float(roc_auc_score(y, p)),
                MEAN_ACTUAL=float(y.mean()),
                MEAN_PREDICTED=float(np.mean(p)),
            )
        )
    return rows


# ----------------------------------------------------------------------------- data I/O
def get_snowflake_session(connection_name: str | None):
    try:
        from snowflake.snowpark.context import get_active_session

        return get_active_session()
    except Exception:
        if connection_name is None:
            return None
        from snowflake.snowpark import Session

        return Session.builder.config("connection_name", connection_name).create()


def load_from_snowflake(session) -> pd.DataFrame:
    session.use_schema(f"{DATABASE}.{SCHEMA}")
    df = session.table("APPLICATIONS").sort("APPLICATION_ID").to_pandas()
    df.columns = [c.upper() for c in df.columns]
    return df


def load_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.upper() for c in df.columns]
    df.insert(0, "APPLICATION_ID", np.arange(1, len(df) + 1))
    return df


# ----------------------------------------------------------------------------- main build
def build_everything(df: pd.DataFrame) -> dict:
    df = df.sort_values("APPLICATION_ID").reset_index(drop=True)
    df["APPLICATION_DATE"] = pd.to_datetime(df["APPLICATION_DATE"]).dt.normalize()
    df["REGIME"] = assign_regime(df["APPLICATION_DATE"])
    for c in ["DEFAULT_FLAG", "PROFIT", "DEVICE_PRICE", "DOWN_PAYMENT_BASE",
              "DOWN_PAYMENT_FINAL", "INTERNAL_RISK_SCORE", "INCOME"]:
        df[c] = pd.to_numeric(df[c]).astype("float64")

    funded = df.dropna(subset=["DEFAULT_FLAG", "CREDIT_SCORE", "INTERNAL_RISK_SCORE"])
    train, test = train_test_split(
        funded, test_size=0.25, random_state=RANDOM_STATE, stratify=funded["DEFAULT_FLAG"]
    )
    y_train = train["DEFAULT_FLAG"].astype(int)
    y_test = test["DEFAULT_FLAG"].astype(int)
    x_train, x_test = prepare_features(train), prepare_features(test)

    # Production models: fit on the 75% training split, so the published holdout
    # evidence describes exactly the models the dashboard uses.
    models = build_models()
    for m in models.values():
        m.fit(x_train, y_train)

    preds_test = {k: m.predict_proba(x_test)[:, 1] for k, m in models.items()}
    metrics = metric_rows("Random 25% holdout (all regimes)", y_test, preds_test)
    r3 = (test["REGIME"] == "Regime 3").to_numpy()
    metrics += metric_rows(
        "Random 25% holdout (Regime 3 only)", y_test[r3], {k: v[r3] for k, v in preds_test.items()}
    )

    # Out-of-time validation: evaluation-only models, never published.
    early = funded[funded["APPLICATION_DATE"] < OOT_CUTOFF]
    late = funded[funded["APPLICATION_DATE"] >= OOT_CUTOFF]
    oot_models = build_models()
    for m in oot_models.values():
        m.fit(prepare_features(early), early["DEFAULT_FLAG"].astype(int))
    oot_preds = {k: m.predict_proba(prepare_features(late))[:, 1] for k, m in oot_models.items()}
    metrics += metric_rows(
        f"Out-of-time: trained before {OOT_CUTOFF:%b %Y}, tested {OOT_CUTOFF:%b}-Dec 2024",
        late["DEFAULT_FLAG"].astype(int),
        oot_preds,
    )
    metrics = pd.DataFrame(metrics)

    # Score every application (funded or not) with the production models.
    x_all = prepare_features(df)
    split = np.where(df["DEFAULT_FLAG"].isna(), "unfunded", "train")
    split[df.index.isin(test.index)] = "test"

    # Support bounds: observed ranges among FUNDED plans per regime.
    bounds = pd.DataFrame(
        [dict(REGIME=r, FEATURE=c, FUNDED_MIN=float(g[c].min()), FUNDED_MAX=float(g[c].max()))
         for r, g in funded.groupby("REGIME") for c in NUMERIC]
    )
    max_internal = funded.groupby("REGIME")["INTERNAL_RISK_SCORE"].max()
    min_credit = funded.groupby("REGIME")["CREDIT_SCORE"].min()
    in_support = (df["INTERNAL_RISK_SCORE"] <= df["REGIME"].map(max_internal)) & (
        df["CREDIT_SCORE"] >= df["REGIME"].map(min_credit)
    )

    scores = pd.DataFrame(
        {
            "APPLICATION_ID": df["APPLICATION_ID"].astype("int64"),
            "REGIME": df["REGIME"],
            "SPLIT": split,
            "PD_HGB": models["HGB"].predict_proba(x_all)[:, 1],
            "PD_INTERNAL": models["INTERNAL"].predict_proba(x_all)[:, 1],
            "PD_CREDIT": models["CREDIT"].predict_proba(x_all)[:, 1],
            "IN_SUPPORT": in_support.astype(bool),
        }
    )

    # Payoffs (G = average profit without default, L = average loss with default),
    # estimated from TRAINING observations only, by regime and final device tier.
    tr = train.copy()
    pay_rows = []
    for regime, g in tr.groupby("REGIME"):
        for tier, gg in [("ALL", g)] + list(g.groupby("DEVICE_FINAL")):
            good = gg.loc[gg["DEFAULT_FLAG"] == 0, "PROFIT"]
            bad = gg.loc[gg["DEFAULT_FLAG"] == 1, "PROFIT"]
            G, L = float(good.mean()), float(-bad.mean())
            pay_rows.append(
                dict(REGIME=regime, DEVICE_TIER=tier, GAIN=G, LOSS=L,
                     BREAK_EVEN_PD=G / (G + L), N_GOOD=int(len(good)), N_BAD=int(len(bad)))
            )
    payoffs = pd.DataFrame(pay_rows)

    # Current policy rule, inferred from historical decisions (internal score cutoffs).
    policy = (
        df.groupby(["REGIME", "INITIAL_DECISION"])["INTERNAL_RISK_SCORE"].max().unstack()
        .rename(columns={"Approve": "APPROVE_MAX", "Conditional": "CONDITIONAL_MAX"})
        [["APPROVE_MAX", "CONDITIONAL_MAX"]].reset_index()
    )
    policy["SOURCE"] = "Inferred from historical initial_decision"

    # Offer defaults: typical price and base down payment by tier (final regime).
    r3_all = df[df["REGIME"] == "Regime 3"]
    # Price belongs to the device actually sold, so group by DEVICE_FINAL.
    tiers = (
        r3_all.groupby("DEVICE_FINAL")
        .agg(MEDIAN_PRICE=("DEVICE_PRICE", "median"), MEAN_BASE_DOWN=("DOWN_PAYMENT_BASE", "mean"))
        .reset_index().rename(columns={"DEVICE_FINAL": "DEVICE_TIER"})
    )
    ref = r3_all[r3_all["DEFAULT_FLAG"].notna()]
    reference = pd.DataFrame(
        [dict(FEATURE=c, REFERENCE_VALUE=float(ref[c].median())) for c in NUMERIC]
        + [dict(FEATURE="EXISTING_CUSTOMER", REFERENCE_VALUE=float(ref["EXISTING_CUSTOMER"].mode()[0]))]
    )

    model_info = pd.DataFrame(
        [dict(MODEL=k, REGISTRY_NAME=rn, VERSION=MODEL_VERSION, SKLEARN_VERSION=sklearn.__version__,
              TRAINED_AT=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
              N_TRAIN=int(len(train)), N_TEST=int(len(test)))
         for rn, k in MODEL_NAMES.items()]
    )

    bundle = dict(
        models=models, features=FEATURES, version=MODEL_VERSION,
        sklearn_version=sklearn.__version__, created=model_info["TRAINED_AT"].iloc[0],
    )
    apps = df.drop(columns=["REGIME"]).copy()
    apps["APPLICATION_DATE"] = apps["APPLICATION_DATE"].dt.date
    return dict(
        bundle=bundle, sample_input=x_train.head(200),
        tables=dict(
            MODEL_METRICS=metrics, APPLICATION_SCORES=scores, PAYOFFS=payoffs,
            SUPPORT_BOUNDS=bounds, POLICY_THRESHOLDS=policy, TIER_DEFAULTS=tiers,
            REFERENCE_VALUES=reference, MODEL_INFO=model_info,
        ),
        applications=apps,
    )


# ----------------------------------------------------------------------------- publish
def publish_local(result: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    result["applications"].to_parquet(os.path.join(out_dir, "APPLICATIONS.parquet"), index=False)
    for name, table in result["tables"].items():
        table.to_parquet(os.path.join(out_dir, f"{name}.parquet"), index=False)
    joblib.dump(result["bundle"], os.path.join(out_dir, BUNDLE_FILE))
    print(f"Local artifacts written to {os.path.abspath(out_dir)}")


def publish_snowflake(session, result: dict, overwrite: bool) -> None:
    session.use_schema(f"{DATABASE}.{SCHEMA}")
    for name, table in result["tables"].items():
        session.write_pandas(table, name, auto_create_table=True, overwrite=True, quote_identifiers=False)
        print(f"wrote {name}: {len(table):,} rows")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, BUNDLE_FILE)
        joblib.dump(result["bundle"], path)
        session.file.put(path, f"@{MODEL_STAGE}", auto_compress=False, overwrite=True)
    print(f"uploaded fallback bundle to @{MODEL_STAGE}/{BUNDLE_FILE}")

    from snowflake.ml.registry import Registry

    registry = Registry(session=session, database_name=DATABASE, schema_name=SCHEMA)
    metrics = result["tables"]["MODEL_METRICS"]
    for registry_name, key in MODEL_NAMES.items():
        if overwrite:
            try:
                registry.get_model(registry_name).delete_version(MODEL_VERSION)
            except Exception:
                pass
        m = metrics[metrics["MODEL"] == key].set_index("EVAL_SAMPLE")["LOG_LOSS"].to_dict()
        registry.log_model(
            result["bundle"]["models"][key],
            model_name=registry_name,
            version_name=MODEL_VERSION,
            sample_input_data=result["sample_input"],
            # target_methods is not a direct argument of log_model(); it goes in options.
            # relax_version=False keeps the exact versions below: SplineTransformer pickles
            # scipy BSpline objects, which cannot be loaded by a different scipy version.
            options={"target_methods": ["predict_proba"], "relax_version": False},
            conda_dependencies=[
                f"scikit-learn=={sklearn.__version__}",
                f"scipy=={scipy.__version__}",
                f"numpy=={np.__version__}",
            ],
            metrics={"test_log_loss": m},
            comment=f"{MODEL_LABELS[key]}; trained on funded plans, 75% stratified split",
        )
        print(f"registered {registry_name}/{MODEL_VERSION}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="read a CSV and write local artifacts")
    parser.add_argument("--csv", default="financing_data.csv")
    parser.add_argument("--out", default="local_artifacts")
    parser.add_argument("--connection", default=None, help="connections.toml name for remote runs")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing model version")
    args, _ = parser.parse_known_args()

    session = None if args.local else get_snowflake_session(args.connection)
    df = load_from_csv(args.csv) if session is None else load_from_snowflake(session)
    result = build_everything(df)
    print(result["tables"]["MODEL_METRICS"][["EVAL_SAMPLE", "MODEL", "N", "LOG_LOSS", "AUC"]]
          .to_string(index=False))
    if session is None:
        publish_local(result, args.out)
    else:
        publish_snowflake(session, result, args.overwrite)


if __name__ == "__main__":
    main()