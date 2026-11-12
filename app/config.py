"""Central configuration. Change names here if your Snowflake objects differ."""
import os

DATABASE = "DEVICE_FINANCING_DB"
SCHEMA = "FINANCING"
MODEL_STAGE = "MODEL_STAGE"
MODEL_VERSION = "V1"
BUNDLE_FILE = "default_models.joblib"

REGISTRY_MODELS = {
    "HGB": "DEFAULT_MODEL_HGB",
    "INTERNAL": "DEFAULT_MODEL_INTERNAL",
    "CREDIT": "DEFAULT_MODEL_CREDIT",
}
MODEL_LABELS = {
    "HGB": "Gradient boosting (proposed)",
    "INTERNAL": "Internal risk score only",
    "CREDIT": "Credit score only",
}

# Local fallback (offline demo / presentation backup)
LOCAL_ARTIFACTS = os.environ.get(
    "DASHBOARD_LOCAL_ARTIFACTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "local_artifacts"),
)

REGIME_BREAKS = ("2023-08-01", "2024-05-01")
REGIMES = ["Regime 1", "Regime 2", "Regime 3"]
REGIME_LABELS = {
    "Regime 1": "Regime 1 · Jan 2023 – Jul 2023",
    "Regime 2": "Regime 2 · Aug 2023 – Apr 2024 (looser)",
    "Regime 3": "Regime 3 · May 2024 onward (tighter)",
}
TIERS = ["Low", "Mid", "High"]

# Hard input limits from the data dictionary; outside these, the app routes to Review.
INPUT_LIMITS = {
    "CREDIT_SCORE": (300, 850),
    "INTERNAL_RISK_SCORE": (-3.74, 6.87),
    "AGE": (18, 79),
    "INCOME": (15_000, 200_000),
}

SCENARIO_APPLICATIONS = 1_000_000