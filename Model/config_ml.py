"""Configuration shared by the TensorFlow training and inference pipeline."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "Datasets"
ASTERIX_PIPELINE_DIR = ROOT_DIR / "yezhik-asterix"
TRAINING_GLOB = "training_*.parquet"
RANKING_PATH = DATA_DIR / "ranking.parquet"
SUBMISSION_TEMPLATE_PATH = DATA_DIR / "submitting.parquet"
MODEL_PATH = ROOT_DIR / "model.keras"
PREPROCESSOR_PATH = ROOT_DIR / "model_preprocessor.json"

TARGET = "TAXITIME_SEC_mvt"
ID_COLUMN = "MVT_ID_mvt"
TIME_COLUMN = "BLOCK_TIME_UTC_mvt"
SCHEDULE_COLUMN = "SCHED_TIME_UTC_mvt"
RF_SCORE_COLUMN = "RF_Complexity_Score"

CATEGORICAL_FEATURES = [
    "PHASE_mvt",
    "ADEP_mvt",
    "ADES_mvt",
    "AIRCRAFT_TYPE_mvt",
    "RUNWAY_mvt",
    "STAND_mvt",
    "FLIGHT_RULE_mvt",
    "FLIGHT_TYPE_flt",
    "MARKET_SEGMENT_flt",
    "AIRCRAFT_OPERATOR_flt",
]

DEFAULT_MODEL_PARAMS = {
    "hidden_units": (128, 64),
    "dropout": 0.15,
    "learning_rate": 0.001,
    "batch_size": 1024,
    "epochs": 50,
    "patience": 7,
}
