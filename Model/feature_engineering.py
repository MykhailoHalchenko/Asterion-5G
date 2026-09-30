"""Feature construction and deterministic TensorFlow input encoding."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from .config_ml import CATEGORICAL_FEATURES, RF_SCORE_COLUMN, SCHEDULE_COLUMN, TIME_COLUMN
except ImportError:
    from config_ml import CATEGORICAL_FEATURES, RF_SCORE_COLUMN, SCHEDULE_COLUMN, TIME_COLUMN


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready, human-readable features from movement records."""
    data = frame.copy()
    event_time = _event_time(data)
    features = pd.DataFrame(index=data.index)
    features["hour_utc"] = event_time.dt.hour.astype("float32")
    features["day_of_week"] = event_time.dt.dayofweek.astype("float32")
    features["phase_of_day"] = (event_time.dt.hour // 6).astype("float32")
    features["is_weekend"] = (event_time.dt.dayofweek >= 5).astype("float32")

    if SCHEDULE_COLUMN in data:
        scheduled = pd.to_datetime(data[SCHEDULE_COLUMN], utc=True, errors="coerce")
        features["departure_delay_min"] = (
            (event_time - scheduled).dt.total_seconds() / 60.0
        )
    else:
        features["departure_delay_min"] = np.nan

    features["congestion_15m"] = _congestion(data, event_time)
    features["rf_complexity_score"] = _numeric(data, RF_SCORE_COLUMN)

    for column in CATEGORICAL_FEATURES:
        if column in data:
            features[column] = data[column].fillna("__missing__").astype(str)
        else:
            features[column] = "__missing__"
    return features


def fit_encoder(features: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    categorical = [column for column in CATEGORICAL_FEATURES if column in features]
    categories = {
        column: sorted(features[column].astype(str).unique().tolist())
        for column in categorical
    }
    encoded = _encode(features, categories)
    numeric_columns = [column for column in encoded.columns]
    means = encoded[numeric_columns].mean().fillna(0.0)
    scales = encoded[numeric_columns].std().replace(0, 1).fillna(1.0)
    state = {
        "categories": categories,
        "columns": numeric_columns,
        "means": means.to_dict(),
        "scales": scales.to_dict(),
    }
    return ((encoded - means) / scales).to_numpy(dtype=np.float32), state


def transform_features(features: pd.DataFrame, state: dict[str, Any]) -> np.ndarray:
    encoded = _encode(features, state["categories"]).reindex(
        columns=state["columns"], fill_value=0.0
    )
    encoded = encoded.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    means = pd.Series(state["means"]).reindex(state["columns"]).fillna(0.0)
    scales = pd.Series(state["scales"]).reindex(state["columns"]).replace(0, 1).fillna(1.0)
    return ((encoded - means) / scales).to_numpy(dtype=np.float32)


def _event_time(data: pd.DataFrame) -> pd.Series:
    if TIME_COLUMN in data:
        block = pd.to_datetime(data[TIME_COLUMN], utc=True, errors="coerce")
    else:
        block = pd.Series(pd.NaT, index=data.index, dtype="datetime64[ns, UTC]")
    if "MVT_TIME_UTC_mvt" in data:
        movement = pd.to_datetime(data["MVT_TIME_UTC_mvt"], utc=True, errors="coerce")
        block = block.fillna(movement)
    if block.isna().any():
        raise ValueError("Feature engineering received rows without a usable event time.")
    return block


def _congestion(data: pd.DataFrame, event_time: pd.Series) -> pd.Series:
    phase = data.get("PHASE_mvt", pd.Series("", index=data.index)).astype(str).str.upper()
    valid = phase.isin(["DEP", "ARR"]) & event_time.notna()
    timestamps = event_time.astype("int64").to_numpy() // 10**9
    ordered = np.sort(timestamps[valid.to_numpy()])
    left = np.searchsorted(ordered, timestamps - 900, side="left")
    right = np.searchsorted(ordered, timestamps + 900, side="right")
    counts = right - left - valid.to_numpy().astype(np.int64)
    return pd.Series(np.where(valid, counts, 0), index=data.index, dtype="float32")


def _numeric(data: pd.DataFrame, column: str) -> pd.Series:
    if column not in data:
        return pd.Series(np.nan, index=data.index, dtype="float32")
    return pd.to_numeric(data[column], errors="coerce").astype("float32")


def _encode(features: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    numeric = features.drop(columns=list(categories), errors="ignore").copy()
    numeric = numeric.apply(pd.to_numeric, errors="coerce")
    encoded = [numeric]
    for column, values in categories.items():
        values = list(values)
        one_hot = pd.get_dummies(features[column].astype(str), prefix=column)
        one_hot = one_hot.reindex(columns=[f"{column}_{value}" for value in values], fill_value=0)
        encoded.append(one_hot.astype("float32"))
    return pd.concat(encoded, axis=1)
