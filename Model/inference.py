"""Generate a competition submission from a saved TensorFlow model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import tensorflow as tf

try:
    from .config_ml import ID_COLUMN, MODEL_PATH, PREPROCESSOR_PATH, RANKING_PATH, SUBMISSION_TEMPLATE_PATH, TARGET
    from .data_loader import load_inference_data
    from .feature_engineering import build_features, transform_features
except ImportError:
    from config_ml import ID_COLUMN, MODEL_PATH, PREPROCESSOR_PATH, RANKING_PATH, SUBMISSION_TEMPLATE_PATH, TARGET
    from data_loader import load_inference_data
    from feature_engineering import build_features, transform_features


def predict(
    model_path=MODEL_PATH,
    preprocessor_path=PREPROCESSOR_PATH,
    feature_path=RANKING_PATH,
    submission_template=SUBMISSION_TEMPLATE_PATH,
    output="submission.csv",
):
    template = load_inference_data(submission_template)
    features = load_inference_data(feature_path)
    if ID_COLUMN not in template or ID_COLUMN not in features:
        raise KeyError(f"Both files must contain {ID_COLUMN}.")
    data = template[[ID_COLUMN]].merge(features, on=ID_COLUMN, how="left", suffixes=("", "_features"))
    missing = data["MVT_TIME_UTC_mvt"].isna() if "MVT_TIME_UTC_mvt" in data else pd.Series(True, index=data.index)
    if missing.any():
        raise ValueError(f"Feature data is missing for {int(missing.sum())} submission rows.")
    state = json.loads(Path(preprocessor_path).read_text(encoding="utf-8"))
    x = transform_features(build_features(data), state)
    predictions = tf.keras.models.load_model(model_path).predict(x, verbose=0).ravel()
    result = pd.DataFrame({ID_COLUMN: template[ID_COLUMN], TARGET: predictions.clip(min=0)})
    result.to_csv(output, index=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--preprocessor", type=Path, default=PREPROCESSOR_PATH)
    parser.add_argument("--features", type=Path, default=RANKING_PATH)
    parser.add_argument("--template", type=Path, default=SUBMISSION_TEMPLATE_PATH)
    parser.add_argument("--output", type=Path, default="submission.csv")
    args = parser.parse_args()
    predict(args.model, args.preprocessor, args.features, args.template, args.output)
