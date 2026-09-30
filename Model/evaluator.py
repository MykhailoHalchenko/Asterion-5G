"""Evaluate a saved model and generate permutation feature importance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

try:
    from .config_ml import DATA_DIR, MODEL_PATH, PREPROCESSOR_PATH, TARGET
    from .data_loader import load_training_data
    from .feature_engineering import build_features, transform_features
except ImportError:
    from config_ml import DATA_DIR, MODEL_PATH, PREPROCESSOR_PATH, TARGET
    from data_loader import load_training_data
    from feature_engineering import build_features, transform_features


def evaluate(model_path=MODEL_PATH, preprocessor_path=PREPROCESSOR_PATH, data_dir=DATA_DIR, output="feature_importance.png"):
    data = load_training_data(data_dir).sort_values("MVT_TIME_UTC_mvt")
    validation = data.iloc[int(len(data) * 0.8):]
    state = json.loads(Path(preprocessor_path).read_text(encoding="utf-8"))
    x = transform_features(build_features(validation), state)
    y = validation[TARGET].to_numpy(dtype=np.float32)
    model = tf.keras.models.load_model(model_path)
    predictions = model.predict(x, verbose=0).ravel()
    errors = predictions - y
    metrics = {"mae": float(np.mean(np.abs(errors))), "rmse": float(np.sqrt(np.mean(errors ** 2)))}

    baseline = metrics["mae"]
    importance = []
    for index, name in enumerate(state["columns"]):
        shuffled = x.copy()
        shuffled[:, index] = np.random.default_rng(42 + index).permutation(shuffled[:, index])
        score = float(np.mean(np.abs(model.predict(shuffled, verbose=0).ravel() - y)))
        importance.append((name, max(0.0, score - baseline)))
    importance.sort(key=lambda item: item[1], reverse=True)
    names, values = zip(*importance[:20])
    plt.figure(figsize=(10, 7))
    plt.barh(names[::-1], values[::-1])
    plt.xlabel("MAE increase after permutation")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--preprocessor", type=Path, default=PREPROCESSOR_PATH)
    parser.add_argument("--output", type=Path, default="feature_importance.png")
    args = parser.parse_args()
    evaluate(args.model, args.preprocessor, args.data_dir, args.output)
