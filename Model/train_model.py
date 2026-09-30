"""Train and save the TensorFlow taxi-time regressor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

try:
    from .config_ml import DEFAULT_MODEL_PARAMS, DATA_DIR, MODEL_PATH, PREPROCESSOR_PATH, TARGET
    from .data_loader import load_training_data
    from .feature_engineering import build_features, fit_encoder
except ImportError:
    from config_ml import DEFAULT_MODEL_PARAMS, DATA_DIR, MODEL_PATH, PREPROCESSOR_PATH, TARGET
    from data_loader import load_training_data
    from feature_engineering import build_features, fit_encoder


def train(data_dir: str | Path = DATA_DIR, model_path: str | Path = MODEL_PATH,
          preprocessor_path: str | Path = PREPROCESSOR_PATH) -> dict[str, float]:
    data = load_training_data(data_dir).sort_values("MVT_TIME_UTC_mvt")
    split = max(1, int(len(data) * 0.8))
    train_data, validation_data = data.iloc[:split], data.iloc[split:]
    x_train, state = fit_encoder(build_features(train_data))
    x_validation = _transform(build_features(validation_data), state)
    y_train = train_data[TARGET].to_numpy(dtype=np.float32)
    y_validation = validation_data[TARGET].to_numpy(dtype=np.float32)

    tf.keras.utils.set_random_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(x_train.shape[1],)),
        tf.keras.layers.Dense(DEFAULT_MODEL_PARAMS["hidden_units"][0], activation="relu"),
        tf.keras.layers.Dropout(DEFAULT_MODEL_PARAMS["dropout"]),
        tf.keras.layers.Dense(DEFAULT_MODEL_PARAMS["hidden_units"][1], activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(DEFAULT_MODEL_PARAMS["learning_rate"]),
        loss=tf.keras.losses.Huber(),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"), tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_mae", patience=DEFAULT_MODEL_PARAMS["patience"], restore_best_weights=True
        )
    ]
    model.fit(
        x_train, y_train, validation_data=(x_validation, y_validation),
        epochs=DEFAULT_MODEL_PARAMS["epochs"], batch_size=DEFAULT_MODEL_PARAMS["batch_size"],
        callbacks=callbacks, verbose=2,
    )
    model.save(model_path)
    Path(preprocessor_path).write_text(json.dumps(state, indent=2), encoding="utf-8")
    metrics = {key: float(value) for key, value in model.evaluate(x_validation, y_validation, verbose=0, return_dict=True).items()}
    print(json.dumps(metrics, indent=2))
    return metrics


def _transform(features, state):
    try:
        from .feature_engineering import transform_features
    except ImportError:
        from feature_engineering import transform_features
    return transform_features(features, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--preprocessor", type=Path, default=PREPROCESSOR_PATH)
    args = parser.parse_args()
    train(args.data_dir, args.model, args.preprocessor)
