"""Loading and domain-level cleaning for movement parquet files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from .config_ml import TARGET
except ImportError:
    from config_ml import TARGET


def read_parquet_files(paths: Iterable[str | Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in paths]
    if not frames:
        raise FileNotFoundError("No parquet files were supplied.")
    return pd.concat(frames, ignore_index=True, sort=False)


def load_training_data(data_dir: str | Path) -> pd.DataFrame:
    directory = Path(data_dir)
    paths = sorted(directory.glob("training_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No training_*.parquet files found in {directory}.")
    return clean_movements(read_parquet_files(paths), require_target=True)


def load_inference_data(path: str | Path) -> pd.DataFrame:
    return clean_movements(pd.read_parquet(path), require_target=False)


def fuse_with_asterix_simulation(
    original_path: str | Path,
    simulation_csv: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """Use the maintained yezhik-asterix fusion pipeline for simulated tracks."""
    import sys

    pipeline_dir = Path(__file__).resolve().parents[1] / "yezhik-asterix"
    if str(pipeline_dir) not in sys.path:
        sys.path.insert(0, str(pipeline_dir))
    from data_fusion import merge_and_export

    return merge_and_export(original_path, simulation_csv, output_path)


def clean_movements(
    frame: pd.DataFrame,
    *,
    require_target: bool = False,
) -> pd.DataFrame:
    """Remove duplicate IDs, missing timestamps and physically invalid targets."""
    result = frame.copy()
    if "MVT_ID_mvt" in result:
        result = result.drop_duplicates(subset=["MVT_ID_mvt"], keep="last")

    if require_target and TARGET not in result.columns:
        raise KeyError(f"Missing target column {TARGET!r}.")
    if TARGET in result.columns:
        result[TARGET] = pd.to_numeric(result[TARGET], errors="coerce")
        result = result[result[TARGET].isna() | (result[TARGET] > 0)]

    for column in ("BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt"):
        if column in result:
            result[column] = pd.to_datetime(result[column], utc=True, errors="coerce")
    available_time = next(
        (column for column in ("BLOCK_TIME_UTC_mvt", "MVT_TIME_UTC_mvt") if column in result),
        None,
    )
    if available_time is None:
        raise KeyError("Dataset must contain BLOCK_TIME_UTC_mvt or MVT_TIME_UTC_mvt.")
    result = result[result[available_time].notna()].reset_index(drop=True)
    return result
