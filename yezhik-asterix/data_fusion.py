from __future__ import annotations

from pathlib import Path

import pandas as pd

from asterix_wrapper import AsterixEncoder


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.rename(columns=lambda column: str(column).strip().lower())

    if "icao24" not in normalized.columns and "icao" in normalized.columns:
        normalized["icao24"] = normalized["icao"]
    if "icao24" not in normalized.columns:
        for column in ("flight_id_mvt", "callsign_flt", "flight_mvt"):
            if column in normalized.columns:
                normalized["icao24"] = normalized[column]
                break
        else:
            raise KeyError("Dataset does not contain an aircraft or flight identifier.")
    if "timestamp" not in normalized.columns:
        if "mvt_time_utc_mvt" in normalized.columns:
            normalized["timestamp"] = normalized["mvt_time_utc_mvt"]
        elif "time_position" in normalized.columns:
            normalized["timestamp"] = normalized["time_position"]
        else:
            raise KeyError("Dataset is missing a timestamp field.")
    if "lat" not in normalized.columns:
        normalized["lat"] = normalized.get("latitude", pd.NA)
    if "lon" not in normalized.columns:
        normalized["lon"] = normalized.get("longitude", pd.NA)

    if "source" not in normalized.columns:
        normalized["source"] = "unknown"

    return normalized


def merge_and_export(original_parquet_path, simulated_csv_path, output_parquet_path, encoder=None):
    original_df = _normalize_columns(pd.read_parquet(original_parquet_path))
    simulated_df = _normalize_columns(pd.read_csv(simulated_csv_path))

    original_df["source"] = original_df["source"].fillna("opensky").astype(str)
    simulated_df["source"] = simulated_df["source"].fillna("simulated").astype(str)

    for frame in (original_df, simulated_df):
        if pd.api.types.is_datetime64_any_dtype(frame["timestamp"]):
            timestamp = pd.to_datetime(frame["timestamp"], utc=True)
            epoch = pd.Timestamp("1970-01-01", tz="UTC")
            frame["timestamp"] = (timestamp - epoch).dt.total_seconds()
        else:
            frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    original_df["icao24"] = original_df["icao24"].astype("string")
    simulated_df["icao24"] = simulated_df["icao24"].astype("string")

    # Original observations take precedence over simulated rows with the same key.
    fused = pd.concat([original_df, simulated_df], ignore_index=True, sort=False)
    fused = fused.drop_duplicates(subset=["icao24", "timestamp"], keep="first")
    fused = fused.sort_values(["timestamp", "icao24"]).reset_index(drop=True)

    encoder = encoder or AsterixEncoder()
    fused["cat062_hex"] = fused.apply(lambda row: encoder.encode_row(row), axis=1)

    output_path = Path(output_parquet_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fused.to_parquet(output_path, index=False)

    print(f"Dataset successfully fused. Saved to {output_path}")
    return fused


if __name__ == "__main__":
    merge_and_export("training_2025-01-01_2025-02-01.parquet", "simulation_points.csv", "fused_dataset.parquet")