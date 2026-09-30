from __future__ import annotations

from pathlib import Path

import pandas as pd


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.rename(columns=lambda column: str(column).strip().lower())

    if "timestamp" not in normalized.columns:
        if "mvt_time_utc_mvt" in normalized.columns:
            normalized["timestamp"] = normalized["mvt_time_utc_mvt"]
        elif "time_position" in normalized.columns:
            normalized["timestamp"] = normalized["time_position"]
        elif "time" in normalized.columns:
            normalized["timestamp"] = normalized["time"]
        else:
            raise KeyError("Parquet does not contain a timestamp field.")

    if "icao24" not in normalized.columns and "icao" in normalized.columns:
        normalized["icao24"] = normalized["icao"]
    if "icao24" not in normalized.columns:
        for column in ("flight_id_mvt", "callsign_flt", "flight_mvt"):
            if column in normalized.columns:
                normalized["icao24"] = normalized[column]
                break
        else:
            raise KeyError("Dataset does not contain an aircraft or flight identifier.")

    if "lat" not in normalized.columns:
        if "latitude" in normalized.columns:
            normalized["lat"] = normalized["latitude"]
        else:
            normalized["lat"] = pd.NA

    if "lon" not in normalized.columns:
        if "longitude" in normalized.columns:
            normalized["lon"] = normalized["longitude"]
        else:
            normalized["lon"] = pd.NA

    if "onground" not in normalized.columns and "on_ground" in normalized.columns:
        normalized["onground"] = normalized["on_ground"]

    return normalized


def find_gaps_in_tracks(input_parquet_path, gap_threshold_seconds=5.0, output_csv_path="blind_spots_for_simulation.csv"):
    df = pd.read_parquet(input_parquet_path)
    df = _normalize_columns(df)

    df["icao24"] = df["icao24"].astype("string")
    if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        timestamp = pd.to_datetime(df["timestamp"], utc=True)
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        df["timestamp"] = (timestamp - epoch).dt.total_seconds()
    else:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.sort_values(["icao24", "timestamp"]).copy()

    df["prev_timestamp"] = df.groupby("icao24")["timestamp"].shift(1)
    df["prev_lat"] = df.groupby("icao24")["lat"].shift(1)
    df["prev_lon"] = df.groupby("icao24")["lon"].shift(1)
    df["gap_seconds"] = df["timestamp"] - df["prev_timestamp"]

    blind_spots = df[
        (df["prev_timestamp"].notna())
        & (df["gap_seconds"] > 0)
        & (df["gap_seconds"] > gap_threshold_seconds)
    ].copy()

    final_blind_spots = blind_spots[
        [
            "icao24",
            "prev_timestamp",
            "prev_lat",
            "prev_lon",
            "timestamp",
            "lat",
            "lon",
            "gap_seconds",
        ]
    ].rename(
        columns={
            "prev_timestamp": "previous_timestamp",
            "prev_lat": "previous_lat",
            "prev_lon": "previous_lon",
            "timestamp": "current_timestamp",
            "lat": "current_lat",
            "lon": "current_lon",
        }
    ).sort_values(["current_timestamp", "icao24"]).reset_index(drop=True)

    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_blind_spots.to_csv(output_path, index=False)

    print(f"Found {len(final_blind_spots)} blind spots. Exported to CSV: {output_path}")
    return final_blind_spots


if __name__ == "__main__":
    find_gaps_in_tracks("training_2025-01-01_2025-02-01.parquet", 5.0)