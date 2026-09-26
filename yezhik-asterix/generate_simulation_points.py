from __future__ import annotations

from pathlib import Path

import pandas as pd


def _interpolate_point(prev_lat, prev_lon, current_lat, current_lon, ratio):
    lat = prev_lat + (current_lat - prev_lat) * ratio
    lon = prev_lon + (current_lon - prev_lon) * ratio
    return lat, lon


def generate_simulation_points(
    blind_spots_csv_path,
    interval_seconds=10.0,
    output_csv_path="simulation_points.csv",
):
    df = pd.read_csv(blind_spots_csv_path)

    required = [
        "icao24",
        "previous_timestamp",
        "previous_lat",
        "previous_lon",
        "current_timestamp",
        "current_lat",
        "current_lon",
        "gap_seconds",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"CSV is missing required columns: {missing}")

    records = []
    for _, row in df.iterrows():
        prev_ts = float(row["previous_timestamp"])
        curr_ts = float(row["current_timestamp"])
        if pd.isna(prev_ts) or pd.isna(curr_ts):
            continue

        total_gap = max(float(row["gap_seconds"]) or (curr_ts - prev_ts), 1.0)
        steps = max(1, int(total_gap / max(interval_seconds, 1e-9)))

        for step in range(1, steps + 1):
            ratio = step / steps
            sim_ts = prev_ts + (curr_ts - prev_ts) * ratio
            lat, lon = _interpolate_point(
                row["previous_lat"],
                row["previous_lon"],
                row["current_lat"],
                row["current_lon"],
                ratio,
            )

            records.append(
                {
                    "icao24": row["icao24"],
                    "timestamp": sim_ts,
                    "lat": lat,
                    "lon": lon,
                    "velocity": 0.0,
                    "source": "simulated",
                    "simulated_from_previous_timestamp": prev_ts,
                    "simulated_to_current_timestamp": curr_ts,
                    "gap_seconds": float(row["gap_seconds"]),
                }
            )

    result = pd.DataFrame(records)
    if result.empty:
        result = pd.DataFrame(columns=[
            "icao24", "timestamp", "lat", "lon", "velocity", "source",
            "simulated_from_previous_timestamp", "simulated_to_current_timestamp", "gap_seconds",
        ])
    else:
        result["icao24"] = result["icao24"].astype("string")
        result = result.sort_values(["timestamp", "icao24"]).reset_index(drop=True)

    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"Згенеровано {len(result)} симуляційних точок. Збережено в CSV: {output_path}")
    return result


if __name__ == "__main__":
    generate_simulation_points("blind_spots_for_simulation.csv", interval_seconds=10.0)
