"""Bridge reconstructed 5G tracks into ASTERIX-compatible Parquet records."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def _asterix_encoder_class():
    pipeline_dir = Path(__file__).resolve().parents[1] / "yezhik-asterix"
    if str(pipeline_dir) not in sys.path:
        sys.path.insert(0, str(pipeline_dir))
    return importlib.import_module("asterix_wrapper").AsterixEncoder


def local_xy_to_wgs84(
    x: float,
    y: float,
    *,
    origin_latitude: float = 50.45,
    origin_longitude: float = 30.52,
) -> tuple[float, float]:
    """Convert local simulator metres to an approximate WGS-84 position."""
    latitude = origin_latitude + float(y) / 111_320.0
    longitude_scale = 111_320.0 * max(0.1, abs(math.cos(math.radians(origin_latitude))))
    longitude = origin_longitude + float(x) / longitude_scale
    return latitude, longitude


def track_to_asterix_row(
    track: Any,
    *,
    track_number: int,
    origin_latitude: float = 50.45,
    origin_longitude: float = 30.52,
) -> dict[str, Any]:
    """Build one tabular record from a 5G ``TrackEstimate``."""
    timestamp = track.timestamp.astimezone(timezone.utc).timestamp()
    latitude, longitude = local_xy_to_wgs84(
        track.x,
        track.y,
        origin_latitude=origin_latitude,
        origin_longitude=origin_longitude,
    )
    vx = float(track.speed) * float(math.cos(track.azimuth_rad))
    vy = float(track.speed) * float(math.sin(track.azimuth_rad))
    encoder = _asterix_encoder_class()
    cat062_hex = encoder().encode(
        lat=latitude,
        lon=longitude,
        speed=track.speed,
        vx=vx,
        vy=vy,
        timestamp=timestamp,
        track_number=track_number,
    )
    return {
        "icao24": f"5g-{track_number:06d}",
        "timestamp": timestamp,
        "lat": latitude,
        "lon": longitude,
        "velocity": float(track.speed),
        "accuracy": float(track.accuracy_m),
        "range_m": float(track.range_m),
        "azimuth_rad": float(track.azimuth_rad),
        "elevation_rad": float(track.elevation_rad),
        "source": "5g-rf-sensing",
        "cat062_hex": cat062_hex,
    }


def export_tracks(
    tracks: list[Any],
    output_path: str | Path,
    *,
    origin_latitude: float = 50.45,
    origin_longitude: float = 30.52,
) -> pd.DataFrame:
    """Encode tracks as CAT 062 and save the combined table as Parquet."""
    rows = [
        track_to_asterix_row(
            track,
            track_number=index,
            origin_latitude=origin_latitude,
            origin_longitude=origin_longitude,
        )
        for index, track in enumerate(tracks, start=1)
    ]
    result = pd.DataFrame(rows)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(destination, index=False)
    return result


def simulate_and_export(
    samples: int,
    output_path: str | Path,
    *,
    seed: int = 7,
    interval_seconds: float = 0.1,
) -> pd.DataFrame:
    """Run the 5G simulator for a finite batch and export ASTERIX records."""
    if samples < 1:
        raise ValueError("samples must be at least 1")
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")

    rf_sensing = importlib.import_module("5Gsim.rf_sensing")
    simulator = rf_sensing.RFSensingSimulator(seed=seed, noise_power=1e-4)
    tracks = []
    for index in range(samples):
        point = rf_sensing.TrajectoryPoint(
            timestamp=datetime.now(timezone.utc),
            x=40.0 + 10.0 * index * interval_seconds,
            y=20.0 + 2.0 * index * interval_seconds,
            z=2.0,
            speed=10.0,
        )
        tracks.append(simulator.process(point))
    return export_tracks(tracks, output_path)
