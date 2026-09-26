import pandas as pd

from data_fusion import merge_and_export
from find_blind_spots import find_gaps_in_tracks
from generate_simulation_points import generate_simulation_points


def test_find_gaps_in_tracks_detects_a_gap(tmp_path):
    parquet_path = tmp_path / "opensky_gap.parquet"
    pd.DataFrame(
        [
            {"icao24": "a1b2c3", "timestamp": 1_700_000_000, "lat": 50.45, "lon": 30.52, "velocity": 10.5, "onground": True},
            {"icao24": "a1b2c3", "timestamp": 1_700_000_010, "lat": 50.46, "lon": 30.53, "velocity": 10.5, "onground": True},
            {"icao24": "a1b2c3", "timestamp": 1_700_001_000, "lat": 50.50, "lon": 30.60, "velocity": 10.5, "onground": True},
            {"icao24": "a1b2c3", "timestamp": 1_700_001_010, "lat": 50.51, "lon": 30.61, "velocity": 10.5, "onground": True},
        ]
    ).to_parquet(parquet_path, index=False)

    result = find_gaps_in_tracks(parquet_path, gap_threshold_seconds=30.0)

    assert len(result) == 1
    assert result["icao24"].iloc[0] == "a1b2c3"
    assert result["gap_seconds"].iloc[0] > 30
    assert result["previous_lat"].iloc[0] == 50.46
    assert result["current_lat"].iloc[0] == 50.5


def test_generate_simulation_points_creates_intermediate_rows(tmp_path):
    blind_spots_path = tmp_path / "blind_spots.csv"
    pd.DataFrame(
        [
            {
                "icao24": "a1b2c3",
                "previous_timestamp": 1_700_000_000,
                "previous_lat": 50.45,
                "previous_lon": 30.52,
                "current_timestamp": 1_700_000_030,
                "current_lat": 50.50,
                "current_lon": 30.60,
                "gap_seconds": 30.0,
            }
        ]
    ).to_csv(blind_spots_path, index=False)

    points = generate_simulation_points(blind_spots_path, interval_seconds=10.0)

    assert len(points) >= 2
    assert points["icao24"].iloc[0] == "a1b2c3"
    assert points["source"].iloc[0] == "simulated"
    assert points["lat"].iloc[0] > 50.45
    assert points["lat"].iloc[0] < 50.50


def test_find_gaps_in_tracks_detects_equal_threshold_value(tmp_path):
    parquet_path = tmp_path / "equal_gap.parquet"
    pd.DataFrame(
        [
            {"icao24": "a1b2c3", "timestamp": 1_700_000_000, "lat": 50.45, "lon": 30.52},
            {"icao24": "a1b2c3", "timestamp": 1_700_000_009, "lat": 50.46, "lon": 30.53},
        ]
    ).to_parquet(parquet_path, index=False)

    result = find_gaps_in_tracks(parquet_path, gap_threshold_seconds=9.0)

    assert len(result) == 0


def test_find_gaps_in_tracks_detects_gap_above_threshold(tmp_path):
    parquet_path = tmp_path / "above_threshold_gap.parquet"
    pd.DataFrame(
        [
            {"icao24": "a1b2c3", "timestamp": 1_700_000_000, "lat": 50.45, "lon": 30.52},
            {"icao24": "a1b2c3", "timestamp": 1_700_000_010, "lat": 50.46, "lon": 30.53},
        ]
    ).to_parquet(parquet_path, index=False)

    result = find_gaps_in_tracks(parquet_path, gap_threshold_seconds=9.0)

    assert len(result) == 1
    assert result["gap_seconds"].iloc[0] == 10.0


def test_find_gaps_in_tracks_ignores_non_positive_gap(tmp_path):
    parquet_path = tmp_path / "non_positive_gap.parquet"
    pd.DataFrame(
        [
            {"icao24": "a1b2c3", "timestamp": 1_700_000_010, "lat": 50.45, "lon": 30.52},
            {"icao24": "a1b2c3", "timestamp": 1_700_000_010, "lat": 50.46, "lon": 30.53},
        ]
    ).to_parquet(parquet_path, index=False)

    result = find_gaps_in_tracks(parquet_path, gap_threshold_seconds=1.0)

    assert len(result) == 0


def test_merge_and_export_adds_simulated_rows_and_parquet(tmp_path):
    original_path = tmp_path / "original.parquet"
    simulated_path = tmp_path / "simulated.csv"
    output_path = tmp_path / "fused.parquet"

    pd.DataFrame(
        [
            {"icao24": "a1b2c3", "timestamp": 1_700_000_000, "lat": 50.45, "lon": 30.52, "velocity": 10.5, "source": "opensky"},
            {"icao24": "a1b2c3", "timestamp": 1_700_000_010, "lat": 50.46, "lon": 30.53, "velocity": 10.7, "source": "opensky"},
        ]
    ).to_parquet(original_path, index=False)

    pd.DataFrame(
        [
            {"icao24": "a1b2c3", "timestamp": 1_700_000_005, "lat": 50.455, "lon": 30.525, "velocity": 11.0, "source": "simulated"},
        ]
    ).to_csv(simulated_path, index=False)

    merged = merge_and_export(original_path, simulated_path, output_path)

    assert output_path.exists()
    assert len(merged) == 3
    assert set(merged["source"].unique()) == {"opensky", "simulated"}
    assert merged["cat062_hex"].str.len().gt(0).all()


def test_merge_and_export_keeps_original_when_timestamp_is_duplicated(tmp_path):
    original_path = tmp_path / "original.parquet"
    simulated_path = tmp_path / "simulated.csv"
    output_path = tmp_path / "fused.parquet"

    pd.DataFrame(
        [
            {"icao24": "a1b2c3", "timestamp": 1_700_000_010, "lat": 50.46, "lon": 30.53, "source": "opensky"},
        ]
    ).to_parquet(original_path, index=False)
    pd.DataFrame(
        [
            {"icao24": "a1b2c3", "timestamp": 1_700_000_010, "lat": 50.99, "lon": 30.99, "source": "simulated"},
        ]
    ).to_csv(simulated_path, index=False)

    merged = merge_and_export(original_path, simulated_path, output_path)

    assert len(merged) == 1
    assert merged.iloc[0]["source"] == "opensky"
    assert merged.iloc[0]["lat"] == 50.46


def test_merge_and_export_sorts_by_timestamp_then_icao24(tmp_path):
    original_path = tmp_path / "original.parquet"
    simulated_path = tmp_path / "simulated.csv"
    output_path = tmp_path / "fused.parquet"

    pd.DataFrame(
        [
            {"icao24": "b2", "timestamp": 20, "lat": 50.0, "lon": 30.0, "source": "opensky"},
            {"icao24": "z9", "timestamp": 10, "lat": 50.0, "lon": 30.0, "source": "opensky"},
        ]
    ).to_parquet(original_path, index=False)
    pd.DataFrame(
        [
            {"icao24": "a1", "timestamp": 10, "lat": 50.0, "lon": 30.0, "source": "simulated"},
        ]
    ).to_csv(simulated_path, index=False)

    merged = merge_and_export(original_path, simulated_path, output_path)

    assert list(zip(merged["timestamp"], merged["icao24"])) == [(10, "a1"), (10, "z9"), (20, "b2")]
