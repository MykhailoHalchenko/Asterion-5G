from __future__ import annotations

import argparse
from pathlib import Path

from data_fusion import merge_and_export
from find_blind_spots import _normalize_columns, find_gaps_in_tracks
from generate_simulation_points import generate_simulation_points
from taxiway_tracks import build_pair_profiles, generate_ground_tracks


def parse_args():
    parser = argparse.ArgumentParser(description="Offline ASTERIX training Parquet pipeline")
    parser.add_argument("--input", default="training_2025-01-01_2025-02-01.parquet", help="Training Parquet with flight records")
    parser.add_argument("--threshold", type=float, default=1.0, help="Gap threshold in seconds")
    parser.add_argument("--interval", type=float, default=300.0, help="Simulation point interval in seconds")
    parser.add_argument("--blind-spots-output", default="blind_spots_for_simulation.csv", help="Intermediate blind spot CSV path")
    parser.add_argument("--simulation-output", default="simulation_points.csv", help="Intermediate simulation CSV path")
    parser.add_argument("--fused-output", default="fused_dataset.parquet", help="Final Parquet output")
    parser.add_argument("--ground-tracks", action="store_true", help="Generate OSM taxiway tracks for one airport")
    parser.add_argument("--airport-icao", default="EDDF", help="ICAO code of the airport to simulate")
    parser.add_argument("--airport-place", default="Frankfurt Airport, Germany", help="OSM place query for that airport")
    parser.add_argument("--pair-profiles-output", default="taxi_pair_profiles.csv", help="Mean taxi times and virtual distances")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.ground_tracks:
        import pandas as pd

        movements = pd.read_parquet(args.input)
        airport = args.airport_icao.upper()
        local_airport = movements["ADEP_mvt"].where(
            movements["PHASE_mvt"].eq("DEP"), movements["ADES_mvt"]
        )
        airport_movements = movements[local_airport.eq(airport)]
        profiles = build_pair_profiles(airport_movements)
        profiles.to_csv(args.pair_profiles_output, index=False)
        print(f"Stand-runway profiles saved to: {args.pair_profiles_output}")

        simulation = generate_ground_tracks(
            args.input,
            airport_icao=airport,
            airport_place=args.airport_place,
            output_csv_path=args.simulation_output,
            interval_seconds=args.interval,
        )
        if not simulation.empty:
            merge_and_export(args.input, args.simulation_output, args.fused_output)
        else:
            raise RuntimeError(
                "OSM tracks were not generated. Check airport identifiers, OSM feature refs, and route diagnostics."
            )
        return

    blind = find_gaps_in_tracks(
        args.input,
        gap_threshold_seconds=args.threshold,
        output_csv_path=args.blind_spots_output,
    )

    if blind.empty:
        print("No large gaps found. Creating an empty simulation dataset.")
        simulation = []
    else:
        simulation = generate_simulation_points(
            args.blind_spots_output,
            interval_seconds=args.interval,
            output_csv_path=args.simulation_output,
        )

    if not simulation.empty:
        merge_and_export(
            original_parquet_path=args.input,
            simulated_csv_path=args.simulation_output,
            output_parquet_path=args.fused_output,
        )
    else:
        # still write empty parquet file with columns as fallback
        import pandas as pd

        empty = _normalize_columns(pd.read_parquet(args.input))
        if "timestamp" in empty.columns and "icao24" in empty.columns:
            empty["timestamp"] = pd.to_numeric(empty["timestamp"], errors="coerce")
            empty["icao24"] = empty["icao24"].astype("string")
            empty = empty.sort_values(["timestamp", "icao24"]).reset_index(drop=True)
        empty["source"] = "opensky"
        empty["cat062_hex"] = ""
        empty.to_parquet(args.fused_output, index=False)
        print(f"Empty fused dataset saved to: {args.fused_output}")


if __name__ == "__main__":
    main()
