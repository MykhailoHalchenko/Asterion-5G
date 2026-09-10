"""Run the end-to-end 5G sensing demonstration."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from importlib import import_module
from pathlib import Path

rf_sensing = import_module("5Gsim.rf_sensing")
router = import_module("5Gsim.router")
map_modelling = import_module("5Gsim.map_modelling")
RFSensingSimulator = rf_sensing.RFSensingSimulator
TrajectoryPoint = rf_sensing.TrajectoryPoint
to_asterix_records = rf_sensing.to_asterix_records
TrackRouter = router.TrackRouter
plot_map = map_modelling.plot_map


def main() -> None:
    """Simulate a short aircraft track and print the router payload."""

    start = datetime.now(timezone.utc)
    trajectory = [
        TrajectoryPoint(
            timestamp=start + timedelta(milliseconds=100 * index),
            x=40.0 + index,
            y=20.0,
            z=2.0,
            speed=10.0,
        )
        for index in range(3)
    ]

    simulator = RFSensingSimulator(seed=7, noise_power=1e-4)
    tracks = simulator.simulate(trajectory)
    records = to_asterix_records(tracks)

    router = TrackRouter()
    for track in tracks:
        router.publish(track)

    routed_records = [router.receive(timeout=1.0) for _ in tracks]
    map_path = Path(__file__).resolve().parent / "airport_map.png"
    plot_map(tracks, output_path=map_path)
    print(json.dumps({
        "samples": len(routed_records),
        "raw_csi_shape": [
            simulator.config.num_ofdm_symbols,
            simulator.config.num_rx_antennas,
            simulator.config.fft_size,
        ],
        "asterix_records": records,
        "router_records": routed_records,
        "map_path": str(map_path),
    }, indent=2))


if __name__ == "__main__":
    main()