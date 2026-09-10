"""Run the end-to-end 5G sensing demonstration."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
import time

rf_sensing = import_module("5Gsim.rf_sensing")
router = import_module("5Gsim.router")
RFSensingSimulator = rf_sensing.RFSensingSimulator
TrajectoryPoint = rf_sensing.TrajectoryPoint
to_asterix_records = rf_sensing.to_asterix_records
RelayRouter = router.RelayRouter
map_modelling = import_module("5Gsim.map_modelling")
plot_signals = map_modelling.plot_signals


def main() -> None:
    """Run continuously until the operator presses Ctrl+C."""

    simulator = RFSensingSimulator(seed=7, noise_power=1e-4)
    router = RelayRouter()
    tracks = []
    sample_index = 0
    root = Path(__file__).resolve().parent
    logs_dir = root / "output-logs"
    graphs_dir = root / "graphs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "rf-sensing.json"
    graph_path = graphs_dir / "signals.png"
    log_records: list[dict[str, object]] = []
    log_path.write_text("[]\n", encoding="utf-8")
    print(f"RF outputs: {log_path}", flush=True)
    print(f"Signal graph: {graph_path}", flush=True)
    try:
        import drjit as dr
        import matplotlib.pyplot as plt
        plt.ion()
        signal_figure = None
        while True:
            timestamp = datetime.now(timezone.utc)
            phase = sample_index * 0.1
            point = TrajectoryPoint(
                timestamp=timestamp,
                x=40.0 + 10.0 * phase,
                y=20.0 + 2.0 * phase,
                z=2.0,
                speed=10.0,
            )
            track = simulator.process(point)
            routed = router.relay(track)
            router.receive_relay(timeout=1.0)
            tracks.append(track)
            tracks = tracks[-100:]
            signal_figure = plot_signals(
                tracks,
                figure=signal_figure,
                output_path=graph_path,
            )
            plt.show(block=False)
            # Keep a device-compatible Dr.Jit scalar in the simulation loop.
            dr_metric = dr.square(dr.scalar.Float(track.speed))
            dr.eval(dr_metric)
            record = {
                "sample": sample_index,
                "track": routed,
                "drjit_speed_squared": float(dr_metric),
            }
            log_records.append(record)
            log_path.write_text(
                json.dumps(log_records, ensure_ascii=True, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(record), flush=True)
            sample_index += 1
            time.sleep(0.1)
    except KeyboardInterrupt:
        log_records.append({
            "event": "stopped",
            "samples": sample_index,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        log_path.write_text(
            json.dumps(log_records, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nStopped by Ctrl+C after {sample_index} samples.")


if __name__ == "__main__":
    main()