"""Sionna-compatible stochastic mmWave RF-sensing simulation core."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TypedDict

import numpy as np

try:
    from .mimo_3d import AircraftState, MIMO3DChannel
    from .config import RadioConfig
    from .signal_postprocessing import SignalPostProcessor, TrackEstimate
except ImportError:
    from mimo_3d import AircraftState, MIMO3DChannel
    from config import RadioConfig
    from signal_postprocessing import SignalPostProcessor, TrackEstimate


class AsterixTrackRecord(TypedDict):
    timestamp: float
    x: float
    y: float
    speed: float
    accuracy: float


@dataclass(frozen=True)
class TrajectoryPoint:
    timestamp: datetime
    x: float
    y: float
    speed: float
    z: float = 2.0


class gNodeB:
    """Generate pilot reflections and reconstruct aircraft tracks."""

    def __init__(
        self,
        config: RadioConfig | None = None,
        seed: int | None = 7,
        noise_power: float = 1e-3,
    ):
        self.config = config or RadioConfig()
        self.noise_power = noise_power
        self.channel = MIMO3DChannel(self.config, seed=seed)
        self.postprocessor = SignalPostProcessor(self.config)

    def reference_signals(self) -> np.ndarray:
        return np.ones(
            (
                self.config.num_ofdm_symbols,
                self.config.num_tx_antennas,
                self.config.fft_size,
            ),
            dtype=np.complex128,
        )

    def sense(self, point: TrajectoryPoint) -> tuple[np.ndarray, np.ndarray]:
        state = AircraftState(
            position=(point.x, point.y, point.z),
            velocity=(point.speed, 0.0, 0.0),
        )
        return self.channel.generate(
            state,
            num_symbols=self.config.num_ofdm_symbols,
            noise_power=self.noise_power,
        )

    def process(self, point: TrajectoryPoint) -> TrackEstimate:
        _, received = self.sense(point)
        return self.postprocessor.reconstruct(
            received,
            point.timestamp,
            antenna_positions=self.channel.rx_positions,
        )


class RFSensingSimulator(gNodeB):
    """Compatibility name for callers using the previous simulator API."""

    def simulate(self, trajectory: list[TrajectoryPoint]) -> list[TrackEstimate]:
        return [self.process(point) for point in trajectory]


def to_asterix_record(track: TrackEstimate) -> AsterixTrackRecord:
    return {
        "timestamp": track.timestamp.astimezone(timezone.utc).timestamp(),
        "x": track.x,
        "y": track.y,
        "speed": track.speed,
        "accuracy": track.accuracy_m,
    }


def to_asterix_records(tracks: list[TrackEstimate]) -> list[AsterixTrackRecord]:
    return [to_asterix_record(track) for track in tracks]
