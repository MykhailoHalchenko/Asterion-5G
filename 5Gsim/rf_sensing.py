"""Sionna-compatible stochastic mmWave RF-sensing simulation core."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TypedDict

try:
    from .config import RadioConfig
    from .gNodeB import gNodeB
    from .signal_postprocessing import TrackEstimate
except ImportError:
    from config import RadioConfig
    from gNodeB import gNodeB
    from signal_postprocessing import TrackEstimate


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
