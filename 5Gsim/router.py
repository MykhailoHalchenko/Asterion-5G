"""Timestamped hand-off queue for ASTERIX CAT 062 packaging."""

from __future__ import annotations

import json
import queue
from datetime import timezone
from typing import Any

try:
    from .signal_postprocessing import TrackEstimate
except ImportError:
    from signal_postprocessing import TrackEstimate


class TrackRouter:
    def __init__(self, maxsize: int = 0):
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=maxsize)

    @staticmethod
    def format_track(track: TrackEstimate, unix_timestamp: float | None = None) -> dict[str, Any]:
        timestamp = (
            unix_timestamp
            if unix_timestamp is not None
            else track.timestamp.astimezone(timezone.utc).timestamp()
        )
        return {
            "timestamp": float(timestamp),
            "x": track.x,
            "y": track.y,
            "speed": track.speed,
            "accuracy": track.accuracy_m,
            "range_m": track.range_m,
            "azimuth_rad": track.azimuth_rad,
            "elevation_rad": track.elevation_rad,
        }

    def publish(self, track: TrackEstimate, unix_timestamp: float | None = None) -> None:
        self._queue.put(self.format_track(track, unix_timestamp), block=True)

    def receive(self, timeout: float | None = None) -> dict[str, Any]:
        return self._queue.get(block=True, timeout=timeout)

    def to_json(self, track: TrackEstimate, unix_timestamp: float | None = None) -> str:
        return json.dumps(self.format_track(track, unix_timestamp), separators=(",", ":"))