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


class RelayRouter(TrackRouter):
    """Relay 5G track payloads to a downstream ASTERIX packaging stage."""

    def __init__(self, maxsize: int = 0) -> None:
        super().__init__(maxsize=maxsize)
        self._relay_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=maxsize)

    def relay(self, track: TrackEstimate, unix_timestamp: float | None = None) -> dict[str, Any]:
        payload = self.format_track(track, unix_timestamp)
        payload["source"] = "5g-rf-sensing"
        self._relay_queue.put(payload, block=True)
        return payload

    def receive_relay(self, timeout: float | None = None) -> dict[str, Any]:
        return self._relay_queue.get(block=True, timeout=timeout)

    def relay_json(self, track: TrackEstimate, unix_timestamp: float | None = None) -> str:
        return json.dumps(self.relay(track, unix_timestamp), separators=(",", ":"))