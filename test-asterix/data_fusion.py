"""Merge decoded OpenSky and simulated 5G trajectories."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from asterix_decoder import Cat062Decoder


class TrajectoryStitcher:
    def __init__(self, decoder: Cat062Decoder | None = None) -> None:
        self.decoder = decoder or Cat062Decoder()

    @staticmethod
    def _timestamp(record: dict[str, Any]) -> float:
        value = record.get("timestamp")
        if value is None:
            value = record.get("time")
        if value is None:
            raise ValueError("Trajectory record has no timestamp")
        return float(value)

    def merge_streams(
        self,
        opensky_bytes: bytes,
        simulated_5g_bytes: bytes,
    ) -> list[dict[str, Any]]:
        """Decode both streams, preserve source metadata, and sort by time."""
        merged: list[dict[str, Any]] = []
        for source, payload in (("opensky", opensky_bytes), ("5g", simulated_5g_bytes)):
            for record in self.decoder.iter_records(payload):
                enriched = dict(record)
                enriched["source"] = source
                # CAT 062 time-of-day is intentionally retained as a relative value.
                enriched["timestamp"] = float(record["time_of_day_raw"]) / 128.0
                merged.append(enriched)
        merged.sort(key=self._timestamp)
        return merged

    @staticmethod
    def merge_records(*streams: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        records = [dict(record) for stream in streams for record in stream]
        records.sort(key=TrajectoryStitcher._timestamp)
        return records
