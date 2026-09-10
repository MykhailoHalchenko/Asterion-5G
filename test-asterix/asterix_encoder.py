"""ASTERIX CAT 062 encoder backed by the installed libasterix schema."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from asterix.generated import Cat_062_1_20

__all__ = ["Cat062Encoder"]


class Cat062Encoder:
    """Encode reconstructed Cartesian tracks as CAT 062 records."""

    Spec = Cat_062_1_20

    @staticmethod
    def _record_values(data: Mapping[str, Any]) -> dict[str, Any]:
        timestamp = float(data["timestamp"])
        # CAT 062 item 070 is seconds from midnight, not UNIX time.
        seconds_of_day = timestamp % 86_400.0
        values: dict[str, Any] = {
            "010": (
                int(data.get("sac", 0)),
                int(data.get("sic", 62)),
            ),
            "070": seconds_of_day,
            "100": (float(data["x"]), float(data["y"])),
        }
        if "vx" in data and "vy" in data:
            values["185"] = (float(data["vx"]), float(data["vy"]))
        elif "speed" in data:
            values["185"] = (float(data["speed"]), 0.0)
        return values

    def create_record(self, data_dict: Mapping[str, Any]) -> Any:
        """Create one generated CAT 062 record from a track mapping."""
        required = {"timestamp", "x", "y"}
        missing = required.difference(data_dict)
        if missing:
            raise ValueError(f"CAT 062 track is missing fields: {sorted(missing)}")
        return self.Spec.cv_record.create(self._record_values(data_dict))

    def encode_datablock(self, records_list: Sequence[Any]) -> bytes:
        """Encode generated records into one ASTERIX data block."""
        if not records_list:
            return b""
        return self.Spec.create(list(records_list)).unparse().to_bytes()

    def encode_tracks(self, tracks: Sequence[Mapping[str, Any]]) -> bytes:
        return self.encode_datablock([self.create_record(track) for track in tracks])

    def encode_record(self, track: Mapping[str, Any]) -> bytes:
        """Encode one track without requiring callers to build Records."""
        return self.encode_datablock([self.create_record(track)])

    def encode_stream(
        self,
        tracks: Iterable[Mapping[str, Any]],
        batch_size: int = 128,
    ) -> bytes:
        """Encode an iterator in bounded batches into a concatenated stream."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        encoded: list[bytes] = []
        batch: list[Mapping[str, Any]] = []
        for track in tracks:
            batch.append(track)
            if len(batch) == batch_size:
                encoded.append(self.encode_tracks(batch))
                batch.clear()
        if batch:
            encoded.append(self.encode_tracks(batch))
        return b"".join(encoded)
