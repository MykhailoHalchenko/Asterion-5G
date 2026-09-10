"""Input adapters for OpenSky, JSON/CSV, and CAT 062 trajectory datasets."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .asterix_decoder import Cat062Decoder
except ImportError:
    from asterix_decoder import Cat062Decoder


class DatasetParser:
    """Normalize supported dataset formats to track dictionaries."""

    def __init__(self, decoder: Cat062Decoder | None = None) -> None:
        self.decoder = decoder or Cat062Decoder()

    @staticmethod
    def _number(value: Any, name: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"field {name!r} must be numeric") from exc
        if result != result or result in (float("inf"), float("-inf")):
            raise ValueError(f"field {name!r} must be finite")
        return result

    @classmethod
    def normalize_record(cls, record: Mapping[str, Any]) -> dict[str, Any]:
        """Map common OpenSky and simulator names to one stable schema."""
        timestamp = record.get("timestamp", record.get("time", record.get("last_contact")))
        if isinstance(timestamp, datetime):
            instant = timestamp
        else:
            instant = datetime.fromtimestamp(cls._number(timestamp, "timestamp"), timezone.utc)
        if instant.tzinfo is None:
            instant = instant.replace(tzinfo=timezone.utc)
        x = record.get("x", record.get("longitude"))
        y = record.get("y", record.get("latitude"))
        if x is None or y is None:
            raise ValueError("record requires x/y or longitude/latitude")
        return {
            "timestamp": instant.astimezone(timezone.utc).timestamp(),
            "x": cls._number(x, "x"),
            "y": cls._number(y, "y"),
            "speed": cls._number(record.get("speed", record.get("velocity", 0.0)), "speed"),
            "accuracy": cls._number(record.get("accuracy", 0.0), "accuracy"),
            "source": str(record.get("source", "dataset")),
        }

    def parse_records(self, records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [self.normalize_record(record) for record in records]

    def parse_json(self, text: str) -> list[dict[str, Any]]:
        value = json.loads(text)
        if isinstance(value, Mapping):
            value = value.get("records", value.get("data", [value]))
        if not isinstance(value, list):
            raise ValueError("JSON dataset must contain a record list")
        return self.parse_records(value)

    def parse_csv(self, text: str) -> list[dict[str, Any]]:
        return self.parse_records(csv.DictReader(text.splitlines()))

    def parse_asterix(self, payload: bytes) -> list[dict[str, Any]]:
        records = []
        for record in self.decoder.iter_records(payload):
            timestamp = record.get("timestamp_seconds")
            if timestamp is None or record.get("x") is None or record.get("y") is None:
                continue
            records.append(self.normalize_record({
                "timestamp": timestamp,
                "x": record["x"],
                "y": record["y"],
                "speed": abs(record.get("vx") or 0.0),
                "accuracy": 0.0,
                "source": "asterix-cat062",
            }))
        return records

    def parse_file(self, path: str | Path) -> list[dict[str, Any]]:
        file_path = Path(path)
        payload = file_path.read_bytes()
        suffix = file_path.suffix.lower()
        if suffix == ".json":
            return self.parse_json(payload.decode("utf-8"))
        if suffix in {".csv", ".tsv"}:
            return self.parse_csv(payload.decode("utf-8"))
        return self.parse_asterix(payload)

    def iter_file(self, path: str | Path) -> Iterator[dict[str, Any]]:
        yield from self.parse_file(path)
