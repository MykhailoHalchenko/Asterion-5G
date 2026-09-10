"""CAT 062 stream decoder using libasterix generated specifications."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from asterix.base import Bits, RawDatablock
from asterix.generated import Cat_062_1_20


class Cat062Decoder:
    Spec = Cat_062_1_20

    def parse_stream(self, rx_bytes: bytes) -> list[RawDatablock]:
        if not rx_bytes:
            return []
        blocks = RawDatablock.parse(Bits.from_bytes(rx_bytes))
        if isinstance(blocks, ValueError):
            raise ValueError(f"Unable to parse ASTERIX stream: {blocks}") from blocks
        return blocks

    def extract_records(self, db: RawDatablock) -> list[Any]:
        if db.get_category() != 62:
            return []
        parsed = self.Spec.cv_uap.parse(db.get_raw_records())
        if isinstance(parsed, ValueError):
            raise ValueError(f"Unable to parse CAT 062 records: {parsed}") from parsed
        return parsed

    @staticmethod
    def _nested_uint(record: Any, item: str, component: str) -> int | None:
        try:
            value = record.get_item(item)
            variation = value.arg.variation
            return variation.get_item(component).as_uint()
        except (AttributeError, KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _signed(value: int | None, bits: int) -> float | None:
        if value is None:
            return None
        if value & (1 << (bits - 1)):
            value -= 1 << bits
        return float(value)

    def get_physical_values(self, rec: Any) -> dict[str, Any]:
        """Return decoded values and raw ASTERIX integers.

        libasterix exposes component-level encoded values through ``as_uint``;
        retaining those values makes the decoder lossless even for optional
        or edition-specific fields.
        """
        values: dict[str, Any] = {}
        try:
            values["category"] = 62
            values["sac"] = self._nested_uint(rec, "010", "SAC")
            values["sic"] = self._nested_uint(rec, "010", "SIC")
            values["x_raw"] = self._nested_uint(rec, "100", "X")
            values["y_raw"] = self._nested_uint(rec, "100", "Y")
            values["vx_raw"] = self._nested_uint(rec, "185", "VX")
            values["vy_raw"] = self._nested_uint(rec, "185", "VY")
            time_item = rec.get_item("070")
            values["time_of_day_raw"] = time_item.as_uint()
            x = self._signed(values["x_raw"], 24)
            y = self._signed(values["y_raw"], 24)
            vx = self._signed(values["vx_raw"], 16)
            vy = self._signed(values["vy_raw"], 16)
            values["x"] = None if x is None else x / 2.0
            values["y"] = None if y is None else y / 2.0
            values["vx"] = None if vx is None else vx / 4.0
            values["vy"] = None if vy is None else vy / 4.0
            values["timestamp_seconds"] = values["time_of_day_raw"] / 128.0
        except (AttributeError, KeyError, TypeError, ValueError):
            values["category"] = 62
        return values

    def iter_records(self, rx_bytes: bytes) -> Iterator[dict[str, Any]]:
        for block in self.parse_stream(rx_bytes):
            for record in self.extract_records(block):
                yield self.get_physical_values(record)
