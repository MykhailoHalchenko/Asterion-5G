from __future__ import annotations

import math
from typing import Any, Optional


class AsterixEncoder:
    """Create a valid CAT 062 envelope using the installed libasterix generated classes."""

    def __init__(self):
        try:
            from asterix.generated import Cat_062_1_20, Record_51
        except ImportError as exc:  # pragma: no cover - dependency issue
            raise RuntimeError(
                "The asterix Python package is not available. Install ast-tool-py or libasterix."
            ) from exc

        self.Category = Cat_062_1_20
        self.Record = Record_51

    @staticmethod
    def _has_number(value: Any) -> bool:
        if value is None:
            return False
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    def encode(
        self,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        alt: Optional[float] = None,
        speed: Optional[float] = None,
        vx: Optional[float] = None,
        vy: Optional[float] = None,
        timestamp: Optional[float] = None,
        *,
        icao24: Optional[str] = None,
        track_number: Optional[int] = None,
        data_source_id: tuple[int, int] = (0, 0),
        service_id: int = 0,
    ) -> str:
        """Return an ASTERIX CAT 062 edition 1.20 record as a hex string."""
        record_args: dict[str, Any] = {"010": data_source_id, "015": service_id}

        if self._has_number(track_number):
            record_args["040"] = int(track_number) % 65_536
        has_lat = self._has_number(lat)
        has_lon = self._has_number(lon)
        if has_lat or has_lon:
            if not has_lat or not has_lon:
                raise ValueError("CAT 062 WGS-84 position requires both latitude and longitude.")
            record_args["105"] = (("LAT", float(lat)), ("LON", float(lon)))
        has_vx = self._has_number(vx)
        has_vy = self._has_number(vy)
        if has_vx or has_vy:
            if not has_vx or not has_vy:
                raise ValueError("CAT 062 Cartesian velocity requires both vx and vy.")
            record_args["185"] = (("VX", float(vx)), ("VY", float(vy)))
        if self._has_number(timestamp):
            from datetime import datetime, timezone

            time_of_day = datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
            seconds = (
                time_of_day.hour * 3600
                + time_of_day.minute * 60
                + time_of_day.second
                + time_of_day.microsecond / 1_000_000
            )
            record_args["070"] = seconds

        record = self.Record.create(record_args)
        category = self.Category.create([record])
        return category.unparse().to_bytes().hex()

    def encode_row(self, row: dict[str, Any]) -> str:
        return self.encode(
            lat=row.get("lat"),
            lon=row.get("lon"),
            alt=row.get("geo_altitude") or row.get("altitude"),
            speed=row.get("velocity"),
            vx=row.get("vx"),
            vy=row.get("vy"),
            timestamp=row.get("timestamp"),
            icao24=row.get("icao24"),
            track_number=row.get("track_number"),
            data_source_id=row.get("data_source_id", (0, 0)),
            service_id=row.get("service_id", 0),
        )


if __name__ == "__main__":
    encoder = AsterixEncoder()
    print(encoder.encode(lat=50.45, lon=30.52, alt=1200.0, speed=10.5))