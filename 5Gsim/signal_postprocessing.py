"""ToA, AoA, Doppler and airport-coordinate reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

try:
    from .config import LIGHT_SPEED, RadioConfig
except ImportError:
    from config import LIGHT_SPEED, RadioConfig


@dataclass(frozen=True)
class TrackEstimate:
    timestamp: datetime
    x: float
    y: float
    speed: float
    range_m: float
    azimuth_rad: float
    elevation_rad: float
    accuracy_m: float


class SignalPostProcessor:
    def __init__(self, config: RadioConfig | None = None):
        self.config = config or RadioConfig()
        self._previous_range: float | None = None
        self._previous_time: datetime | None = None

    @staticmethod
    def _normalize(received: np.ndarray) -> np.ndarray:
        value = np.asarray(received, dtype=np.complex128)
        if value.ndim not in (2, 3):
            raise ValueError(
                "received symbols must have shape (symbols, subcarriers) or "
                "(symbols, antennas, subcarriers)"
            )
        return value

    def estimate_range(self, received: np.ndarray) -> float:
        value = self._normalize(received)
        spectrum = np.mean(value, axis=tuple(range(value.ndim - 1)))
        frequencies = np.arange(spectrum.size) * self.config.subcarrier_spacing
        phase = np.unwrap(np.angle(spectrum))
        weights = np.abs(spectrum) ** 2
        if np.sum(weights) <= 1e-12:
            return 0.0
        slope = np.polyfit(frequencies, phase, 1, w=np.sqrt(weights))[0]
        return float(max(0.0, -slope * LIGHT_SPEED / (2.0 * np.pi)))

    def estimate_angles(
        self,
        received: np.ndarray,
        antenna_positions: np.ndarray | None = None,
    ) -> tuple[float, float]:
        value = self._normalize(received)
        if value.ndim != 3 or antenna_positions is None:
            return 0.0, 0.0
        snapshot = np.mean(value, axis=(0, 2))
        phase = np.unwrap(np.angle(snapshot / (snapshot[0] + 1e-12)))
        wavelength = LIGHT_SPEED / self.config.carrier_frequency
        design = np.column_stack((antenna_positions[:, 1], antenna_positions[:, 2]))
        gradient = np.linalg.lstsq(
            design[1:],
            phase[1:] * wavelength / (2.0 * np.pi),
            rcond=None,
        )[0]
        direction_y, direction_z = np.clip(gradient, -1.0, 1.0)
        elevation = float(np.arcsin(direction_z))
        horizontal = max(np.cos(elevation), 1e-9)
        azimuth = float(np.arcsin(np.clip(direction_y / horizontal, -1.0, 1.0)))
        return azimuth, elevation

    def estimate_doppler(self, received: np.ndarray) -> float:
        value = self._normalize(received)
        if value.shape[0] < 2:
            return 0.0
        correlation = value[1:] * value[:-1].conj()
        phase = np.unwrap(np.angle(np.mean(correlation, axis=tuple(range(1, correlation.ndim)))))
        return float(
            np.median(phase) / (2 * np.pi * self.config.symbol_duration)
        )

    def reconstruct(
        self,
        received: np.ndarray,
        timestamp: datetime,
        antenna_positions: np.ndarray | None = None,
    ) -> TrackEstimate:
        range_m = self.estimate_range(received)
        azimuth, elevation = self.estimate_angles(received, antenna_positions)
        x = range_m * np.cos(elevation) * np.cos(azimuth)
        y_coord = range_m * np.cos(elevation) * np.sin(azimuth)
        if self._previous_range is None or self._previous_time is None:
            speed = 0.0
        else:
            delta_t = (timestamp - self._previous_time).total_seconds()
            speed = (range_m - self._previous_range) / delta_t if delta_t > 0 else 0.0
        self._previous_range, self._previous_time = range_m, timestamp
        return TrackEstimate(
            timestamp=timestamp.astimezone(timezone.utc),
            x=float(x),
            y=float(y_coord),
            speed=float(speed),
            range_m=range_m,
            azimuth_rad=azimuth,
            elevation_rad=elevation,
            accuracy_m=float(LIGHT_SPEED / (self.config.fft_size * self.config.subcarrier_spacing)),
        )