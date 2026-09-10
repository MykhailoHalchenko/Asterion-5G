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
        oversampling = 32
        profile = np.abs(np.fft.ifft(spectrum, n=self.config.fft_size * oversampling)) ** 2
        peak = int(np.argmax(profile))
        if 0 < peak < profile.size - 1:
            left, center, right = profile[peak - 1], profile[peak], profile[peak + 1]
            denominator = (left - 2.0 * center + right)
            correction = 0.5 * (left - right) / denominator if abs(denominator) > 1e-12 else 0.0
        else:
            correction = 0.0
        delay = (peak + correction) / (self.config.bandwidth * oversampling)
        return float(max(0.0, delay * LIGHT_SPEED))

    def estimate_range_from_channel(self, channel: np.ndarray) -> float:
        value = np.asarray(channel, dtype=np.complex128)
        if value.ndim != 4:
            raise ValueError("channel tensor must have shape (symbols, rx, tx, subcarriers)")
        return self.estimate_range(np.mean(value, axis=2))

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

    def estimate_doppler_from_channel(self, channel: np.ndarray) -> float:
        value = np.asarray(channel, dtype=np.complex128)
        if value.ndim != 4:
            raise ValueError("channel tensor must have shape (symbols, rx, tx, subcarriers)")
        strongest_index = np.unravel_index(np.argmax(np.abs(value[0])), value[0].shape)
        rx_index, tx_index, subcarrier_index = strongest_index
        tone = value[:, rx_index, tx_index, subcarrier_index]
        if tone.size < 2:
            return 0.0
        phase = np.unwrap(np.angle(tone[1:] * tone[:-1].conj()))
        return float(np.median(phase) / (2 * np.pi * self.config.symbol_duration))

    def estimate_angles_from_channel(
        self, channel: np.ndarray, antenna_positions: np.ndarray
    ) -> tuple[float, float]:
        value = np.asarray(channel, dtype=np.complex128)
        if value.ndim != 4:
            raise ValueError("channel tensor must have shape (symbols, rx, tx, subcarriers)")
        snapshot = np.mean(value, axis=(0, 2, 3))
        wavelength = LIGHT_SPEED / self.config.carrier_frequency
        azimuth_grid = np.deg2rad(np.linspace(-90.0, 90.0, 361))
        elevation_grid = np.deg2rad(np.linspace(-25.0, 25.0, 101))
        best_power = -1.0
        best_angles = (0.0, 0.0)
        y_positions = antenna_positions[:, 1]
        z_positions = antenna_positions[:, 2]
        for elevation in elevation_grid:
            horizontal = np.cos(elevation)
            for azimuth in azimuth_grid:
                steering = np.exp(
                    1j * 2 * np.pi / wavelength
                    * (y_positions * horizontal * np.sin(azimuth) + z_positions * np.sin(elevation))
                )
                power = float(abs(np.vdot(steering, snapshot)) ** 2)
                if power > best_power:
                    best_power = power
                    best_angles = (float(azimuth), float(elevation))
        return best_angles

    def reconstruct(
        self,
        received: np.ndarray,
        timestamp: datetime,
        antenna_positions: np.ndarray | None = None,
        channel: np.ndarray | None = None,
    ) -> TrackEstimate:
        if channel is not None:
            range_m = self.estimate_range_from_channel(channel)
            doppler_hz = self.estimate_doppler_from_channel(channel)
            if antenna_positions is None:
                azimuth, elevation = 0.0, 0.0
            else:
                azimuth, elevation = self.estimate_angles_from_channel(channel, antenna_positions)
        else:
            range_m = self.estimate_range(received)
            doppler_hz = self.estimate_doppler(received)
            azimuth, elevation = self.estimate_angles(received, antenna_positions)
        x = range_m * np.cos(elevation) * np.cos(azimuth)
        y_coord = range_m * np.cos(elevation) * np.sin(azimuth)
        radial_speed = doppler_hz * LIGHT_SPEED / (2.0 * self.config.carrier_frequency)
        if self._previous_range is None or self._previous_time is None:
            speed = radial_speed
        else:
            delta_t = (timestamp - self._previous_time).total_seconds()
            range_speed = (range_m - self._previous_range) / delta_t if delta_t > 0 else 0.0
            speed = 0.7 * radial_speed + 0.3 * range_speed
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