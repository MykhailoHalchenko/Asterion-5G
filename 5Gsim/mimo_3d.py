"""Geometry-aware stochastic 3D MIMO channel (without ray tracing)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from .config import LIGHT_SPEED, RadioConfig
except ImportError:
    from config import LIGHT_SPEED, RadioConfig


@dataclass(frozen=True)
class AircraftState:
    position: tuple[float, float, float]
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)


def _upa(rows: int, cols: int, spacing: float) -> np.ndarray:
    y, z = np.meshgrid(
        (np.arange(cols) - (cols - 1) / 2.0) * spacing,
        (np.arange(rows) - (rows - 1) / 2.0) * spacing,
    )
    return np.column_stack((np.zeros(rows * cols), y.ravel(), z.ravel()))


class MIMO3DChannel:
    """Generate a specular aircraft cluster plus diffuse spatial paths."""

    def __init__(self, config: RadioConfig | None = None, seed: int | None = 7):
        self.config = config or RadioConfig()
        self.rng = np.random.default_rng(seed)
        wavelength = LIGHT_SPEED / self.config.carrier_frequency
        self.tx_positions = _upa(self.config.tx_rows, self.config.tx_cols, wavelength / 2)
        self.rx_positions = _upa(self.config.rx_rows, self.config.rx_cols, wavelength / 2)
        self._sionna_arrays: tuple[Any, Any] | None = None

    @property
    def num_tx(self) -> int:
        return self.config.num_tx_antennas

    @property
    def num_rx(self) -> int:
        return self.config.num_rx_antennas

    def sionna_arrays(self) -> tuple[Any, Any]:
        if self._sionna_arrays is None:
            try:
                from sionna.phy.channel.tr38901 import PanelArray
            except ImportError as exc:
                raise RuntimeError("Sionna PHY is required for antenna arrays") from exc
            kwargs = {
                "polarization": "single",
                "polarization_type": "V",
                "antenna_pattern": "omni",
                "carrier_frequency": self.config.carrier_frequency,
            }
            self._sionna_arrays = (
                PanelArray(self.config.tx_rows, self.config.tx_cols, **kwargs),
                PanelArray(self.config.rx_rows, self.config.rx_cols, **kwargs),
            )
        return self._sionna_arrays

    def spatial_covariance(self, angle: float, spread: float = 0.12) -> np.ndarray:
        wavelength = LIGHT_SPEED / self.config.carrier_frequency
        steering = np.exp(1j * 2 * np.pi * self.rx_positions[:, 1] * np.sin(angle) / wavelength)
        covariance = np.outer(steering, steering.conj()) + spread * np.eye(self.num_rx)
        return covariance / np.trace(covariance).real * self.num_rx

    def _steering(self, positions: np.ndarray, azimuth: float, elevation: float) -> np.ndarray:
        wavelength = LIGHT_SPEED / self.config.carrier_frequency
        direction = np.array([
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ])
        return np.exp(1j * 2 * np.pi * positions @ direction / wavelength)

    def generate(
        self,
        state: AircraftState,
        num_symbols: int | None = None,
        noise_power: float = 1e-3,
        diffuse_paths: int = 4,
    ) -> tuple[np.ndarray, np.ndarray]:
        if noise_power < 0 or diffuse_paths < 0:
            raise ValueError("noise_power and diffuse_paths must be non-negative")
        symbols = num_symbols or self.config.num_ofdm_symbols
        target = np.asarray(state.position, dtype=float)
        distance = float(np.linalg.norm(target))
        if distance <= 0:
            raise ValueError("aircraft position must differ from the base station")
        azimuth = float(np.arctan2(target[1], target[0]))
        elevation = float(np.arctan2(target[2], np.linalg.norm(target[:2])))
        tx = self._steering(self.tx_positions, azimuth, elevation)
        rx = self._steering(self.rx_positions, azimuth, elevation)
        channel = np.outer(rx, tx.conj()) / distance
        channel *= np.exp(1j * self.rng.uniform(-np.pi, np.pi))
        for _ in range(diffuse_paths):
            path_azimuth = azimuth + self.rng.normal(0.0, 0.04)
            path_elevation = elevation + self.rng.normal(0.0, 0.02)
            diffuse_tx = self._steering(self.tx_positions, path_azimuth, path_elevation)
            diffuse_rx = self._steering(self.rx_positions, path_azimuth, path_elevation)
            coefficient = (
                self.rng.normal() + 1j * self.rng.normal()
            ) * 0.08 / max(distance, 1.0)
            channel += coefficient * np.outer(diffuse_rx, diffuse_tx.conj())
        delta = target
        radial_velocity = float(np.dot(delta, np.asarray(state.velocity)) / distance)
        wavelength = LIGHT_SPEED / self.config.carrier_frequency
        time_phase = np.exp(1j * 2 * np.pi * radial_velocity * np.arange(symbols) / wavelength)
        frequency_phase = np.exp(
            -1j * 2 * np.pi * np.arange(self.config.fft_size)
            * distance / LIGHT_SPEED * self.config.subcarrier_spacing
        )
        h = time_phase[:, None, None, None] * channel[None, :, :, None] * frequency_phase
        pilots = np.ones((symbols, self.num_tx, self.config.fft_size), dtype=np.complex128)
        y = np.einsum("srtf,stf->srf", h, pilots)
        if noise_power:
            y += np.sqrt(noise_power / 2) * (
                self.rng.normal(size=y.shape) + 1j * self.rng.normal(size=y.shape)
            )
        return h, y