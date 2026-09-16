"""Sionna PHY-based stochastic 3D MIMO channel (without ray tracing)."""

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


class MIMO3DChannel:
    """Generate aircraft channels with Sionna's 3GPP CDL model."""

    def __init__(self, config: RadioConfig | None = None, seed: int | None = 7):
        self.config = config or RadioConfig()
        try:
            import torch
            from sionna.phy.channel import AWGN
            from sionna.phy.channel import cir_to_ofdm_channel
            from sionna.phy.channel.tr38901 import CDL, PanelArray
            from sionna.phy.ofdm import ResourceGrid
        except ImportError as exc:
            raise RuntimeError(
                "Sionna PHY and its Torch backend are required for the channel"
            ) from exc

        if seed is not None:
            torch.manual_seed(seed)
        self._torch = torch
        self._cir_to_ofdm_channel = cir_to_ofdm_channel
        self._awgn = AWGN()
        self._sionna_arrays = (
            PanelArray(
                self.config.tx_rows,
                self.config.tx_cols,
                polarization="single",
                polarization_type="V",
                antenna_pattern="omni",
                carrier_frequency=self.config.carrier_frequency,
            ),
            PanelArray(
                self.config.rx_rows,
                self.config.rx_cols,
                polarization="single",
                polarization_type="V",
                antenna_pattern="omni",
                carrier_frequency=self.config.carrier_frequency,
            ),
        )
        self.tx_positions = self._sionna_arrays[0].ant_pos.detach().cpu().numpy()
        self.rx_positions = self._sionna_arrays[1].ant_pos.detach().cpu().numpy()
        self.resource_grid = ResourceGrid(
            num_ofdm_symbols=self.config.num_ofdm_symbols,
            fft_size=self.config.fft_size,
            subcarrier_spacing=self.config.subcarrier_spacing,
            num_tx=self.config.num_tx_antennas,
            num_streams_per_tx=1,
            precision="single",
        )
        self._frequencies = torch.fft.fftshift(
            torch.fft.fftfreq(
                self.config.fft_size,
                d=1.0 / self.config.bandwidth,
                dtype=torch.float32,
            )
        )
        self._channel_model = CDL(
            model="C",
            delay_spread=100e-9,
            carrier_frequency=self.config.carrier_frequency,
            ut_array=self._sionna_arrays[1],
            bs_array=self._sionna_arrays[0],
            direction="downlink",
            ut_velocity=torch.zeros(3, dtype=torch.float32),
            precision="single",
        )

    @property
    def num_tx(self) -> int:
        return self.config.num_tx_antennas

    @property
    def num_rx(self) -> int:
        return self.config.num_rx_antennas

    def sionna_arrays(self) -> tuple[Any, Any]:
        return self._sionna_arrays

    def spatial_covariance(self, angle: float, spread: float = 0.12) -> np.ndarray:
        steering = np.exp(
            1j
            * 2
            * np.pi
            * self.rx_positions[:, 1]
            * np.sin(angle)
            / (LIGHT_SPEED / self.config.carrier_frequency)
        )
        covariance = np.outer(steering, steering.conj()) + spread * np.eye(self.num_rx)
        return covariance / np.trace(covariance).real * self.num_rx

    def generate(
        self,
        state: AircraftState,
        num_symbols: int | None = None,
        noise_power: float = 1e-3,
    ) -> tuple[np.ndarray, np.ndarray]:
        if noise_power < 0:
            raise ValueError("noise_power must be non-negative")
        symbols = num_symbols or self.config.num_ofdm_symbols
        target = np.asarray(state.position, dtype=float)
        distance = float(np.linalg.norm(target))
        if distance <= 0:
            raise ValueError("aircraft position must differ from the base station")
        torch = self._torch
        radial_velocity = float(np.dot(target, np.asarray(state.velocity)) / distance)

        path_gain, path_delay = self._channel_model(
            1,
            symbols,
            self.config.bandwidth,
        )
        path_delay = path_delay + distance / LIGHT_SPEED
        channel = self._cir_to_ofdm_channel(
            self._frequencies.to(path_gain.device),
            path_gain,
            path_delay,
            normalize=False,
        )
        channel = channel[0, 0, :, 0, :, :, :].permute(2, 0, 1, 3)

        doppler_hz = 2.0 * radial_velocity / (
            LIGHT_SPEED / self.config.carrier_frequency
        )
        doppler = torch.exp(
            1j
            * 2
            * torch.pi
            * doppler_hz
            * torch.arange(
                symbols,
                dtype=torch.float32,
                device=channel.device,
            )
            * self.config.symbol_duration
        )
        channel = channel * doppler[:, None, None, None]
        pilots = torch.ones(
            (symbols, self.num_tx, self.config.fft_size),
            dtype=channel.dtype,
            device=channel.device,
        )
        received = torch.einsum("srtf,stf->srf", channel, pilots)
        if noise_power:
            received = self._awgn(
                received,
                torch.as_tensor(
                    noise_power,
                    dtype=torch.float32,
                    device=received.device,
                ),
            )
        return channel.detach().cpu().numpy(), received.detach().cpu().numpy()