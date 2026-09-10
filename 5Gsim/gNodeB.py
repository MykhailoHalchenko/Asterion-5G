"""gNodeB endpoint for the 5G RF-sensing pipeline.

The class owns the channel and DSP state.  Keeping this state in one endpoint
also makes it possible to use several independent gNodeBs in a relay setup.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Protocol

import numpy as np

from .config import RadioConfig
from .mimo_3d import AircraftState, MIMO3DChannel
from .signal_postprocessing import SignalPostProcessor, TrackEstimate


class SensingPoint(Protocol):
    timestamp: datetime
    x: float
    y: float
    z: float
    speed: float


@dataclass(frozen=True)
class GNodeBStatus:
    """Observable endpoint counters useful for monitoring and tests."""

    processed_frames: int
    last_timestamp: datetime | None
    last_range_m: float | None
    last_speed_mps: float | None


@dataclass(frozen=True)
class SensingFrame:
    """Optional raw-frame container for diagnostics and offline replay."""

    channel: np.ndarray
    received: np.ndarray
    track: TrackEstimate


class gNodeB:
    """Generate pilots, sense a target, and reconstruct its trajectory."""

    def __init__(
        self,
        config: RadioConfig | None = None,
        seed: int | None = 7,
        noise_power: float = 1e-3,
        node_id: str = "gnodeb-0",
    ) -> None:
        if noise_power < 0:
            raise ValueError("noise_power must be non-negative")
        self.config = config or RadioConfig()
        self.node_id = node_id
        self.noise_power = float(noise_power)
        self.channel = MIMO3DChannel(self.config, seed=seed)
        self.postprocessor = SignalPostProcessor(self.config)
        self._processed_frames = 0
        self._last_track: TrackEstimate | None = None
        self._relay: Callable[[dict[str, Any]], None] | None = None
        self._history: list[TrackEstimate] = []

    @property
    def status(self) -> GNodeBStatus:
        track = self._last_track
        return GNodeBStatus(
            processed_frames=self._processed_frames,
            last_timestamp=track.timestamp if track else None,
            last_range_m=track.range_m if track else None,
            last_speed_mps=track.speed if track else None,
        )

    def configure_noise(self, noise_power: float) -> None:
        """Update receiver noise for subsequent frames."""
        if noise_power < 0:
            raise ValueError("noise_power must be non-negative")
        self.noise_power = float(noise_power)

    def set_relay(self, relay: Callable[[dict[str, Any]], None] | None) -> None:
        """Register a downstream callback for reconstructed track payloads."""
        self._relay = relay

    def reference_signals(self) -> np.ndarray:
        """Return deterministic pilots with OFDM-compatible dimensions."""
        return np.ones(
            (
                self.config.num_ofdm_symbols,
                self.config.num_tx_antennas,
                self.config.fft_size,
            ),
            dtype=np.complex128,
        )

    def pilot_mask(self) -> np.ndarray:
        """Return the active subcarrier mask used by the pilot grid."""
        return np.ones(self.config.fft_size, dtype=bool)

    @staticmethod
    def _validate_point(point: SensingPoint) -> None:
        for name in ("x", "y", "z", "speed"):
            value = float(getattr(point, name))
            if not np.isfinite(value):
                raise ValueError(f"trajectory point field {name!r} must be finite")
        if point.timestamp.tzinfo is None:
            raise ValueError("trajectory timestamp must be timezone-aware")

    def _state_from_point(self, point: SensingPoint) -> AircraftState:
        self._validate_point(point)
        return AircraftState(
            position=(float(point.x), float(point.y), float(point.z)),
            velocity=(float(point.speed), 0.0, 0.0),
        )

    def sense_state(self, state: AircraftState) -> tuple[np.ndarray, np.ndarray]:
        """Generate a channel tensor and received signal for a 3D state."""
        return self.channel.generate(
            state,
            num_symbols=self.config.num_ofdm_symbols,
            noise_power=self.noise_power,
        )

    def sense(self, point: SensingPoint) -> tuple[np.ndarray, np.ndarray]:
        return self.sense_state(self._state_from_point(point))

    def process(self, point: SensingPoint) -> TrackEstimate:
        """Process one point and update endpoint state."""
        channel, received = self.sense(point)
        track = self.postprocessor.reconstruct(
            received,
            point.timestamp.astimezone(timezone.utc),
            antenna_positions=self.channel.rx_positions,
            channel=channel,
        )
        self._processed_frames += 1
        self._last_track = track
        self._history.append(track)
        if len(self._history) > 1024:
            del self._history[:-1024]
        if self._relay is not None:
            self._relay(self.track_payload(track))
        return track

    def process_frame(self, point: SensingPoint) -> SensingFrame:
        """Process one point while retaining the generated RF tensors."""
        channel, received = self.sense(point)
        track = self.postprocessor.reconstruct(
            received,
            point.timestamp.astimezone(timezone.utc),
            antenna_positions=self.channel.rx_positions,
            channel=channel,
        )
        self._processed_frames += 1
        self._last_track = track
        self._history.append(track)
        if len(self._history) > 1024:
            del self._history[:-1024]
        if self._relay is not None:
            self._relay(self.track_payload(track))
        return SensingFrame(channel, received, track)

    def history(self, limit: int | None = None) -> tuple[TrackEstimate, ...]:
        """Return immutable recent estimates for a monitoring consumer."""
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        values = self._history if limit is None else ([] if limit == 0 else self._history[-limit:])
        return tuple(values)

    def latest_payload(self) -> dict[str, Any]:
        return self.track_payload()

    def quality_metrics(self, track: TrackEstimate | None = None) -> dict[str, float]:
        """Expose stable scalar quality indicators for relay monitoring."""
        value = track or self._last_track
        if value is None:
            raise RuntimeError("no reconstructed track is available")
        return {
            "range_m": float(value.range_m),
            "speed_mps": float(value.speed),
            "accuracy_m": float(value.accuracy_m),
            "azimuth_rad": float(value.azimuth_rad),
            "elevation_rad": float(value.elevation_rad),
        }

    def is_ready(self) -> bool:
        return self._last_track is not None

    def antenna_positions(self) -> dict[str, np.ndarray]:
        """Return copies so monitoring code cannot mutate channel geometry."""
        return {
            "tx": np.array(self.channel.tx_positions, copy=True),
            "rx": np.array(self.channel.rx_positions, copy=True),
        }

    def validate(self) -> None:
        """Raise if the endpoint configuration is not simulation-safe."""
        if self.config.num_tx_antennas < 1 or self.config.num_rx_antennas < 1:
            raise ValueError("gNodeB requires at least one TX and RX antenna")
        if self.config.fft_size < 2 or self.config.num_ofdm_symbols < 2:
            raise ValueError("OFDM grid must contain at least two bins and symbols")

    def close(self) -> None:
        """Detach relay callbacks and release accumulated estimate history."""
        self._relay = None
        self._history.clear()

    def process_batch(self, points: Iterable[SensingPoint]) -> list[TrackEstimate]:
        return [self.process(point) for point in points]

    def track_payload(self, track: TrackEstimate | None = None) -> dict[str, Any]:
        """Create a JSON/ASTERIX-ready payload from the latest estimate."""
        value = track or self._last_track
        if value is None:
            raise RuntimeError("no reconstructed track is available")
        return {
            "node_id": self.node_id,
            "timestamp": value.timestamp.astimezone(timezone.utc).timestamp(),
            "x": float(value.x),
            "y": float(value.y),
            "speed": float(value.speed),
            "range_m": float(value.range_m),
            "azimuth_rad": float(value.azimuth_rad),
            "elevation_rad": float(value.elevation_rad),
            "accuracy": float(value.accuracy_m),
        }

    def channel_shapes(self) -> dict[str, tuple[int, ...]]:
        """Expose channel dimensions without allocating a simulation frame."""
        return {
            "rx_positions": tuple(self.channel.rx_positions.shape),
            "tx_positions": tuple(self.channel.tx_positions.shape),
        }

    def reset(self) -> None:
        """Reset DSP history and endpoint counters."""
        self.postprocessor = SignalPostProcessor(self.config)
        self._processed_frames = 0
        self._last_track = None
        self._history.clear()
