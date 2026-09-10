"""Global radio and airport topology configuration."""

from dataclasses import dataclass

CARRIER_FREQ = 28e9
SUBCARRIER_SPACING = 120e3
FFT_SIZE = 256
NUM_OFDM_SYMBOLS = 14
BANDWIDTH = FFT_SIZE * SUBCARRIER_SPACING
NUM_TX_ROWS = 2
NUM_TX_COLS = 4
NUM_RX_ROWS = 2
NUM_RX_COLS = 4
NOISE_FIGURE_DB = 7.0
LIGHT_SPEED = 299_792_458.0

GNODEB_TOPOLOGY: dict[str, tuple[float, float, float]] = {
    "gnodeb_0": (0.0, 0.0, 12.0),
    "gnodeb_1": (120.0, 0.0, 10.0),
    "gnodeb_2": (0.0, 120.0, 10.0),
}


@dataclass(frozen=True)
class RadioConfig:
    carrier_frequency: float = CARRIER_FREQ
    subcarrier_spacing: float = SUBCARRIER_SPACING
    fft_size: int = FFT_SIZE
    num_ofdm_symbols: int = NUM_OFDM_SYMBOLS
    tx_rows: int = NUM_TX_ROWS
    tx_cols: int = NUM_TX_COLS
    rx_rows: int = NUM_RX_ROWS
    rx_cols: int = NUM_RX_COLS
    noise_figure_db: float = NOISE_FIGURE_DB

    @property
    def num_tx_antennas(self) -> int:
        return self.tx_rows * self.tx_cols

    @property
    def num_rx_antennas(self) -> int:
        return self.rx_rows * self.rx_cols

    @property
    def bandwidth(self) -> float:
        return self.fft_size * self.subcarrier_spacing

    @property
    def symbol_duration(self) -> float:
        return 1.0 / self.subcarrier_spacing