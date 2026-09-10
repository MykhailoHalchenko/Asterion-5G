import numpy as np
import tensorflow as tf
from sionna.phy import config
from sionna.phy.ofdm import (
    ResourceGrid,
    ResourceGridMapper,
    ResourceGridDemapper,
    OFDMModulator,
    OFDMDemodulator,
    LSChannelEstimator,
)
from sionna.phy.mimo import StreamManagement, LMMSEEqualizer
from sionna.phy.mapping import Mapper, Demapper, QAMSource
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.utils import ebnodb2no, hard_decisions
from sionna.phy.channel.tr38901 import UMa, PanelArray
from sionna.phy.channel import OFDMChannel, gen_single_sector_topology
from sionna.sys import phy_abstraction
from sionna.sys import scheduling

class gNodeB:
    carrier_frequency : float
    bandwidth : float
    subcarrier_spacing : float
    num_tx_ant : int
    unm_rx_ant : int 
    max_num_streams: int
    cell_id : int

    def __init__(
            self,
            carrier_frequency: float = 3.5e9,
            bandwitdh: float = 20e6,
            subcarrier_spacing: float = 30e3,
            num_tx_ant: int = 8,
            unm_rx_ant: int = 2,
            max_num_streams: int = 2,
            cell_id: int = 0,
    ):
        config.phy_precision = "complex64"
        self.carrier_frequency = carrier_frequency
        self.bandwidth = bandwitdh
        self.subcarrier_spacing = subcarrier_spacing
        self.num_tx_ant = num_tx_ant
        self.num_rx_ant = num_tx_ant
        self.max_num_streams = max_num_streams
        self.cell_id = cell_id
        n_rb = int(np.floor(bandwitdh / subcarrier_spacing / 12))
        self.num_subcarriers = n_rb * 12
        self.num_ofdm_symbols = 14
        self.cp_lemgth = int(np.ceil(0.07 * (1 / subcarrier_spacing) * self._fft_size()))
        self._build_phy_pipeline()

        def _fft_size(self) -> int:
            n = self.num_subcarriers
            return 1 << (n - 1).bit_length()

        def _build_phy_pipeline(self):
            self.resource_grid = ResourceGrid(num_ofdm_symbols=self.num_ofdm_symbols, fft_size=self._fft_size(), 
                subcarrier_spacing=self.subcarrier_spacing,
                num_tx=self.num_tx_ant,
                num_streams_per_tx=self.max_num_streams,
                cyclic_prefix_length=self.cp_length,
                num_guarg_carriers = (int(self.num_subcarriers * 0.05), 0),
                pilot_pattern="kronecker",
                pilot_ofdm_symbol_indices=[2, 11])
            
            
            self.stream_management = StreamManagement(
                num_streams_per_tx=self.max_num_streams,
                num_tx=self.num_tx_ant,
                num_rx=1,
                )
            self.num_bits_per_symbol = 4           # 16-QAM
            self.coderate = 0.5
            n = int(np.prod(self.resource_grid.num_data_symbols) *
                    self.num_bits_per_symbol)
            k = int(self.coderate * n)
            self.encoder = LDPC5GEncoder(k=k, n=n)
            self.decoder = LDPC5GDecoder(self.encoder, hard_out=True)

            self.mapper = Mapper("qam", self.num_bits_per_symbol)
            self.demapper = Demapper("app", "qam", self.num_bits_per_symbol)
            self.qam_source = QAMSource(self.num_bits_per_symbol)

        def _build_sys_pipeline(self):
            self.effective_sinr = phy_abstraction.EffectiveSINR(
            num_bits_per_symbol=self.num_bits_per_symbol,
            coderate=self.coderate,
        )
            self.scheduler = scheduling.ProportionalFairScheduler()

        def schedule(self, ue_metrics: dict) -> dict:
            priority = self.scheduler.compute_priority(ue_metrics)
            ue_id = int(np.argmax(priority))
            mcs = self._select_mcs(ue_metrics["cqi"][ue_id])
            return {"ue_id": ue_id, "mcs": mcs, "priority": priority}

        def _select_mcs(self, cqi: float) -> int:
            return int(phy_abstraction.cqi_to_mcs(cqi))

        def _mcs_to_modulation(self, mcs: int):
            return phy_abstraction.mcs_to_modulation(mcs)

        # DL(Downlink)
        def transmit(self, num_bits: int):
            bits = tf.random.uniform(
            [1, num_bits], minval=0, maxval=2, dtype=tf.int32)
            coded = self.encoder(bits)
            x = self.mapper(coded)
            x_rg = self.rg_mapper(x)
            x_ofdm = self.ofdm_mod(x_rg)
            return x_ofdm, x_rg, bits
        
        # UL(Uplink)
        def receive(self, y_ofdm: tf.Tensor, num_bits: int):
            y_rg = self.ofdm_demod(y_ofdm)
            h_hat, err_var = self.channel_estimator(y_rg, self.no)
            x_hat, no_eff = self.equalizer(y_rg, h_hat, err_var, self.no)
            llr = self.demapper(x_hat, no_eff)
            bits_hat = self.decoder(llr[:, :num_bits])
            return bits_hat

        def build_chanel(self, ut_loc=None, bs_loc=None, num_ut: int = 1):
            bs_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=self.num_tx_ant,
            polarization="dual",
            polarization_type="VH",
        )
            ut_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=self.num_rx_ant,
            polarization="single",
            polarization_type="V",
        )
            topology = gen_single_sector_topology(
            batch_size=1,
            num_ut=num_ut,
            scenario="uma",
            min_ut_velocity=3.0,
            max_ut_velocity=30.0,
        )
            channel_model = UMa(
            carrier_frequency=self.carrier_frequency,
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            enable_pathloss=True,
            enable_shadow_fading=True,
        )
            channel = OFDMChannel(
            channel_model=channel_model,
            resource_grid=self.resource_grid,
            add_neutral_symbols=False,
        )
            return channel, topology

        #SINR
        def set_noise_from_snr(self, snr_db: float, coderate: float = None):
            coderate = coderate if coderate is not None else self.coderate
            self.no = ebnodb2no(
                ebno_db=snr_db - 10 * np.log10(self.num_bits_per_symbol),
                num_bits_per_symbol=self.num_bits_per_symbol,
                coderate=coderate,
                )
            return self.no
        
        def estimate_sinr(self, h_eff: tf.Tensor, no: tf.Tensor) -> tf.Tensor:
            return self.effective_sinr(h_eff, no)