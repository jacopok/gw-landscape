import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from .gwfish import GWFishDetector
from .lisa import LISA
from .plot import plot_characteristic_noise_strain, FIG_PATH
from GWFish.modules.waveforms import LALFD_Waveform
from GWFish.modules.detection import projection
from GWFish.modules.horizon import compute_SNR

def plot_characteristic_signal_strain(params, gwfish_detector, waveform_model='IMRPhenomD', **plot_kwargs):
    
    data_params = {
        'frequencyvector': gwfish_detector.frequencyvector,
        'f_ref': 50.
    }
    waveform_obj = LALFD_Waveform(waveform_model, params, data_params)
    polarizations = waveform_obj()
    timevector = waveform_obj.t_of_f
    
    signal = projection(
        params,
        gwfish_detector,
        polarizations,
        timevector
    )
    mean_square_signal = np.sqrt(np.sum(abs(signal)**2, axis=1))[:, np.newaxis]
    plt.plot(gwfish_detector.frequencyvector, 2 * mean_square_signal * gwfish_detector.frequencyvector, **plot_kwargs)


def plot_bbh(total_mass, fig_path):
    detector_list = [
        LISA(),
        GWFishDetector('LISA'),
        GWFishDetector('LGWA'),
    ]
    plot_characteristic_noise_strain(detector_list)
    params = {
        "theta_jn": 0.,
        "ra": 0.,
        "dec": 0.,
        "psi": 0.,
        "phase": 0.,
        "geocent_time": 1800000000,
        "luminosity_distance": 167246.75453318,
        "redshift": 15,
        "mass_1": total_mass/2,
        "mass_2": total_mass/2,
    }
    LGWA_SNR = compute_SNR(params, GWFishDetector('LGWA').gdet, waveform_model='IMRPhenomD')
    LISA_SNR = compute_SNR(params, GWFishDetector('LISA').gdet, waveform_model='IMRPhenomD')
    plot_characteristic_signal_strain(params, GWFishDetector('LGWA').gdet, label=f'LGWA projection, SNR={LGWA_SNR:.1f}')
    plot_characteristic_signal_strain(params, GWFishDetector('LISA').gdet, label=f'LISA projection, SNR={LISA_SNR:.1f}')
    
    plt.legend()
    plt.ylim(1e-24, 1e-15)

    plt.savefig(fig_path, dpi=200)
    plt.close()

if __name__ == '__main__':
    mass = 3e3
    plot_bbh(mass, FIG_PATH / f'sensitivities_{mass:.0e}.png')