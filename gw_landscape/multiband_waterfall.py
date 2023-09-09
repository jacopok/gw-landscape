from .gwfish import GWFishDetector
from .plot import FIG_PATH
from .waterfall import plot_snr_area, compute_horizon_from_masses
import numpy as np
import matplotlib.pyplot as plt
from GWFish.modules.horizon import horizon, compute_SNR, find_optimal_location, Network

def multiband_waterfall():

    ET = GWFishDetector('ET')
    LGWA = GWFishDetector('LGWA')

    base_params = {
        "theta_jn": 0.,
        "ra": 0.,
        "dec": 0.,
        "psi": 0.,
        "phase": 0.,
        "geocent_time": 1800000000,
    }

    masses = np.logspace(.5, 5.5, num=200)
    
    snr = 10.

    params_et = find_optimal_location(base_params | {'mass_1': 100., 'mass_2': 100.}, ET.gdet)
    params_lgwa = find_optimal_location(base_params | {'mass_1': 100., 'mass_2': 100.}, LGWA.gdet)

    _, horizon_et = compute_horizon_from_masses(params_et, masses, ET.gdet, snr, source_frame_trick=False)
    _, horizon_lgwa = compute_horizon_from_masses(params_lgwa, masses, LGWA.gdet, snr, source_frame_trick=False)

    _, horizon_et_misaligned = compute_horizon_from_masses(params_lgwa, masses, ET.gdet, snr, source_frame_trick=False)
    _, horizon_lgwa_misaligned = compute_horizon_from_masses(params_et, masses, LGWA.gdet, snr, source_frame_trick=False)
    
    et_aligned_horizon = np.minimum(horizon_et, horizon_lgwa_misaligned)
    lgwa_aligned_horizon = np.minimum(horizon_et_misaligned, horizon_lgwa)
    
    plt.fill_between(
        masses, np.maximum(
            et_aligned_horizon, 
            lgwa_aligned_horizon), 
        alpha=.2, color='grey',
        label='Multiband detection'
    )
    
    plt.plot(masses, horizon_et, color='blue', label='ET horizon')
    plt.plot(masses, horizon_lgwa, color='green', label='LGWA horizon')

    plt.xscale('log')
    plt.xlim(masses[0], masses[-1])
    
    plt.yscale('log')
    plt.ylim(2e-1, 2e1)

    plt.legend()
    plt.show()

if __name__ == '__main__':
    multiband_waterfall()