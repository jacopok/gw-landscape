from .gwfish import GWFishDetector
from .plot import FIG_PATH, make_redshift_distance_axes
from .waterfall import plot_snr_area, compute_horizon_from_masses
import numpy as np
import matplotlib.pyplot as plt
from GWFish.modules.horizon import horizon, compute_SNR, find_optimal_location, Network
from labellines import labelLine


BASE_PARAMS = {
    "theta_jn": 0.,
    "ra": 0.,
    "dec": 0.,
    "psi": 0.,
    "phase": 0.,
    "geocent_time": 1800000000,
}
    
def multiband_horizons(reference_mass, masses, detector_1, detector_2, snr):
    
    params_1 = find_optimal_location(BASE_PARAMS | {'mass_1': reference_mass/2., 'mass_2': reference_mass/2.}, detector_1)
    params_2 = find_optimal_location(BASE_PARAMS | {'mass_1': reference_mass/2., 'mass_2': reference_mass/2.}, detector_2)

    _, horizon_1 = compute_horizon_from_masses(params_1, masses, detector_1, snr, source_frame_trick=False)
    _, horizon_2 = compute_horizon_from_masses(params_2, masses, detector_2, snr, source_frame_trick=False)

    _, horizon_1_misaligned = compute_horizon_from_masses(params_2, masses, detector_1, snr, source_frame_trick=False)
    _, horizon_2_misaligned = compute_horizon_from_masses(params_1, masses, detector_2, snr, source_frame_trick=False)
    
    aligned_horizon_1 = np.minimum(horizon_1, horizon_2_misaligned)
    aligned_horizon_2 = np.minimum(horizon_1_misaligned, horizon_2)
    
    return horizon_1, horizon_2, np.maximum(aligned_horizon_1, aligned_horizon_2)

def label_last_line(label, position, masses, curve):
    
    idx = np.searchsorted(masses, position)
    
    # spacing for the labels on a log scale
    approx_value = curve[idx]
    yoffset = - approx_value / 5
    
    labelLine(
        plt.gca().get_lines()[-1], 
        position,
        label=label,
        align=True, 
        outline_color='none',
        yoffset=yoffset,
        fontsize=7,
    )

def multiband_waterfall_et_lgwa():

    ET = GWFishDetector('ET').gdet
    LGWA = GWFishDetector('LGWA').gdet

    masses = np.geomspace(10, 2e4, num=100)
    
    snr_threshold = 8.

    snrs = [8., 50., 250., 1000.]
    label_positions = [1000, 1200, 1500, 2000]

    horizon_et, horizon_lgwa, horizon_et_lgwa = multiband_horizons(100., masses, ET, LGWA, snr_threshold)
    
    ax_redshift, ax_distance = make_redshift_distance_axes()

    
    ax_redshift.plot(masses, horizon_et, color='blue', label='ET horizon')
    label_last_line(f'SNR={snr_threshold}', 200, masses, horizon_et)
    ax_redshift.plot(masses, horizon_lgwa, color='red', label='LGWA horizon')
    label_last_line(f'SNR={snr_threshold}', 3e3, masses, horizon_lgwa)
    # ax_redshift.plot(masses, horizon_lisa, color='green', label='LISA horizon')

    params_et = find_optimal_location(BASE_PARAMS | {'mass_1': 100./2., 'mass_2': 100./2.}, ET)

    for snr, label_position in zip(snrs, label_positions):
        
        _, network_horizon = compute_horizon_from_masses(
            params_et, 
            masses, 
            Network(['ET', 'LGWA']), 
            snr, 
            source_frame_trick=False,
            recompute_location=True,
        )
        
        network_horizon_multiband = np.minimum(network_horizon, horizon_et_lgwa)
        
        ax_redshift.fill_between(
            masses, network_horizon_multiband, 
            alpha=.2, color='grey',
        )
        ax_redshift.plot(masses, network_horizon_multiband, alpha=0., c='black')
        if label_position is not None:
            label_last_line(f'SNR$\geq${snr:.0f}', label_position, masses, network_horizon_multiband)

    ax_redshift.set_xscale('log')
    ax_redshift.set_xlim(masses[0], masses[-1])
    
    ax_redshift.set_yscale('log')
    ax_redshift.set_ylim(5e-2, 8e1)
    ax_redshift.set_xlabel('Total binary mass [M$_\odot$]')
    
    plt.title('Multiband horizon for equal-mass BBH')
    plt.legend()
    plt.savefig(FIG_PATH / 'multiband_horizon_et_lgwa.pdf', dpi=200)

def multiband_waterfall_lisa_lgwa():

    LGWA = GWFishDetector('LGWA').gdet
    LISA = GWFishDetector('LISA').gdet
    
    masses = np.geomspace(50, 2e7, num=100)
    
    snr_threshold = 8.
    snrs = [8., 50., 250., 1000.]
    label_positions = [7e3, 5e3, 5e3, 1e4]

    horizon_lisa, horizon_lgwa, horizon_lisa_lgwa = multiband_horizons(1e4, masses, LISA, LGWA, snr_threshold)
        
    ax_redshift, ax_distance = make_redshift_distance_axes()
    
    ax_redshift.plot(masses, horizon_lisa, color='green', label='LISA horizon')
    label_last_line(f'SNR={snr_threshold}', 1e6, masses, horizon_lisa)
    ax_redshift.plot(masses, horizon_lgwa, color='red', label='LGWA horizon')
    label_last_line(f'SNR={snr_threshold}', 500, masses, horizon_lgwa)

    params_lgwa = find_optimal_location(BASE_PARAMS | {'mass_1': 1e4/2., 'mass_2': 1e4/2.}, LGWA)

    for snr, label_position in zip(snrs, label_positions):
        
        _, network_horizon = compute_horizon_from_masses(
            params_lgwa, 
            masses, 
            Network(['LISA', 'LGWA']), 
            snr, 
            source_frame_trick=False,
            recompute_location=True,
        )
        
        network_horizon_multiband = np.minimum(network_horizon, horizon_lisa_lgwa)
        
        ax_redshift.fill_between(
            masses, network_horizon_multiband, 
            alpha=.2, color='grey',
        )
        ax_redshift.plot(masses, network_horizon_multiband, alpha=0., c='black')
        if label_position is not None:
            label_last_line(f'SNR$\geq${snr:.0f}', label_position, masses, network_horizon_multiband)

    ax_redshift.set_xscale('log')
    ax_redshift.set_xlim(masses[0], masses[-1])
    
    ax_redshift.set_yscale('log')
    ax_redshift.set_ylim(5e-2, 1e2)
    ax_redshift.set_xlabel('Total binary mass [M$_\odot$]')
    
    plt.title('Multiband horizon for equal-mass BBH')
    plt.legend()
    plt.savefig(FIG_PATH / 'multiband_horizon_lisa_lgwa.pdf', dpi=200)


if __name__ == '__main__':
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Serif"
    })

    multiband_waterfall_et_lgwa()
    multiband_waterfall_lisa_lgwa()