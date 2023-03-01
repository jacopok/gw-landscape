from .gwfish import GWFishDetector
from GWFish.modules.horizon import horizon, compute_SNR
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from labellines import labelLine
from tqdm import tqdm
from .plot import FIG_PATH

def find_optimal_parameters(gwfish_detector):
    """
    A simple optimization to get a locally-optimal set of extrinsic parameters
    """
    def parameters_from_vector(x):
        theta_jn, ra, dec, psi, phase, geocent_time = x
        return {
            "mass_1": 1.4,
            "mass_2": 1.4,
            "redshift": 0.,
            "luminosity_distance": 40.,
            "theta_jn": theta_jn,
            "ra": ra,
            "dec": dec,
            "psi": psi,
            "phase": phase,
            "geocent_time": geocent_time,
        }
    def to_minimize(x):
        return - compute_SNR(parameters_from_vector(x), gwfish_detector, waveform_model="IMRPhenomD")
    
    res = minimize(to_minimize, [0., 0., 0., 0., 0., 1187008882.], tol=1e-6)

    best_parameters = parameters_from_vector(res.x)
    best_parameters.pop('mass_1')
    best_parameters.pop('mass_2')
    best_parameters.pop('redshift')
    best_parameters.pop('luminosity_distance')
    return best_parameters

def plot_snr_area(params, masses, gwfish_detector, SNR, label_line, **plot_kwargs):
    
    redshifts = []
    for mass in tqdm(masses, leave=False, unit="Masses"):
        try:
            distance, redshift = horizon(
                params = params | {'mass_1': mass/2., 'mass_2': mass/2.},
                detector = gwfish_detector,
                target_SNR = SNR,
                waveform_model='IMRPhenomD'
            )
        except ValueError as e:
            redshift = 0.
            # raise e
        redshifts.append(redshift)
    plt.fill_between(masses, redshifts, **plot_kwargs)
    if not label_line:
        return
    plt.plot(masses, redshifts, alpha=0., c='black')
    label_position = masses[np.argmax(redshifts)] / 5
    if label_position < masses[0]:
        label_position = masses[0] 
    if label_position > masses[-1]:
        label_position = masses[-1] 
    labelLine(
        plt.gca().get_lines()[-1], 
        label_position,
        label=f'SNR={SNR}',
        align=True, 
        outline_color='none',
        # yoffset=-1,
        fontsize=7,
    )

def plot_all(fig_path):
    LISA = GWFishDetector('LISA')
    LGWA = GWFishDetector('LGWA')
    ET = GWFishDetector('ET')
    masses = np.logspace(0.5, 7.5, num=200)

    params = {
        "theta_jn": 0.,
        "ra": 0.,
        "dec": 0.,
        "psi": 0.,
        "phase": 0.,
        "geocent_time": 1800000000,
    }

    detectors_colors = {
        ET.gdet: 'red',
        LGWA.gdet: 'blue',
        LISA.gdet: 'green',
    }
    
    # snr_list = [10, 30, 100, 300]
    snr_list = [10, 100]
    
    label_line=True
    for detector, color in tqdm(detectors_colors.items(), unit="detectors"):
        label=detector.name
        for snr in tqdm(snr_list, leave=False, unit="SNRs"):
            plot_snr_area(params, masses, detector, snr, label_line, color=color, alpha=.2, label=label)
            label=None
        label_line = False

    plt.legend()
    plt.xscale('log')
    plt.ylim(0, 20)
    plt.title('Horizon for equal-mass BBH')
    plt.xlabel('Total binary mass $M$ [$M_{\odot}$]')
    plt.ylabel('Redshift $z$')
    plt.savefig(fig_path, dpi=400)

if __name__ == '__main__':
    plot_all(FIG_PATH / 'horizon.png')