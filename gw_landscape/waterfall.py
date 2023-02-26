from .gwfish import GWFishDetector
from GWFish.modules.horizon import horizon, compute_SNR
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from labellines import labelLine
from tqdm import tqdm

def find_optimal_parameters(gwfish_detector):
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

def plot_single(masses, gwfish_detector, SNR, **plot_kwargs):
    
    params = find_optimal_parameters(gwfish_detector)
        
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
            # redshift = 0.
            raise e
        redshifts.append(redshift)
    plt.fill_between(masses, redshifts, **plot_kwargs)
    plt.plot(masses, redshifts, alpha=0., label=SNR, c='black')
    maxmass = masses[np.argmax(redshifts)]
    labelLine(
        plt.gca().get_lines()[-1], 
        maxmass*2,
        label=SNR,
        align=True, 
        outline_color=None,
        # yoffset=-1,
        # fontsize=14
    )

def plot_all():
    LGWA = GWFishDetector('LGWA')
    ET = GWFishDetector('ET')
    n_f = len(ET.gdet.frequencyvector)
    f_max = ET.gdet.frequencyvector[-1]
    ET.gdet.frequencyvector = np.geomspace(1., f_max, num=n_f)
    masses = np.logspace(0.5, 6.5, num=150)
    
    detectors_colors = {
        # LGWA.gdet: 'blue',
        ET.gdet: 'red'
    }
    
    snr_list = [10, 20, 40, 100]
    # snr_list = [10, 20]
    
    for detector, color in tqdm(detectors_colors.items(), unit="detectors"):
        for snr in tqdm(snr_list, leave=False, unit="SNRs"):
            plot_single(masses, detector, snr, color=color, alpha=.2)
    plt.xscale('log')
    plt.title('Horizon for equal-mass BBH')
    plt.xlabel('Total binary mass $M$ [$M_{\odot}$]')
    plt.ylabel('Redshift $z$')
    plt.show()

if __name__ == '__main__':
    plot_all()