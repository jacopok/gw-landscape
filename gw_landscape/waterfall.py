from .gwfish import GWFishDetector
from GWFish.modules.horizon import horizon, compute_SNR, find_optimal_location, Network
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from labellines import labelLine
from tqdm import tqdm
from .plot import FIG_PATH, make_redshift_distance_axes

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

def compute_horizon_from_masses(params, masses, gwfish_detector, SNR, detector_frame_trick=True, recompute_location=False, optimal_locations=None):

    redshifts = []
    
    if optimal_locations is not None:
        iterator = tqdm(zip(masses, optimal_locations), leave=False, unit="Masses")
    else:
        iterator = tqdm(masses, leave=False, unit="Masses")
    
    for obj in iterator:
        
        if optimal_locations is None:
            mass = obj
        else:
            mass, (ra, dec) = obj
            params['ra'] = ra
            params['dec'] = dec
        
        if recompute_location:
            params = find_optimal_location(params | {'mass_1': mass/2., 'mass_2': mass/2.}, gwfish_detector, maxiter=20)
        
        try:
            distance, redshift = horizon(
                params = params | {'mass_1': mass/2., 'mass_2': mass/2.},
                detector = gwfish_detector,
                target_SNR = SNR,
                waveform_model='IMRPhenomD',
                source_frame_masses=not detector_frame_trick,
            )
        except ValueError as e:
            redshift = 0.
            # raise e
        redshifts.append(redshift)
    
    if detector_frame_trick:
        source_frame_masses = np.array(masses)/(1+np.array(redshifts))
    else:
        source_frame_masses = masses
    
    return source_frame_masses, redshifts

def plot_snr_area(params, masses, gwfish_detector, SNR, label_line, ax=None, **plot_kwargs):
    
    source_frame_masses, redshifts = compute_horizon_from_masses(params, masses, gwfish_detector, SNR)
    
    if ax is None:
        ax = plt.gca()
    
    ax.fill_between(source_frame_masses, redshifts, **plot_kwargs)
    if not label_line:
        return
    ax.plot(source_frame_masses, redshifts, alpha=0., c='black')
    label_position = source_frame_masses[np.argmax(redshifts)] / 3
    if label_position < source_frame_masses[0]:
        label_position = source_frame_masses[0] 
    if label_position > source_frame_masses[-1]:
        label_position = source_frame_masses[-1]
    labelLine(
        ax.get_lines()[-1], 
        label_position,
        label=f'SNR={SNR}',
        align=True, 
        outline_color='none',
        # yoffset=-1,
        fontsize=7,
    )

def plot_all(fig_path, log=False):
    LISA = GWFishDetector('LISA')
    LGWA = GWFishDetector('LGWA')
    ET = GWFishDetector('ET')
    detectors_colors = {
        ET.gdet: 'red',
        LGWA.gdet: 'blue',
        LISA.gdet: 'green',
    }
    masses = np.logspace(0.5, 7.5, num=150)

    base_params = {
        "theta_jn": 0.,
        "ra": 0.,
        "dec": 0.,
        "psi": 0.,
        "phase": 0.,
        "geocent_time": 1800000000,
    }
    
    # snr_list = [9, 10]
    snr_list = [10, 30, 100, 300]
    
    ax_redshift, ax_distance = make_redshift_distance_axes()
    
    label_line=True
    for detector, color in tqdm(detectors_colors.items(), unit="detectors"):
        label=detector.name
        
        params = find_optimal_location(base_params | {'mass_1': 100., 'mass_2': 100.}, detector)
        
        for snr in tqdm(snr_list, leave=False, unit="SNRs"):
            plot_snr_area(params, masses, detector, snr, label_line, color=color, alpha=.2, label=label, ax=ax_redshift)
            label=None
        label_line = False

    ax_redshift.legend()
    ax_redshift.set_xscale('log')
    ax_redshift.set_xlim(masses[0], masses[-1])
    
    if log:
        ax_redshift.set_yscale('log')
        ax_redshift.set_ylim(2e-1, 4e2)
    else:
        ax_redshift.set_ylim(0, 20)
    
    ax_redshift.set_title('Horizon for equal-mass BBH')
    ax_redshift.set_xlabel('Total binary mass $M$ [$M_{\odot}$]')
    ax_redshift.set_ylabel('Redshift $z$')
    plt.savefig(fig_path, dpi=400)
    plt.close()
    

def plot_network(fig_path, log=False):
    ET_LGWA = Network(['ET', 'LGWA'])

    masses = np.logspace(0.5, 6.5, num=50)
    
    base_params = {
        "theta_jn": 0.,
        "ra": 0.,
        "dec": 0.,
        "psi": 0.,
        "phase": 0.,
        "geocent_time": 1800000000,
    }
    
    snr_list = [10, 30, 100, 300]
    
    label_line=True
    label='ET+LGWA'
    color = 'orange'
    
    params = find_optimal_location(base_params | {'mass_1': 100., 'mass_2': 100.}, ET_LGWA)
    print(params)
    
    for snr in tqdm(snr_list, leave=False, unit="SNRs"):
        plot_snr_area(params, masses, ET_LGWA, snr, label_line, color=color, alpha=.2, label=label)
        label=None
    label_line = False

    plt.legend()
    plt.xscale('log')
    plt.xlim(masses[0], masses[-1])
    
    if log:
        plt.yscale('log')
        plt.ylim(2e-1, 4e2)
    else:
        plt.ylim(0, 20)
    
    plt.title('Horizon for equal-mass BBH')
    plt.xlabel('Total binary mass $M$ [$M_{\odot}$]')
    plt.ylabel('Redshift $z$')
    plt.savefig(fig_path, dpi=400)
    plt.close()

if __name__ == '__main__':
    # plot_all(FIG_PATH / 'horizon.png', log=False)
    # plot_network(FIG_PATH / 'horizon_log_network.png', log=True)
    plot_all(FIG_PATH / 'horizon_log.png', log=True)
