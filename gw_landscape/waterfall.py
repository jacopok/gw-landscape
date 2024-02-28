from .gwfish import GWFishDetector
from GWFish.modules.horizon import horizon, compute_SNR, find_optimal_location, Network
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from labellines import labelLine
from tqdm import tqdm
from pathlib import Path
import hashlib
from .plot import FIG_PATH, make_redshift_distance_axes
# picked with https://colorbrewer2.org/#type=qualitative&scheme=Set1&n=3
ET_COLOR = '#4daf4a'
LGWA_COLOR = '#377eb8'
LISA_COLOR = '#e41a1c'


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
    
    params_tuples = sorted(params.items(), key=lambda x: x[0])
    params_reprs = [f'{key}={val:.3e}' for key, val in params_tuples]
    
    args_hash_tuple = ((
        *params_reprs, 
        detector_frame_trick, 
        recompute_location,
        optimal_locations
    ))
    
    hasher = hashlib.md5()
    for item in args_hash_tuple:
        hasher.update(repr(item).encode())
    
    args_hash = hasher.hexdigest()[:10]
    
    filename = f'{gwfish_detector.name}_{SNR=}_{args_hash}.npy'
    
    if Path(filename).is_file():
        print(f'Loading {filename}')
        masses, redshifts = np.load(filename)
        return masses, redshifts
    print(f'Computing, will save to {filename}')
    
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
        
        mass_ratio = params.get('mass_ratio', 1.)
        
        mass_1 = mass / (1 + mass_ratio)
        mass_2 = mass * mass_ratio / (1 + mass_ratio)
        
        if recompute_location:
            params = find_optimal_location(params | {'mass_1': mass_1, 'mass_2': mass_2}, gwfish_detector, maxiter=20, seed=1)
        
        try:
            distance, redshift = horizon(
                params = params | {'mass_1': mass_1, 'mass_2': mass_2},
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
    
    np.save(filename, (source_frame_masses, redshifts))
    
    return source_frame_masses, redshifts

def plot_snr_area(params, masses, gwfish_detector, SNR, label_line, ax=None, **plot_kwargs):
    
    source_frame_masses, redshifts = compute_horizon_from_masses(params, masses, gwfish_detector, SNR, recompute_location=True)
    
    if ax is None:
        ax = plt.gca()
    
    ax.fill_between(source_frame_masses, redshifts, **plot_kwargs)
    if not label_line:
        return
    ax.plot(source_frame_masses, redshifts, alpha=0.5, c=plot_kwargs.get('color', None))
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
        color='black',
        # outline_color='none',
        # yoffset=-1,
        fontsize=7,
    )

def plot_all(fig_path, log=False, extra_dict=None):
    LISA = GWFishDetector('LISA')
    LGWA = GWFishDetector('LGWA')
    ET = GWFishDetector('ET')

    detectors_colors = {
        LGWA.gdet: LGWA_COLOR,
        ET.gdet: ET_COLOR,
        LISA.gdet: LISA_COLOR,
    }
    masses = np.logspace(-0, 9, num=200)

    base_params = {
        "theta_jn": 0.,
        "ra": 0.,
        "dec": 0.,
        "psi": 0.,
        "phase": 0.,
        "geocent_time": 1800000000,
    }
    
    if extra_dict is not None:
        base_params.update(extra_dict)
    
    # snr_list = [9, 10]
    # snr_list = [8, 50, 250, 1000]
    snr_list = [8, 50, 250]
    
    ax_redshift, ax_distance = make_redshift_distance_axes(log=log)
    
    label_line=True
    for detector, color in tqdm(detectors_colors.items(), unit="detectors"):
        label=detector.name
        
        # params = find_optimal_location(base_params | {'mass_1': 100., 'mass_2': 100.}, detector)
        
        for snr in tqdm(snr_list, leave=False, unit="SNRs"):
            alpha = 0.2 if detector.name == 'LGWA' else 0.08
            plot_snr_area(base_params, masses, detector, snr, label_line, color=color, alpha=alpha, label=label, ax=ax_redshift)
            label=None
        label_line = False

    ax_redshift.legend()
    ax_redshift.set_xscale('log')
    ax_redshift.set_xlim(masses[0], masses[-1]/50)
    
    if log:
        ax_redshift.set_ylim(1e-2, 5e2)
    else:
        ax_redshift.set_ylim(0, 12)
        # ax_redshift.set_yscale('function', functions=(
        #     lambda x : np.sqrt(x),
        #     lambda x : x**2, 
        # ))
    
    if 'mass_ratio' in base_params:
        ax_redshift.set_title('Horizon for equal-mass BBH')
    else:
        ax_redshift.set_title(f'Horizon for BBH with a mass ratio of {mass_ratio:.0f}')
    
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
    
    params = find_optimal_location(base_params | {'mass_1': 100., 'mass_2': 100.}, ET_LGWA, seed=1)
    
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
    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Serif",
        "font.size": 13,
        "axes.grid": True,
    })

    # plot_all(FIG_PATH / 'horizon.png', log=False)
    # plot_network(FIG_PATH / 'horizon_log_network.png', log=True)
    plot_all(FIG_PATH / 'horizon_q1.pdf', log=False, extra_dict={'mass_ratio': 1.})
    plot_all(FIG_PATH / 'horizon_q2.pdf', log=False, extra_dict={'mass_ratio': 2.})
    plot_all(FIG_PATH / 'horizon_q5.pdf', log=False, extra_dict={'mass_ratio': 5.})
    plot_all(FIG_PATH / 'horizon_q10.pdf', log=False, extra_dict={'mass_ratio': 10.})
