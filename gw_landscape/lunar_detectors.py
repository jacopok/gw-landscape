from .gwfish import GWFishDetector
from GWFish.modules.horizon import horizon, compute_SNR, find_optimal_location, Network
import numpy as np
import matplotlib.pyplot as plt
from labellines import labelLine
from tqdm import tqdm
from pathlib import Path
from scipy.interpolate import interp1d
from .plot import FIG_PATH, plot_characteristic_noise_strain, make_redshift_distance_axes
from .plot_signals import plot_characteristic_signal_strain
from .waterfall import find_optimal_location, plot_snr_area
from .lisa import LISA

def get_detector_list(lunar_only=False):
    
    lgwa = GWFishDetector('LGWA')
    lsga = GWFishDetector('LBI-GND')
    lvirgo = GWFishDetector('LBI-SUS')
    et = GWFishDetector('ET')
    lisa = LISA()
    
    lgwa.annotation_place = (lgwa.frequencies[-1]/10, lgwa.characteristic_strain[-1])
    lvirgo.annotation_place = (lvirgo.frequencies[-1]/8, lvirgo.characteristic_strain[-1])
    lsga.annotation_place = (lsga.frequencies[-1]/10, lsga.characteristic_strain[-1]*2)
    lisa.annotation_place = (3e-5, 3e-17)
    
    colors = ['blue', 'red', 'green', 'black', 'black']
    
    detector_list = [
        lsga,
        lgwa,
        lvirgo,
        lisa,
        et,
    ]
    detectors_colors = {det: color for det, color in zip(detector_list, colors)}
    if lunar_only:
        del detectors_colors[lisa]
        detector_list.remove(lisa)
        del detectors_colors[et]
        detector_list.remove(et)

    return detector_list, detectors_colors

def get_lisa_lbi_gnd():
    psd_path = psd_path=Path(__file__).parent/'data'
    
    lsga = GWFishDetector('LBI-GND', psd_path=psd_path)
    lisa = GWFishDetector('LISA', psd_path = PSD_PATH)
    
    for component in lsga.gdet.components:
        component.psd_data[:, 1] = np.maximum(component.psd_data[:, 1], 1e-37)
        component.Sn = interp1d(component.psd_data[:, 0], component.psd_data[:, 1], bounds_error=False, fill_value=1.)
    
    return [
        lsga, 
        lisa
    ], {
        lsga: 'green',
        lisa: 'blue',
    }


def preliminary_stamp():
    plt.gcf().text(0.75, 0.8, 'Preliminary',
         fontsize=50, color='red',
         ha='right', va='bottom', alpha=0.3)

def plot_psds(fig_path, detector_list, colors):
    
    plot_characteristic_noise_strain(detector_list, colors=colors.values())

    plt.xlim(2e-5, 1e2)
    plt.ylim(1e-24, 1e-15)
    plt.savefig(fig_path, dpi=200)
    plt.close()


def plot_waterfall(fig_path, detector_list, detectors_colors, log=False):
    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Serif"
    })
    

    masses = np.logspace(.5, 11, num=400)

    base_params = {
        "theta_jn": 0.,
        "ra": 0.,
        "dec": 0.,
        "psi": 0.,
        "phase": 0.,
        "geocent_time": 1800000000,
    }
    
    snr_list = [10]
    
    label_line=True
    for detector, color in tqdm(detectors_colors.items(), unit="detectors"):
        label=detector.gdet.name
        
        params = find_optimal_location(base_params | {'mass_1': 1e6, 'mass_2': 1e6}, detector.gdet)
        print(params)
        
        for snr in tqdm(snr_list, leave=False, unit="SNRs"):
            plot_snr_area(params, masses, detector.gdet, snr, label_line, color=color, alpha=.2, label=label)
            label=None
        label_line = False

    plt.xscale('log')
    plt.xlim(masses[0], masses[-1])
    
    if log:
        plt.yscale('log')
        plt.ylim(2e-1, 5e2)
    else:
        plt.ylim(0, 20)
    plt.xlim(np.sqrt(10), np.sqrt(10)*1e7)
    
    plt.title('Horizon for equal-mass BBH')
    make_redshift_distance_axes()
    plt.xlabel('Source-frame total binary mass $M$ [$M_{\odot}$]')
    plt.ylabel('Redshift $z$')
    plt.legend(loc='upper right')
    
    plt.savefig(fig_path, dpi=400)
    plt.close()

if __name__ == '__main__':
    
    plot_psds(
        FIG_PATH / 'landscape_lunar.pdf', 
        *get_detector_list(lunar_only=False)
    )
    
    plot_waterfall(
        FIG_PATH / 'lunar_waterfall.pdf', 
        *get_detector_list(lunar_only=True), 
        log=True
    )
