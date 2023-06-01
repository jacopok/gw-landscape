import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import warnings
from functools import partial
from scipy.interpolate import interp1d
from .gwfish import GWFishDetector
from .lisa import LISA
from .einstein_telescope import EinsteinTelescopeCryo, EinsteinTelescopeHF
from .plot import plot_characteristic_noise_strain, FIG_PATH, make_time_axis_fancy, make_frequency_period_axes, set_color_cycle
from GWFish.modules.waveforms import LALFD_Waveform
from GWFish.modules.detection import projection
from GWFish.modules.horizon import compute_SNR


T_20_HZ = 157.86933774
REF_FREQ = 20.
REF_MCHIRP = 1.2187707886145736

def make_time_to_merger_axis_secondary(mchirp, include_month=False):
    """ This function will create a secondary, "detached" axis
    on top of the image.
    """
    ax_time2 = plt.gca().secondary_xaxis('top', functions=(
        partial(time_to_merger, mchirp=mchirp), 
        partial(inverse_time_to_merger, mchirp=mchirp)
    ))
    ax_time2.set_xlabel('Time to merger')
    plt.gcf().subplots_adjust(top=0.75)

    # Offset the right spine of par2.  The ticks and label have already been
    # placed on the right by twinx above.
    ax_time2.spines["top"].set_position(('outward', 35.))
    # Having been created by twinx, par2 has its frame off, so the line of its
    # detached spine is invisible.  First, activate the frame but make the patch
    # and spines invisible.
    make_patch_spines_invisible(ax_time2)
    # Second, show the right spine.
    ax_time2.spines["top"].set_visible(True)
    
    subdivisions = {
        # 1e-3: 'millisecond',
        1.: 'second',
        60.: 'minute',
        3600.: 'hour',
        3600*24.: 'day',
        3600*24*365.24: 'year',
        1e3*3600*24*365.24: '$10^3$ years',
        1e6*3600*24*365.24: '$10^6$ years',
    }
    if include_month:
        subdivisions[3600*24*30.] = 'month'
    
    make_time_axis_fancy(ax_time2, subdivisions)

def make_time_to_merger_axis(mchirp, subdivisions=None):
    ax_time2 = plt.gca().secondary_xaxis('top', functions=(
        partial(time_to_merger, mchirp=mchirp), 
        partial(inverse_time_to_merger, mchirp=mchirp)
    ))
    ax_time2.set_xlabel('Time to merger')
    
    if subdivisions is None:
        subdivisions = {
            1.: 'second',
            60.: 'minute',
            3600.: 'hour',
            3600*24.: 'day',
            3600*24*365.24: 'year',
            1e3*3600*24*365.24: '$10^3$ years',
            1e6*3600*24*365.24: '$10^6$ years',
        }
        
    make_time_axis_fancy(ax_time2, subdivisions)

def time_to_merger(f, mchirp = REF_MCHIRP):
    # time in seconds, frequency in Hz, chirp mass in solar masses
    return T_20_HZ * (f / REF_FREQ)**(-8/3) * (mchirp / REF_MCHIRP)**(-5/3)

def inverse_time_to_merger(t, mchirp = REF_MCHIRP):
    return REF_FREQ * (t / T_20_HZ)**(-3/8) * (mchirp / REF_MCHIRP)**(-5/8)

def chirp_mass(m1, m2):
    return (m1*m2)**(3/5) / (m1 + m2)**(1/5)

def get_projection(params, gwfish_detector, frequencies=None, waveform_model='IMRPhenomD'):

    if frequencies is None:
        frequencies = gwfish_detector.frequencyvector
    else:
        frequencies = frequencies[:, None]
    data_params = {
        'frequencyvector': frequencies,
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
    return signal, polarizations

def plot_characteristic_signal_strain(params, gwfish_detector, waveform_model='IMRPhenomD', **plot_kwargs):
    
    signal, _ = get_projection(params, gwfish_detector, waveform_model=waveform_model)
    
    mean_square_signal = np.sqrt(np.sum(abs(signal)**2, axis=1))[:, np.newaxis]
    
    mask = mean_square_signal > 0
    plt.plot(
        gwfish_detector.frequencyvector[mask], 
        2 * mean_square_signal[mask] 
        * gwfish_detector.frequencyvector[mask], 
        **plot_kwargs)


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

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

if __name__ == '__main__':
    mass = 3e3
    plot_bbh(mass, FIG_PATH / f'sensitivities_{mass:.0e}.png')