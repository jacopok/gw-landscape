#! /bin/bash/python3
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from .gwfish import GWFishDetector
from .lisa import LISA
from .pta import PPTA, SKA
from .detectors import Detector

FIG_PATH = Path(__file__).resolve().parent.parent / 'plots'

def plot_characteristic_noise_strain(detector_list: list[Detector]):
    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Serif"
    })
    
    cmap = plt.get_cmap('Dark2')
    colors = (cmap(index) for index in np.linspace(0, 1, num=8))
    plt.gca().set_prop_cycle(color=colors)
    
    for detector in detector_list:
        ls = '-' if detector.working else '--'
        lines = plt.plot(detector.frequencies, detector.characteristic_strain, ls=ls, lw=1.5)
        plt.annotate(detector.name, detector.annotation_place, color=lines[0].get_color())

    plt.xscale('log')
    plt.yscale('log')
    ax_freq = plt.gca()
    
    ax_time = plt.gca().secondary_xaxis('top', functions=(lambda x: 1/x, lambda x: 1/x))
    ax_time.set_xlabel('Period')
    ax_freq.set_xlabel('Frequency [Hz]')
    ax_freq.set_ylabel('Characteristic noise strain')
    
    
    for ax in [
        ax_freq.get_xaxis(), 
        # ax_time.get_xaxis(), 
        ax_freq.get_yaxis(),
    ]: 
        ax.set_major_locator(matplotlib.ticker.LogLocator(subs=[1], numticks=10000))
        ax.set_minor_locator(matplotlib.ticker.LogLocator(subs=np.arange(1, 10), numticks=10000))
    
    times = {
        1e-3: 'millisecond',
        1: 'second',
        60: 'minute',
        3600: 'hour',
        3600*24: 'day',
        3600*24*30: 'month',
        3600*24*365.24: 'year',
    }
    
    ax_time.get_xaxis().set_major_locator(matplotlib.ticker.FixedLocator(list(times.keys())))
    ax_time.get_xaxis().set_major_formatter(matplotlib.ticker.FixedFormatter(list(times.values())))
    
    ax_freq.grid(visible=True, which='major')
    ax_freq.grid(visible=False, which='minor')

def plot_landscape():
    detector_list = [   
        GWFishDetector('ET'),
        GWFishDetector('VIR'),
        LISA(),
        GWFishDetector('LGWA'),
        PPTA(),
        SKA(),
    ]
    plot_characteristic_noise_strain(detector_list)
    plt.savefig(FIG_PATH / 'landscape.png', dpi=200)
    plt.close()

def plot_hf():
    
    detector_list_hf = [   
        GWFishDetector('ET'),
        GWFishDetector('VIR'),
        LISA(),
    ]
    plot_characteristic_noise_strain(detector_list_hf)
    plt.savefig(FIG_PATH / 'landscape_hf.png', dpi=200)
    plt.close()
    
def plot_lgwa():
    
    lgwa = GWFishDetector('LGWA')
    soundcheck = GWFishDetector('LGWA_Soundcheck')
    
    lgwa.annotation_place = (lgwa.frequencies[-1]*1.5, lgwa.characteristic_strain[-1])
    soundcheck.annotation_place = (soundcheck.frequencies[-1]*1.5, soundcheck.characteristic_strain[-1])
    
    detector_list_hf = [
        GWFishDetector('ET'),
        GWFishDetector('VIR'),
        LISA(),
        lgwa,
        soundcheck,
    ]
    plot_characteristic_noise_strain(detector_list_hf)
    plt.savefig(FIG_PATH / 'landscape_lgwa.png', dpi=200)
    plt.close()
    
if __name__ == '__main__':
    
    plot_landscape()
    plot_hf()
    plot_lgwa()