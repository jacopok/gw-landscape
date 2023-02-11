#! /bin/bash/python3
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from .gwfish import GWFishDetector
from .lisa import LISA
from .pta import PTA

FIG_PATH = Path(__file__).resolve().parent.parent / 'plots'

if __name__ == '__main__':
    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })
    
    detector_list = [
        GWFishDetector('ET'),
        GWFishDetector('VIR'),
        LISA(),
        GWFishDetector('LGWA'),
        PTA(),
    ]
    
    cmap = plt.get_cmap('Dark2')
    colors = (cmap(index) for index in np.linspace(0, 1, num=8))
    plt.gca().set_prop_cycle(color=colors)
    
    for detector in detector_list:
        ls = None if detector.working else '--'
        lines = plt.plot(detector.frequencies, detector.characteristic_strain, ls=ls, lw=2)
        plt.annotate(detector.name, detector.annotation_place, color=lines[0].get_color())

    plt.xscale('log')
    plt.yscale('log')
    ax_freq = plt.gca()
    
    ax_time = plt.gca().secondary_xaxis('top', functions=(lambda x: 1/x, lambda x: 1/x))
    ax_time.set_xlabel('Period [s]')
    ax_freq.set_xlabel('Frequency [Hz]')
    ax_freq.set_ylabel('Characteristic noise strain')
    
    
    for ax in [
        ax_freq.get_xaxis(), 
        ax_time.get_xaxis(), 
        ax_freq.get_yaxis(),
    ]: 
        ax.set_major_locator(matplotlib.ticker.LogLocator(subs='all', numticks=10000))
        ax.set_minor_locator(matplotlib.ticker.LogLocator(subs=np.arange(1, 10), numticks=10000))
    
    plt.grid('on')

    plt.savefig(FIG_PATH / 'landscape.png', dpi=200)