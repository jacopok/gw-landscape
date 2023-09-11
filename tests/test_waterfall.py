import numpy as np
import matplotlib.pyplot as plt
from gw_landscape.waterfall import compute_horizon_from_masses
from GWFish.modules.detection import Network
import pytest


def test_source_frame_trick():
    
    network = Network(['ET', 'LGWA'])
    
    masses = np.geomspace(10, 1e6, num=100)
    params = {
        "theta_jn": 0.,
        "ra": 0.,
        "dec": 0.,
        "psi": 0.,
        "phase": 0.,
        "geocent_time": 1800000000,
    }
    mass1, hor1 = compute_horizon_from_masses(params, masses, network, 10, detector_frame_trick=True)
    mass2, hor2 = compute_horizon_from_masses(params, masses, network, 10, detector_frame_trick=False)
    
    plt.loglog(mass1, hor1)
    plt.loglog(mass2, hor2)
    plt.show()