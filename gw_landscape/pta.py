import numpy as np
from .detectors import Detector

class PTA(Detector):
    
    def __init__(self, delta_t: float = 1/1e-6, obs_time: float = 1/5e-9):
        self.delta_t = delta_t
        self.obs_time = obs_time
        self.pivot_sensitivity = 1e-23
    
    @property
    def frequencies(self):
        return np.geomspace(1/self.obs_time, 1/self.delta_t)
    
    @property
    def psd(self):
        return self.pivot_sensitivity * (self.frequencies / self.frequencies[0])
    
    @property
    def name(self):
        return "PTA"

    @property
    def working(self):
        return True
