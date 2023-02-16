"""
This code is based on a review by [Renzini+2022](https://arxiv.org/abs/2202.00178),
and it makes use of a formalism by [Hazboun+2019](https://arxiv.org/abs/1907.04341).

It uses the python package [hasasia](http://dx.doi.org/10.21105/joss.01775).
"""

import numpy as np
from .detectors import Detector
import hasasia.sensitivity as hsen
import hasasia.sim as hsim


class PPTA(Detector):
    
    def __init__(self):
        rng = np.random.default_rng(192)
        phi = rng.uniform(0, 2*np.pi,size=26)
        cos_theta = rng.uniform(-1,1,size=26)
        #This ensures a uniform distribution across the sky.
        theta = np.arccos(cos_theta)

        timespan=np.array([15.0, 14.2, 14.2, 7.8, 14.2, 14.1, 14.2, 12.3, 7.4, 7.0, 14.2, 14.2, 14.2, 14.2, 14.2, 7.2, 14.2, 13.8, 5.4, 14.2, 14.2, 14.1, 14.2, 13.9, 14.1, 8.2])
        sigma=np.array([0.59, 1.04, 0.97, 0.65, 1.62, 9.16, 3.08, 0.87, 1.29, 0.83, 0.58, 1.25, 2.76, 0.32, 1.05, 2.15, 0.46, 24.05, 0.67, 2.19, 0.24, 3.18, 2.31, 0.77, 0.98, 0.26])*1e-6
        log10_A_rn = [-14.56, -14.26, -13.04, -30.0, -30.0, -30.0, -30.0,  -30.0,  -30.0,  -30.0, -14.34,  -30.0, -12.85, -30.0, -30.0, -30.0, -30.0, -13.26, -30.0, -16.86, -14.74, -14.33, -30.0, -30.0, -30.0, -30.0]
        A_rn = 10**np.array(log10_A_rn)
        gamma_rn = [2.99, 4.17, 1.09, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,3.81, 0.01,0.98, 0.01, 0.01, 0.01, 0.01,5.02, 0.01, 7.49, 4.05, 5.39,  0.01, 0.01, 0.01, 0.01]
        alpha_rn = (3 - np.array(gamma_rn))/2
        psrs = hsim.sim_pta(timespan=timespan,cad=26,sigma=sigma, #1e-7,
                            phi=phi,theta=theta,
                            A_rn=A_rn,alpha=alpha_rn,freqs=self.frequencies)

        spectra = []
        for p in psrs:
            sp = hsen.Spectrum(p, freqs=self.frequencies)
            sp.NcalInv
            spectra.append(sp)

        self.sensitivity_curve = hsen.DeterSensitivityCurve(spectra)

    @property
    def frequencies(self):
        return np.logspace(np.log10(5e-10),np.log10(5e-7),500)
    
    @property
    def psd(self):
        return self.sensitivity_curve.S_eff
    
    @property
    def name(self):
        return "PPTA DR2"

    @property
    def working(self):
        return True

class SKA(Detector):

    def __init__(self):
        n_4 = 50
        rng = np.random.default_rng(192)
        phi_4 = rng.uniform(0, 2*np.pi,size=50)
        cos_theta_4 = rng.uniform(-1,1,size=50)
        theta_4 = np.arccos(cos_theta_4)

        gamma_rn_4 = rng.uniform(1,7,size=n_4) #truncnorm.rvs(a=0,b=10,loc=4,scale=2,size=n_4)
        log10_A_rn_4 = rng.uniform(-20,-14,size=n_4)
        A_rn_4 = 10**np.array(log10_A_rn_4)
        alpha_rn_4 = (3 - np.array(gamma_rn_4))/2

        psrs4 = hsim.sim_pta(timespan=15.,cad=26.,sigma=30e-9, #1e-7,
                            phi=phi_4,theta=theta_4,
                            A_rn=A_rn_4,alpha=alpha_rn_4,freqs=self.frequencies)
        spectra4 = []
        for p in psrs4:
            sp = hsen.Spectrum(p, freqs=self.frequencies)
            sp.NcalInv
            spectra4.append(sp)
        
        self.sensitivity_curve = hsen.DeterSensitivityCurve(spectra4)

    @property
    def frequencies(self):
        return np.logspace(np.log10(5e-10),np.log10(5e-7),500)
    
    @property
    def psd(self):
        return self.sensitivity_curve.S_eff
    
    @property
    def name(self):
        return "SKA 15yr"

    @property
    def working(self):
        return False
    
    @property
    def annotation_place(self) -> tuple[float, float]:
        
        return (self.frequencies[-1]*1.5, self.characteristic_strain[-1])