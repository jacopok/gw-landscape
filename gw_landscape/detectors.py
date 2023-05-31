from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
DATA_FOLDER = Path(__file__).parent / 'data'

class Detector(ABC):
    
    def __init__(self):
        self._annotation_place = None
    
    @property
    @abstractmethod
    def frequencies(self) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def psd(self) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def working(self) -> bool:
        ...

    @property
    def characteristic_strain(self) -> np.ndarray:
        return np.sqrt(self.frequencies * self.psd)
    
    @property
    def annotation_place(self) -> tuple[float, float]:
        if self._annotation_place is not None:
            return self._annotation_place
        
        max_strain_idx = np.argmax(self.characteristic_strain)
        
        self._annotation_place = (self.frequencies[max_strain_idx]*1.5, self.characteristic_strain[max_strain_idx])
        if self.name == 'Virgo':
            self._annotation_place = (self.frequencies[max_strain_idx]/15., self.characteristic_strain[max_strain_idx])
        
        return self._annotation_place
    
    def psd_interp(self, frequencies):
        return interp1d(self.frequencies, self.psd)(frequencies)
    
    @annotation_place.setter
    def annotation_place(self, value):
        self._annotation_place = value
        
    def snr_integrand(self, signal_projection, frequencies, log_base=np.e):
        """takes as input the signal projection, h = h+F+ + hxFx
        in the frequency domain, and returns 
        and returns the snr integrand 4*f * h**2 / S_n, whose integral
        in d(logf) is the overall SNR
        if the log_base is different from e, return a quantity that should be integrated
        in d(log_b f).
        """
        mean_square_signal = np.sum(abs(signal_projection)**2, axis=1)
        psd = self.psd_interp(frequencies)

        return 4 * mean_square_signal * frequencies / psd * np.log(log_base)