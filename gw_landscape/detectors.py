from abc import ABC, abstractmethod
import numpy as np

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
    
    @annotation_place.setter
    def annotation_place(self, value):
        self._annotation_place = value