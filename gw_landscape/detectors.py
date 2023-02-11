from abc import ABC, abstractmethod
import numpy as np

class Detector(ABC):
    
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
        max_strain_idx = np.argmax(self.characteristic_strain)
        
        if self.name == 'Virgo':
            return (self.frequencies[max_strain_idx]/15., self.characteristic_strain[max_strain_idx])
            
        
        return (self.frequencies[max_strain_idx]*1.5, self.characteristic_strain[max_strain_idx])