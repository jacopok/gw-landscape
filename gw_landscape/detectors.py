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