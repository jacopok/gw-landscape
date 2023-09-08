from .detectors import Detector, DATA_FOLDER
from GWFish.modules.detection import Detector as GDetector
import numpy as np
from pathlib import Path

class GWFishDetector(Detector):
    
    def __init__(self, name: str, **kwargs):
        super().__init__()
        self.gdet = GDetector(name, 
                              parameters=[], fisher_parameters=[], 
                              config=DATA_FOLDER /'detectors_tweaked.yaml', 
                              **kwargs)
        
        self._psd = self.gdet.components[0].psd_data[:, 1]
        # self._psd = np.inf
        # for component in self.gdet.components: 
        #     self._psd = 1/np.sqrt(self._psd**(-2) + component.psd_data[:, 1]**(-2))

    @property
    def frequencies(self):
        return self.gdet.components[0].psd_data[:, 0]
    
    @property
    def psd(self):
        return self._psd
    
    @property
    def name(self):
        if self.gdet.name == 'VIR':
            return 'Virgo'
        if self.gdet.name == 'LGWA_Soundcheck':
            return 'Soundcheck'
        return self.gdet.name
    
    @property
    def working(self):
        return self.gdet.name in ['VIR', 'LHO', 'LLO']