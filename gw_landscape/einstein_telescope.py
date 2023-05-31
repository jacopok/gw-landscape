from .detectors import Detector, DATA_FOLDER
from pathlib import Path
import numpy as np

ET_DATA_FILE = DATA_FOLDER / '18213_ET10kmcolumns.txt'
# columns are: frequencies, HF, LF, HF+LF

class EinsteinTelescopeHF(Detector):
    
    data_file = np.loadtxt(ET_DATA_FILE)
    
    @property
    def name(self):
        return 'ET-HF'
    
    @property
    def working(self):
        return False
    
    @property
    def frequencies(self):
        return self.data_file[:, 0]
    
    @property
    def psd(self):
        return self.data_file[:, 1]
    

class EinsteinTelescopeCryo(Detector):
    
    data_file = np.loadtxt(ET_DATA_FILE)
    
    @property
    def name(self):
        return 'ET-LF'

    @property
    def working(self):
        return False
    
    @property
    def frequencies(self):
        return self.data_file[:, 0]
    
    @property
    def psd(self):
        return self.data_file[:, 2]