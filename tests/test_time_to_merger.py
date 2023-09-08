from gw_landscape.plot_signals import time_to_merger, inverse_time_to_merger
import numpy as np

def test_inversion():
    
    assert np.isclose(100., time_to_merger(inverse_time_to_merger(100)))
    assert np.isclose(1., inverse_time_to_merger(time_to_merger(1.)))

def test_inversion_different_mchirp():
    mchirp = 0.5
    
    assert np.isclose(100., time_to_merger(inverse_time_to_merger(100, mchirp=mchirp), mchirp=mchirp))
    assert np.isclose(1., inverse_time_to_merger(time_to_merger(1., mchirp=mchirp), mchirp=mchirp))

if __name__ == '__main__':
    test_inversion()
    test_inversion_different_mchirp()