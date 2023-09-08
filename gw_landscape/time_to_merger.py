from .plot_signals import time_to_merger, chirp_mass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm, colors


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Serif"
})

mass_grid = np.geomspace(1, 1e7, num=200)
freq_grid = np.geomspace(1e-5, 1e2, num=200)

F, M = np.meshgrid(freq_grid, mass_grid)

M_CHIRP = chirp_mass(M/2, M/2)

T = time_to_merger(F, M_CHIRP)

times = {
        1: 'second',
        60: 'minute',
        3600: 'hour',
        3600*24: 'day',
        3600*24*30: 'month',
        3600*24*365.24: 'year',
        3600*24*365.24*10: '10 years',
}

plt.contourf(F, M, T, levels=list(times.keys()), cmap=cm.PuBu_r, norm=colors.LogNorm())

# plt.axvline(x=0.05, c='black')
# plt.axvline(x=2, c='black')
plt.fill_betweenx([mass_grid[0], mass_grid[-1]], 0.05, 2, alpha=.2, color='red')

mchirp_170817 = 1.188
m_170817 = 1.188 * 2**(6/5)
m_190521 = 150
plt.axhline(m_170817, c='black')
plt.text(2e-5, m_170817*1.3, 'GW170817')
plt.axhline(m_190521, c='black')
plt.text(2e-5, m_190521*1.3, 'GW190521')
plt.text(1e-1, 4e6, 'decihertz')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Total binary mass [$M_\odot$]')
plt.colorbar(format=ticker.FixedFormatter(list(times.values())), label='Time to merger')
plt.savefig('time_to_merger.pdf')