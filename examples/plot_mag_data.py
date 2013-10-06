import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import auroraplot as ap
import auroraplot.dt64tools as dt64
import auroraplot.magdata
import auroraplot.datasets.aurorawatchnet


import scipy.stats

plt.close('all')
st = np.datetime64('2013-09-25T09:00:00+0000')
et = np.datetime64('2013-09-26T09:00:00+0000')

st = np.datetime64('2013-09-20T00:00:00+0000')
et = np.datetime64('2013-09-21T00:00:00+0000')

mdl = []
for site in ['LAN1', 'LAN3', 'METOFFICE1']:
    md = ap.load_data('AURORAWATCHNET', site, 'MagData', st, et)
    if md is not None:
        mdl.append(md)

ap.magdata.stack_plot(mdl, 100e-9)
plt.grid(True)
plt.show()
