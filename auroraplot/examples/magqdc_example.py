import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import auroraplot as ap
import auroraplot.dt64tools as dt64
import auroraplot.magdata

plt.close('all')
t = np.datetime64('2013-09-01T00:00:00+0000')
qdc = ap.magdata.load_qdc('AURORAWATCHNET', 'LAN1', t)

qdc.plot()
plt.title('AURORAWATCHNET / LAN1\nMagnetic field QDC\nSeptember 2013')

plt.show()

