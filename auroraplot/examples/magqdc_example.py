#!/usr/bin/env python

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import auroraplot as ap
import auroraplot.dt64tools as dt64
import auroraplot.magdata
import auroraplot.datasets.aurorawatchnet

plt.close('all')
t = np.datetime64('2013-09-01T00:00:00+0000')
qdc = ap.magdata.load_qdc('AWN', 'LAN1', t)

qdc.plot()
plt.title('AWN / LAN1\nMagnetic field QDC\nSeptember 2013')

plt.show()

