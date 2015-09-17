#!/usr/bin/env python

# Demonstrate loading SAMNET data from the data archive at Lancaster.
# This example assumes you have an STP user account with permission to
# access SAMNET data, available on request, see
# http://spears.lancs.ac.uk/data/
#
# You can avoid entering your username/password each session by
# entering those details into a .netrc file in your home directory

import logging
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import auroraplot as ap
from auroraplot.magdata import MagData

import auroraplot.datasets.samnet


logging.basicConfig(level=getattr(logging, 'WARN'))
logger = logging.getLogger(__name__)

st = np.datetime64('2010-01-01T00:00:00+0000')
et = np.datetime64('2010-01-03T00:00:00+0000')

mag_data = ap.load_data('SAMNET', 'LAN1', 
                        'MagData', st, et, archive='5s')
print(mag_data)


st2 = np.datetime64('2013-03-17T00:00:00+0000')
et2 = np.datetime64('2013-03-18T00:00:00+0000')

# Load data from all SAMNET sites
mdl = []
for site in ap.projects['SAMNET']:
    try:
        tmp = ap.load_data('SAMNET', site, 
                           'MagData', st2, et2, archive='5s')
    except:
        tmp = None

    if tmp is not None and not np.all(np.isnan(tmp.data)) \
            and not np.all(tmp.data == 0):
        mdl.append(tmp)

# Stack plot of all data for H component. Separate traces by 400nT.
ap.magdata.stack_plot(mdl, 400e-9, channel='H')
plt.grid(True)
plt.show()

