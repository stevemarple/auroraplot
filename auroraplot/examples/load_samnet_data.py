#!/usr/bin/env python
import numpy as np
import auroraplot as ap
from auroraplot.magdata import MagData

import auroraplot.datasets.samnet

st = np.datetime64('2010-01-01T00:00:00+0000')
et = np.datetime64('2010-01-03T00:00:00+0000')

mag_data = ap.load_data('SAMNET', 'LAN1', 
                        'MagData', st, et, archive='5s')
print(mag_data)


st2 = np.datetime64('2013-03-17T00:00:00+0000')
et2 = np.datetime64('2013-03-18T00:00:00+0000')

# Load data from all SAMNET sites
mdl = []
for site in ap.networks['SAMNET']:
    tmp = ap.load_data('SAMNET', site, 
                       'MagData', st2, et2, archive='5s')
    if tmp is not None and not np.all(np.isnan(tmp.data)) \
            and not np.all(tmp.data == 0):
        mdl.append(tmp)



