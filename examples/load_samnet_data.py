#!/usr/bin/env python
import numpy as np
import auroraplot as ap
from auroraplot.magdata import MagData

import auroraplot.datasets.samnet

st = np.datetime64('2010-01-01T00:00:00+0000')
et = np.datetime64('2010-01-03T00:00:00+0000')

mag_data = ap.load_data('SAMNET', 'LAN', 
                        'MagData', st, et, archive='5s')
print(mag_data)
