#!/usr/bin/env python
import numpy as np
import matplotlib as mpl
import auroraplot.dt64tools as dt64

from matplotlib import pyplot as plt
plt.close('all')


# Define a line from year 1200 to year 2080
x = np.array(dt64.from_YMD([1200,2080],1,1))
y = np.array([0,1])

x = x.astype('M8[ms]')

# Plot
dt64.plot_dt64(x,y)
plt.show()

print(dt64.strftime(x[0], 'Start time: %Y-%m-%d %H:%M:%S'))
print(dt64.strftime(x[-1], 'End time: %Y-%m-%d %H:%M:%S'))
print('Now try zoom and watch the time units adjust automatically')

