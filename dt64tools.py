# locator class for numpy.Datetime64
'''Tools for numpy.datetime64 and numpy.timedelta64 time classes'''

import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import Locator
from matplotlib.ticker import Formatter

epoch64_ns = np.datetime64('1970-01-01T00:00:00Z','ns')
def dt64_to(t, unit):
    return (t - epoch64_ns) / np.timedelta64(1, unit)

def mean(*a):
    if len(a) == 1:
        return np.mean(a[0] - epoch64_ns) + epoch64_ns
    else:
        tmp = a[0] - epoch64_ns
        for b in a[1:]:
            tmp += b - epoch64_ns
        return (tmp / len(a)) + epoch64_ns

def round(dt, td):
    return (int(np.round((dt - epoch64_ns) / td)) * td) + epoch64_ns

def floor(dt, td):
    return (int(np.floor((dt - epoch64_ns) / td)) * td) + epoch64_ns

def ceil(dt, td):
    return (int(np.ceil((dt - epoch64_ns) / td)) * td) + epoch64_ns

def strftime64(dt64, fstr):
    '''Convert numpy.datetime64 object to string using strftime format string'''
    dt = (dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1,'s')
    return datetime.datetime.utcfromtimestamp(dt).strftime(fstr)

def fmt_dt64_range(st, et):
    day = np.timedelta64(1, 'D')
    if st == floor(st, day) and et == floor(st, day):
        # Start and end on same date. TODO. determine resolution needed
        return strftime64(st, '%Y-%m-%d %H:%M:%S - ') +  \
            strftime64(et, '%H:%M:%S')
    else:
        return strftime64(st, '%Y-%m-%d %H:%M:%S - ') + \
            strftime64(et, '%Y-%m-%d %H:%M:%S')

def plot_dt64(x, y, fmt='bo', dtfmt=None, hold=None, **kwargs):
    ax = plt.gca()
    
    xdate = False
    try:
        xdate = isinstance(x, np.datetime64)
    except:
        pass
    try:
        xdate = isinstance(x[0], np.datetime64)
    except:
        pass
    ydate = False
    try:
        ydate = isinstance(y, np.datetime64)
    except:
        pass
    try:
        ydate = isinstance(y[0], np.datetime64)
    except:
        pass

    if xdate:
        xx = (x - epoch64_ns) / np.timedelta64(1, 'ns')
        ax.xaxis.set_major_locator(Datetime64Locator())
        ax.xaxis.set_major_formatter(Datetime64Formatter())
        plt.xticks(rotation=-25)
    else:
        xx = x
    if ydate:
        yy = (y - epoch64_ns) / np.timedelta64(1, 'ns')
        ax.yaxis.set_major_locator(Datetime64Locator())
        ax.yaxis.set_major_formatter(Datetime64Formatter())
    else:
        yy = y
        
    r = plt.plot(xx, yy)
    ax.tick_params(direction='out')
    return r

class Datetime64Locator(Locator):
    '''Locator class for numpy.datetime64'''
    def __init__(self, units='ns', maxticks=8):
        # Locator.__init__(self)
        self.units = units
        self.maxticks = maxticks
        self.interval = None
        

    def __call__(self):
        '''Return the locations of the ticks'''
        limits = self.axis.get_view_interval()
        limits_dt64 = (limits.astype('m8[' + self.units + ']') + 
                       np.datetime64(0, self.units))
        self.interval = self.calc_interval(limits_dt64)
        first_tick = ceil(limits_dt64[0], self.interval)
        last_tick = floor(limits_dt64[1], self.interval)
        num = np.floor(((last_tick - first_tick) / self.interval) + 1)
        tick_times = np.linspace(first_tick, last_tick, num)
        tick_locs = ((tick_times - np.datetime64(0, self.units)) /
                     np.timedelta64(1, self.units))
        return self.raise_if_exceeds(tick_locs)

        
    def calc_interval(self, st_et):
        plot_interval = np.diff(st_et)[0]
        approx_interval = plot_interval / (self.maxticks - 1)
        intervals = np.array([np.timedelta64(1, 'ns'),
                              np.timedelta64(2, 'ns'),
                              np.timedelta64(5, 'ns'),
                              np.timedelta64(10, 'ns'),
                              np.timedelta64(20, 'ns'),
                              np.timedelta64(50, 'ns'),
                              np.timedelta64(100, 'ns'),
                              np.timedelta64(200, 'ns'),
                              np.timedelta64(500, 'ns'),
                              np.timedelta64(1, 'ms'),
                              np.timedelta64(2, 'ms'),
                              np.timedelta64(5, 'ms'),
                              np.timedelta64(10, 'ms'),
                              np.timedelta64(20, 'ms'),
                              np.timedelta64(50, 'ms'),
                              np.timedelta64(100, 'ms'),
                              np.timedelta64(200, 'ms'),
                              np.timedelta64(500, 'ms'),
                              np.timedelta64(1, 's'),
                              np.timedelta64(2, 's'),
                              np.timedelta64(5, 's'),
                              np.timedelta64(10, 's'),
                              np.timedelta64(15, 's'),
                              np.timedelta64(20, 's'),
                              np.timedelta64(30, 's'),
                              np.timedelta64(1, 'm'),
                              np.timedelta64(2, 'm'),
                              np.timedelta64(5, 'm'),
                              np.timedelta64(10, 'm'),
                              np.timedelta64(15, 'm'),
                              np.timedelta64(20, 'm'),
                              np.timedelta64(30, 'm'),
                              np.timedelta64(1, 'h'),
                              np.timedelta64(2, 'h'),
                              np.timedelta64(4, 'h'),
                              np.timedelta64(6, 'h'),
                              np.timedelta64(8, 'h'),
                              np.timedelta64(12, 'h'),
                              np.timedelta64(1, 'D'),
                              np.timedelta64(2, 'D'),
                              np.timedelta64(5, 'D'),
                              np.timedelta64(10, 'D')])

        try:        
            idx = (approx_interval > intervals).tolist().index(False)
            return intervals[idx]
        except ValueError:
            return intervals[-1]
                
class Datetime64Formatter(Formatter):
    def __init__(self, fmt=None):
        self.fmt = fmt

    
    def __call__(self, x, pos=None):
        # This is also called in ipython to show cursor location,
        # where pos=None
        units = 'ns' # Assume ns for now
        fmt = self.fmt
        if pos is None:
            fmt = '%Y-%m-%dT%H:%M:%SZ'
        elif fmt is None:
            data_interval = \
                np.timedelta64(int(np.diff(self.axis.get_data_interval())), 
                               units)
            tick_locs = self.axis.get_ticklocs()
            if len(tick_locs) >= 2:
                tick_interval = \
                    np.timedelta64(int((tick_locs[-1] - tick_locs[0]) / 
                                       (len(tick_locs) - 1)), units)
            f = []
            # If data from more than one day always include date, even
            # if current zoomed view is part of a day
            if data_interval > np.timedelta64(1, 'D'):
                f.append('%Y-%m-%d')
            
            # Choose seconds and fractional seconds depending on current view
            if tick_interval < np.timedelta64(1, 's'):
                f.append('%H:%M:%S.???')
                # Cannot yet print fractional seconds
                print('TODO: add fractional seconds')
            elif tick_interval < np.timedelta64(1, 'm'):
                f.append('%H:%M:%S')
            elif tick_interval< np.timedelta64(1, 'D'):
                f.append('%H:%M')
            fmt = '\n'.join(f)

        t = np.datetime64(int(x), 'ns')
        return strftime64(t, fmt) 
    
