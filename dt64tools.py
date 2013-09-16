# locator class for numpy.Datetime64
'''Tools for numpy.datetime64 and numpy.timedelta64 time classes'''

import datetime
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import Locator
from matplotlib.ticker import Formatter

import copy

epoch64_ns = np.datetime64('1970-01-01T00:00:00Z','ns')

time_units = ['as', 'fs', 'ps', 'ns', 'us', 'ms', 's', 'm', 'h',
              'D', 'W', 'M', 'Y']

multipliers = {'as': 1e-18, 
               'fs': 1e-15,
               'ps': 1e-12,
               'ns': 1e-9,
               'us': 1e-6,
               'ms': 1e-3,
               's': 1.0,
               'm': 60.0,
               'h': 3600.0,
               'D': 86400.0,
               'W': 7 * 86400.0}

def get_units(t):
    #assert isinstance(t, (np.datetime64, np.timedelta64)), \
    #    'Must be of type numpy.datetime64 or numpy.timedelta64'
    match = re.search('\[(.*)\]', t.dtype.str)
    assert match is not None, 'Failed to parse units in numpy.datetime64 type'
    return match.groups()[0]


def dt64_to(t, to_unit):
    assert to_unit in time_units, 'Unknown unit'
    assert multipliers.has_key(to_unit), 'Only fixed value units supported'

    from_unit = get_units(t)
    assert from_unit in time_units, 'Unknown unit'
    assert multipliers.has_key(from_unit), 'Only fixed value units supported'
    # return (t - epoch64_ns) / np.timedelta64(1, unit)
    
    from_mul = multipliers[from_unit]
    to_mul = multipliers[to_unit]
    if from_mul >= to_mul:
        # No loss of precision, possible overflow
        return t.astype('int64') * int(np.round(from_mul / to_mul))
    else:
        # Possible loss of precision, no overflow
        return t.astype('int64') / int(np.round(to_mul / from_mul))
    
    # return t.astype('m8[' + unit + ']').astype('int64')

def isnat(x):
    # Do not trust that NaT will compare equal with NaT!
    return np.array(x).astype('int') == -(2**63)
    
def get_time_of_day(t):
    td64_units = t.dtype.str.lower() # timedelta64 equivalent units
    d = np.timedelta64(1, 'D').astype(td64_units).astype('int64')
    return np.mod(t.astype(td64_units).astype('int64'), d).astype(td64_units)

def mean(*a):
    if len(a) == 0:
        raise Exception('Require at least one argument')

    d = copy.copy(a[0].dtype)
    if len(a) == 1:
        if len(a[0]):
            return np.mean(a[0].astype('int64')).astype(d)
        else:
            return np.datetime64('nat').astype(d)
    else:
        tmp = a[0].astype('int64')
        for b in a[1:]:
            # Check that the elements of 'b are the same type as
            # elements of a[0]. They do not have to use the same units
            # (days, seconds, ns etc) but a temporary copy will be
            # cast to that unit for the calculation.
            assert isinstance(b.dtype.type(), d.type), \
                'Arrays must hold the same data type'
            tmp += b.astype(d).astype('int')
        return (tmp / len(a)).astype(d)

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

def plot_dt64(x, y, axes=None, x_time_units=None, y_time_units=None, **kwargs):
    if axes is None:
        axes = plt.gca()
    
    xdate = False
    try:
        if x.dtype.type in (np.datetime64, np.timedelta64):
            xdate = True
    except:
        pass

    ydate = False
    try:
        if y.dtype.type in (np.datetime64, np.timedelta64):
            ydate = True
    except:
        pass

    if xdate:
        units = get_units(x)
        assert units in time_units, 'Unknown units'
        assert units not in ['M', 'Y'], 'Time units cannot be month or year'
        if hasattr(axes.xaxis, 'dt64tools'):
            # Axis already has our data
            axis_data = axes.xaxis.dt64tools
            assert x.dtype.type == axis_data.type, 'Cannot add ' \
                + x.dtype.type + ' to axis using ' + axis_data.type
            assert time_units.index(units) \
                >= time_units.index(axis_data.units), \
                'Cannot add time data with units ' + units + \
                ' to existing plot using units ' + axis_data.units
        else:
            if x_time_units is None:
                x_time_units = units
            axis_data = Dt64ToolsData(units=x_time_units, 
                                      type=x.dtype.type)
            axes.xaxis.dt64tools = axis_data

            # Set some default formatting
            ### TODO: Correct locator and formatter for timedelta64
            axes.xaxis.set_major_locator(Datetime64Locator())
            axes.xaxis.set_major_formatter(Datetime64Formatter())
            plt.xticks(rotation=-25)

        # Transform x to suitable units
        xx = dt64_to(x, axis_data.units)
    else:
        xx = x

    if ydate:
        units = get_units(y)
        assert units in time_units, 'Unknown units'
        assert units not in ['M', 'Y'], 'Time units cannot be month or year'
        if hasattr(axes.yaxis, 'dt64tools'):
            # Axis already has our data
            axis_data = axes.yaxis.dt64tools
            assert y.dtype.type == axis_data.type, 'Cannot add ' \
                + y.dtype.type + ' to axis using ' + axis_data.type
            assert time_units.indey(units) \
                >= time_units.indey(axis_data.units), \
                'Cannot add time data with units ' + units + \
                ' to existing plot using units ' + axis_data.units
        else:
            if y_time_units is None:
                y_time_units = units
            axis_data = Dt64ToolsData(units=y_time_units, 
                                      type=y.dtype.type)
            axes.yaxis.dt64tools = axis_data

            # Set some default formatting
            ### TODO: Correct locator and formatter for timedelta64
            axes.yaxis.set_major_locator(Datetime64Locator())
            axes.yaxis.set_major_formatter(Datetime64Formatter())
            plt.yticks(rotation=-25)

        # Transform y to suitable units
        yy = dt64_to(y, axis_data.units)
    else:
        yy = y
        
    r = plt.plot(xx, yy, **kwargs)
    axes.tick_params(direction='out')
    return r


class Dt64ToolsData(object):
    def __init__(self, units, type):
        self.units = units
        self.type = type

class Datetime64Locator(Locator):
    '''Locator class for numpy.datetime64'''
    def __init__(self, maxticks=8):
        self.maxticks = maxticks
        self.interval = None
        

    def __call__(self):
        '''Return the locations of the ticks'''
        if hasattr(self.axis, 'dt64tools'):
            units = self.axis.dt64tools.units
        else:
            units = 'ns'
        limits = self.axis.get_view_interval()
        limits_dt64 = (limits.astype('m8[' + units + ']') + 
                       np.datetime64(0, units))
        self.interval = self.calc_interval(limits_dt64)
        first_tick = ceil(limits_dt64[0], self.interval)
        last_tick = floor(limits_dt64[1], self.interval)
        num = np.floor(((last_tick - first_tick) / self.interval) + 1)
        tick_times = np.linspace(first_tick, last_tick, num)
        tick_locs = ((tick_times - np.datetime64(0, units)) /
                     np.timedelta64(1, units))
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
    
