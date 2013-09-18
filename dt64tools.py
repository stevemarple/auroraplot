# locator class for numpy.Datetime64
'''Tools for numpy.datetime64 and numpy.timedelta64 time classes'''

import datetime
import re
import math
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import Locator
from matplotlib.ticker import Formatter

import copy

epoch64_us = np.datetime64('1970-01-01T00:00:00Z','us')

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
log10_to_unit = {-18: 'as',
                  -15: 'fs',
                  -12: 'ps',
                  -9: 'ns',
                  -6: 'us',
                  -3: 'ms',
                  0: 's'}

def get_units(t):
    #assert isinstance(t, (np.datetime64, np.timedelta64)), \
    #    'Must be of type numpy.datetime64 or numpy.timedelta64'
    match = re.search('\[(.*)\]', t.dtype.str)
    assert match is not None, 'Failed to parse units in numpy.datetime64 type'
    return match.groups()[0]

def from_YMD(year, month, day):
    ya = np.array(year)
    ma = np.array(month)
    da = np.array(day)
    
    r_shape = None
    for a in [ya, ma, da]:
        if a.size != 1:
            assert r_shape is None or r_shape == a.shape, \
                'Shapes must match or be scalar'
            r_shape = a.shape
    if r_shape is None:
        r_shape = [1]

    yaf = ya.flatten()
    maf = ma.flatten()
    daf = da.flatten()

    r = np.zeros(np.prod(r_shape), dtype='M8[D]')
    for n in range(r.size):
        if yaf.size > 1:
            y = yaf[n]
        else:
            y = yaf[0]
        if maf.size > 1:
            m = maf[n]
        else:
            m = maf[0]
        if daf.size > 1:
            d = daf[n]
        else:
            d = daf[0]
        r[n] = np.datetime64(datetime.date(y, m, d))
    return r.reshape(r_shape)



def dt64_to(t, to_unit, returnfloat=False):
    assert to_unit in time_units, 'Unknown unit'
    assert multipliers.has_key(to_unit), 'Only fixed value units supported'

    from_unit = get_units(t)
    assert from_unit in time_units, 'Unknown unit'
    assert multipliers.has_key(from_unit), 'Only fixed value units supported'
    
    from_mul = multipliers[from_unit]
    to_mul = multipliers[to_unit]
    if from_mul >= to_mul:
        if returnfloat:
            return t.astype('int64') * np.round(from_mul / to_mul)
        else:
            # No loss of precision, possible overflow
            return t.astype('int64') * int(np.round(from_mul / to_mul))
    else:
        if returnfloat:
            return t.astype('int64') / np.round(to_mul / from_mul)
        else:
            # Possible loss of precision, no overflow
            return t.astype('int64') / int(np.round(to_mul / from_mul))
    
def isnat(x):
    # Do not trust that NaT will compare equal with NaT!
    return np.array(x).astype('int') == -(2**63)
    
def get_time_of_day(t):
    td64_units = t.dtype.str.lower() # timedelta64 equivalent units
    d = np.timedelta64(1, 'D').astype(td64_units).astype('int64')
    return np.mod(t.astype(td64_units).astype('int64'), d).astype(td64_units)

def _get_tt(a, attr):
    aa = np.array(a) # Handle all array-like possibilities
    aaf = aa.flatten() 
    r = np.zeros_like(aaf, dtype=int)
    for n in range(aaf.size):
        r[n] = getattr(aaf[n].tolist().timetuple(), attr)
    if aa.shape == ():
        return int(r[0])
    else:
        return r.reshape(aa.shape).astype('int')

def get_year(a):
    '''Return year'''
    return _get_tt(a, 'tm_year')

def get_month(a):
    '''Return month'''
    return _get_tt(a, 'tm_mon')

def get_day_of_year(a):
    '''Return day of year in range 1 to 366'''
    return _get_tt(a, 'tm_mday')

def get_day_of_month(a):
    '''Return day of month'''
    return _get_tt(a, 'tm_mday')
# Alias
get_day = get_day_of_month

def get_week_day(a):
    '''Return day of week, in range 0 (Monday) to 6 (Sunday)'''
    return _get_tt(a, 'tm_wday')

def get_hour(a):
    '''Return hour of day'''
    return _get_tt(a, 'tm_hour')

def get_minute(a):
    '''Return hour of day'''
    return _get_tt(a, 'tm_min')

def get_second(a):
    '''Return hour of day'''
    return _get_tt(a, 'tm_sec')


def get_start_of_month(a):
    a = np.array(a)
    af = a.flatten() # Handle all array-like possibilities
    for n in range(af.size):
        d = af[n].tolist()
        som = datetime.date(d.year, d.month, 1)
        af[n] = np.datetime64(som).astype(a.dtype)
    if a.shape == ():
        return af[0]
    else:
        return af.reshape(a.shape)
        
def get_start_of_next_month(a):
    return get_start_of_month(get_start_of_month(a) + np.timedelta64(32, 'D'))

def mean(*a):
    if len(a) == 0:
        raise Exception('Require at least one argument')

    a0 = np.array(a[0])
    d = a0.dtype
    if len(a) == 1:
        if a0.size:
            return np.mean(a0.astype('int64')).astype(d)
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
    return (int(np.round((dt - epoch64_us) / td)) * td) + epoch64_us

def floor(dt, td):
    return (int(np.floor((dt - epoch64_us) / td)) * td) + epoch64_us

def ceil(dt, td):
    return (int(np.ceil((dt - epoch64_us) / td)) * td) + epoch64_us



def fmt_dt64_range(st, et):
    day = np.timedelta64(1, 'D')
    if st == floor(st, day) and et == floor(st, day):
        # Start and end on same date. TODO. determine resolution needed
        return strftime(st, '%Y-%m-%d %H:%M:%S - ') +  \
            strftime(et, '%H:%M:%S')
    else:
        return strftime(st, '%Y-%m-%d %H:%M:%S - ') + \
            strftime(et, '%Y-%m-%d %H:%M:%S')

def plot_dt64(x, y, axes=None, 
              # Our own options
              x_time_units=None, y_time_units=None, plot_func=plt.plot,
              #
              **kwargs):
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
            axes.tick_params(direction='out')

        # Transform y to suitable units
        yy = dt64_to(y, axis_data.units)
    else:
        yy = y
        
    if plot_func is None:
        r = plt.plot(xx, yy, **kwargs)
    else:
        r = plot_func(xx, yy, **kwargs)

        
    # Set up the x and y axis details after having plotted when the
    # axis limits etc are more appropriate

    # if not hasattr(axes.yaxis, 'dt64tools'):
    #     axes.yaxis.dt64tools = axis_data
    #     # Set some default formatting
    #         ### TODO: Correct locator and formatter for timedelta64
    #     axes.yaxis.set_major_locator(Datetime64Locator())
    #     axes.yaxis.set_major_formatter(Datetime64Formatter())
    #     plt.yticks(rotation=-25)
    #     axes.tick_params(direction='out')


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
        if not hasattr(self.axis, 'dt64tools'):
            raise Exception('dt64tools data not found on axis')
        units = self.axis.dt64tools.units
        limits = self.axis.get_view_interval()

        if np.diff(limits) < 1:
            print('limit for tick labels reached')
            return []

        limits_dt64 = (limits.astype('m8[' + units + ']') + 
                       self.axis.dt64tools.type(0, units))
        tick_interval = self.calc_interval(limits_dt64[0], limits_dt64[1])
        if tick_interval is not None:
            self.interval = tick_interval
            first_tick = ceil(limits_dt64[0], self.interval)
            last_tick = floor(limits_dt64[1], self.interval)
            num = np.floor(((last_tick - first_tick) / self.interval) + 1)
            tick_times = np.linspace(first_tick, last_tick, num)
            tick_locs = ((tick_times - np.datetime64(0, units)) /
                         np.timedelta64(1, units))

            # return self.raise_if_exceeds(tick_locs)
            try:
                return self.raise_if_exceeds(tick_locs)
            except RuntimeError as e:
                print('Runtime error: ' + e.message)
                return []

        else:
            return self.raise_if_exceeds(\
                self.calc_irregular_ticks(limits_dt64[0], limits_dt64[1],
                                          units))

    
    def calc_interval(self, st, et):
        '''
        Compute the tick location interval. Only useful for durations
        where regular ticks can be used. See also
        calc_irregular_ticks().
        '''

        # Compute the interval required, as a FP number of seconds,
        # assuming the maximum number of ticks.
        approx_ival_s = dt64_to(et - st, 's', returnfloat=True) \
            / (self.maxticks - 1)

        if approx_ival_s < 1.0:
            approx_ival_log10 = math.log10(approx_ival_s)
            
            # Find the best unit to work in
            unit_log10 = int(np.floor(approx_ival_log10 / 3) * 3)
            best_unit = log10_to_unit.get(unit_log10)

            # Compute the range of intervals to check, eg for
            # approx_ival_s = 19e-9 generate [10, 20, 40, 50] ns. For
            # 190e-3 generate [100, 200, 400, 500] ms. They all use
            # the same unit so there is not problem with loss of
            # precision or dynamic range.
            intervals = (np.array([1, 2, 4, 5]) * 
                         10**int(np.floor(approx_ival_log10 - unit_log10))).astype('m8[' + best_unit + ']')
            approx_ival = np.timedelta64(int(approx_ival_s * 10**-unit_log10), 
                                         best_unit)
            idx = (approx_ival >= intervals).tolist().index(False)
            tick_interval = intervals[idx]
            return intervals[idx]

        elif approx_ival_s < 7 * 86400: # < 3 weeks
            approx_ival = np.timedelta64(int(approx_ival_s), 's')
            
            intervals = [np.timedelta64(1, 's'),
                         np.timedelta64(2, 's'),
                         np.timedelta64(4, 's'),
                         np.timedelta64(5, 's'),
                         np.timedelta64(10, 's'), 
                         np.timedelta64(15, 's'),
                         np.timedelta64(20, 's'),
                         np.timedelta64(30, 's'),
                         np.timedelta64(1, 'm'),
                         np.timedelta64(2, 'm'),
                         np.timedelta64(4, 'm'),
                         np.timedelta64(5, 'm'),
                         np.timedelta64(10, 'm'),
                         np.timedelta64(15, 'm'),
                         np.timedelta64(20, 'm'),
                         np.timedelta64(30, 'm'),
                         np.timedelta64(1, 'h'),
                         np.timedelta64(2, 'h'),
                         np.timedelta64(3, 'h'),
                         np.timedelta64(4, 'h'),
                         np.timedelta64(6, 'h'),
                         np.timedelta64(8, 'h'),
                         np.timedelta64(12, 'h'),
                         np.timedelta64(1, 'D'),
                         np.timedelta64(2, 'D'),
                         np.timedelta64(5, 'D'),
                         np.timedelta64(7, 'D'),
                         np.timedelta64(14, 'D'),
                         np.timedelta64(21, 'D')]
            for ival in intervals:
                if ival >= approx_ival:
                    return ival
            return intervals[-1]

        else:
            return None


    def calc_irregular_ticks(self, st, et, units):
        # Compute the interval required, as a FP number of seconds,
        # assuming the maximum number of ticks.
        approx_ival_s = dt64_to(et - st, 's', returnfloat=True) \
            / (self.maxticks - 1)
             
        if approx_ival_s < 365 * 86400: # Try for up to about 6 months
            # Try using months
            som = [get_start_of_month(st)]
            while som[-1] <= et:
                som.append(get_start_of_next_month(som[-1]))

            if som[0] < st:
                som.pop(0) # Too early, remove
            if som[-1] > et:
                som.pop() # Too late, remove

            month0 = get_month(som) -1

            for n in [1, 2, 3, 4, 6]:
                # Find every nth month, always include January
                idx = np.mod(month0, n) == 0
                if np.where(idx)[0].size <= self.maxticks:
                    som = np.array(som)
                    return dt64_to(som[idx], units).tolist()

        # Use years. 
        approx_ival_y = approx_ival_s / (365.25 * 86400)
        # Begin the search for optimum interval earlier then may
        # be expected because the first tick is not always at the
        # start of the display range. Also dependent on how start
        # time occurs relative to leap years.

        ival_log10 = int(math.log10(approx_ival_y) - 1)
        if ival_log10 < 0:
            ival_log10 = 0
        years = np.array([1, 2, 4, 5], dtype=int) * (10 ** ival_log10)

        # Years are not regularly spaced so check the preliminary
        # result does not exceed maximum number of ticks. If so
        # continue with larger spacing. Should not need more than
        # 3 loops.
        for n in range(3):
            for y in years:
                # Round start year down to multiple of y years
                st2_y = int(np.floor(get_year(st) / y) * y)

                tick_years = range(st2_y, get_year(et) + 1, y)
                if from_YMD(tick_years[0], 1, 1) < st:
                    tick_years.pop(0)
                if from_YMD(tick_years[-1], 1, 1) > et:
                    tick_years.pop()
                if len(tick_years) <= self.maxticks:
                    # Success
                    return dt64_to(from_YMD(tick_years, 1, 1), units).tolist()
            years *= 10

        raise Exception('Failed to compute correct yearly tick spacing')


class Datetime64Formatter(Formatter):
    def __init__(self, fmt=None):
        self.fmt = fmt

    
    def __call__(self, x, pos=None):
        # This is also called in ipython to show cursor location,
        # where pos=None
        
        if not hasattr(self.axis, 'dt64tools'):
            raise Exception('Cannot find axis time units')

        units = self.axis.dt64tools.units

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
            else:
                tick_interval = \
                     np.timedelta64(int(np.diff(self.axis.get_view_interval())),
                                    units)
            f = []
            # If data from more than one day always include date, even
            # if current zoomed view is part of a day
            if data_interval > np.timedelta64(1, 'D'):
                # f.append('%Y-%m-%d')
                if tick_interval is not None and \
                        tick_interval>= np.timedelta64(365, 'D'):
                    f.append('%Y')
                else:
                    f.append('%Y-%m-%d')
            
            # Choose seconds and fractional seconds depending on current view
            if tick_interval < np.timedelta64(1, 's'):
                f.append('%H:%M:%S.%#')
            elif tick_interval < np.timedelta64(1, 'm'):
                f.append('%H:%M:%S')
            elif tick_interval< np.timedelta64(1, 'D'):
                f.append('%H:%M')
            fmt = '\n'.join(f)

        t = np.datetime64(int(x), units)
        return strftime(t, fmt) 
    

def strftime(t, fstr):
    tf = t.flatten() # Handle all array-like possibilities
    r = np.zeros_like(tf, dtype=object)
    for n in range(tf.size):
        r[n] = _strftime(tf[n], fstr)
    if t.shape == ():
        return r[0]
    else:
        return r.reshape(t.shape)


def _strftime(t, fstr, customspec=None):
    '''
    Private strftime function. Must be called with numpy.datetime64,
    not in an array.
    '''

    if isnat(t):
        return 'NaT'
    
    day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                'Friday', 'Saturday', 'Sunday']
    d_name = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    month_name = ['January', 'February', 'March', 'April', 
                  'May', 'June', 'July', 'August', 
                  'September', 'October', 'November', 'December']
    m_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    num_suffix = np.array(['th']*32) # use 1-based indexing
    num_suffix[[1,21,31]] = 'st'
    num_suffix[[2,22]] = 'nd'
    num_suffix[[3,23]] = 'rd'
    
    i = 0
    s = ''
    replacements = []
    while i < len(fstr):
        if fstr[i] == '%':
            # Look to next char
            i += 1
            replacements.append(fstr[i])
        
            if customspec is not None and customspec.has_key(fstr[i]):
                # Insert user custom specifier
                s += customspec[fstr[i]]
                
            elif fstr[i] == '%': # percent
                s += '%'
                
            elif fstr[i] == 'a': # abbreviated day name
                s += d_name[get_week_day(t)]
            elif fstr[i] == 'A': # day name
                s += day_name[get_month(t)]
            elif fstr[i] == 'b': # abbreviated month name
                s += d_name[get_week_day(t) - 1]
            elif fstr[i] == 'B': # month name
                s += day_name[get_month(t) - 1]

            elif fstr[i] == 'd': # day [dd]
                s += '{0:02d}'.format(int(get_day_of_month(t)))
            elif fstr[i] == 'H': # hour [hh]
                s += '{0:02d}'.format(int(get_hour(t)))
            elif fstr[i] == 'j': # day of year [jjj]
                s += '{0:03d}'.format(int(get_day_of_year(t)))
            elif fstr[i] == 'm': #
                s += '{0:02d}'.format(int(get_month(t)))
            elif fstr[i] == 'M': #
                s += '{0:02d}'.format(int(get_minute(t)))
            elif fstr[i] == 'q': # quarter (of year)
                s += '{0:01d}'.format(int(get_month(t))/4 + 1)
            elif fstr[i] == 's': # seconds since unix epoch
                s += '{0:d}'.format(int(dt64_to(t, 's')))
            elif fstr[i] == 'S': # seconds [ss]
                s += '{0:02d}'.format(int(get_second(t)))
            # extension
            elif fstr[i] == 't': # st, nd, rd or th day number suffix
                s += num_suffix[get_day_of_month(t)]
            elif fstr[i] == 'y': # year [yy]
                s += '{0:02d}'.format(int(get_year(t)))
            elif fstr[i] == 'Y': # year [YYYY]
                s += '{0:04d}'.format(int(get_year(t)))
            # extension
            elif fstr[i] == '#': # milliseconds
                s += '{0:03d}'.format(int(np.round(np.mod(dt64_to(t, 'ms', returnfloat=True), 1000))))
            else:
                warnings.warn('Unknown format specifier: ' + fstr[i])
                replacements.pop()
                
        else:
            s += fstr[i]
        
        i += 1

    return s
