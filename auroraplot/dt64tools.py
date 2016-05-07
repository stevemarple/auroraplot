# locator class for numpy.Datetime64
'''Tools for numpy.datetime64 and numpy.timedelta64 time classes'''

import datetime
import re
import math
import logging
import six

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import Locator
from matplotlib.ticker import Formatter

import copy

logger = logging.getLogger(__name__)


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

def get_time_type(t):
    s = t.dtype.name
    return s[:s.rindex('[')]

def get_units(t):
    #assert isinstance(t, (np.datetime64, np.timedelta64)), \
    #    'Must be of type numpy.datetime64 or numpy.timedelta64'
    match = re.search('\[(.*)\]', t.dtype.str)
    assert match is not None, \
        'Failed to parse units in numpy.datetime64 type: ' + str(t.dtype.str)
    return match.groups()[0]

def smallest_unit(a):
    i = None
    for u in a:
        if u is None:
            continue
        elif isinstance(u, six.string_types):
            pass
        else:
            u = get_units(u)

        idx = time_units.index(u)
        if i is None:
            i = idx
        else:
            i = min(i, idx)
    if i is not None:
        return time_units[i]
    else:
        return None

def astype(t, units=None, time_type=None):
    if units is None:
        units = get_units(t)
    elif not isinstance(units, six.string_types):
        units = get_units(units)
    if time_type is None:
        time_type = get_time_type(t)
    elif not isinstance(time_type, six.string_types):
        time_type = get_time_type(time_type)
    return t.astype(time_type + '[' + units + ']')


def match_units(a, inplace=False):
    units = []
    for t in a:
        units.append(get_units(t))
        
    u = smallest_unit(units)
    zero = np.timedelta64(0, u)
    if inplace:
        for t in a:
            if get_units(t) != u:
                t += zero
        return a 
    else:
        r = []
        for t in a:
            r.append(t + zero)
    return r


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
    assert to_unit in multipliers, 'Only fixed value units supported'

    from_unit = get_units(t)
    assert from_unit in time_units, 'Unknown unit'
    assert from_unit in multipliers, 'Only fixed value units supported'
    
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
    return np.array(x).astype('int64') == -(2**63)
    
def get_date(t):
    return t.astype('<M8[D]')

def get_time_of_day(t):
    td64_units = t.dtype.str.lower() # timedelta64 equivalent units
    d = np.timedelta64(24, 'h').astype(td64_units).astype('int64')
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
        return r.reshape(aa.shape).astype('int64')

def get_year(a):
    '''Return year'''
    return _get_tt(a, 'tm_year')

def get_month(a):
    '''Return month'''
    return _get_tt(a, 'tm_mon')

def get_day_of_year(a):
    '''Return day of year in range 1 to 366'''
    return _get_tt(a, 'tm_yday')

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
    '''Return minute of day'''
    return _get_tt(a, 'tm_min')

def get_second(a):
    '''Return second of day'''
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

def get_start_of_previous_month(a):
    return get_start_of_month(get_start_of_month(a) - np.timedelta64(1, 'D'))
        
def get_start_of_next_month(a):
    return get_start_of_month(get_start_of_month(a) 
                              + np.timedelta64(32, 'D'))

def mean(*a):
    return _aggregate(*a, func=np.mean)

def median(*a):
    return _aggregate(*a, func=np.median)

def _aggregate(*a, **kwargs):
    func = kwargs.get('func', np.mean)
    if len(a) == 0:
        raise Exception('Require at least one argument')
    tu = smallest_unit(a)

    a0 = np.array(a[0])
    d = get_time_type(a0) + '[' + tu + ']'
    if len(a) == 1:
        if a0.size:
            return func(a0.astype('int64')).astype(d)
        else:
            return np.datetime64('nat').astype(d)
    else:
        tmp = a[0].astype(d).astype('int64')
        for b in a[1:]:
            # Check that the elements of 'b are the same type as
            # elements of a[0]. They do not have to use the same units
            # (days, seconds, ns etc) but a temporary copy will be
            # cast to that unit for the calculation.
            assert isinstance(b.dtype.type(), a0.dtype.type), \
                'Arrays must hold the same data type'
            tmp += b.astype(d).astype('int64')
        return (tmp / len(a)).astype(d)

def _round_to_func(dt, td, func, func_name):
    td_units = get_units(td)
    if td_units in ('M', 'Y'):
        if func_name not in ('ceil', 'floor'):
            # round(), and possibly other unknown functions, must
            # evaluate the days, hours, etc to decide whether to round
            # up or down.
            raise Exception('Cannot use ' + func_name 
                            + ' with units of ' + td_units)
        # Convert the datetime to units of month or year. Discard parts
        # smaller than this, for floor() they can be ignored
        ret_type = dt.dtype.char + '8[' + td_units + ']'
        dt2 = dt.astype(ret_type)
        if func_name == 'ceil':
            # Parts smaller than month or year matter for ceil so
            # round up by one month or year where they were discarded
            if len(dt2.shape):
                dt2[dt2 - dt < 0] += np.timedelta64(1, td_units)
            elif dt2 < dt:
                dt2 += np.timedelta64(1, td_units)
        dt2f = dt2.astype('float')
        tdf = td.astype('float')
        r = (func(dt2f / tdf) * tdf).astype('int64').astype(ret_type)
        return r

    u = smallest_unit([get_units(dt), get_units(td)])
    dti = dt64_to(dt, u)
    tdi = dt64_to(td, u)
    ret_type = dt.dtype.char + '8[' + u + ']'
    return (func(float(dti) / tdi) * tdi).astype('int64').astype(ret_type)


def round(dt, td):
    '''
    Round numpy.datetime64 or numpy.timedelta64 to nearest interval.

    dt: numpy.datetime64 (or numpy.timedelta64) value to be rounded.
    
    td: numpy.timedelta64 interval to round to.

    Returns: 
    numpy.datetime64 or numpy.timedelta64 rounded to the
    nearest multiple of "td".
    '''
    return _round_to_func(dt, td, np.round, 'round')


def floor(dt, td):
    '''
    Round numpy.datetime64 or numpy.timedelta64 to down to nearest
    interval.

    dt: numpy.datetime64 (or numpy.timedelta64) value to be rounded down.
    
    td: numpy.timedelta64 interval to round down to.

    Returns: 
    numpy.datetime64 or numpy.timedelta64 rounded down the
    nearest multiple of "td".
    '''
    return _round_to_func(dt, td, np.floor, 'floor')


def ceil(dt, td):
    '''
    Round numpy.datetime64 or numpy.timedelta64 to up to nearest
    interval.

    dt: numpy.datetime64 (or numpy.timedelta64) value to be rounded up.
    
    td: numpy.timedelta64 interval to round up to.

    Returns: numpy.datetime64 or numpy.timedelta64 rounded up the
        nearest multiple of "td".
    '''
    return _round_to_func(dt, td, np.ceil, 'ceil')


def dt64_range(start, stop, step):
    '''
    Return a range of numpy.datetime64 values, similar to the Python
    2.x xrange(), or Python 3.x range(), where the values are created
    by a generator.

    start: numpy.datetime64 start time.

    stop: numpy.datetime64 stop time (exclusive).

    step: numpy.timedelta64 interval between times.

    Returns: numpy.datetime64 values.
    '''
    # Initial start time to have the same resolution as successive times
    t = start + 0*step
    while t < stop:
        yield t
        t += step

def fmt_dt64_range(st, et):
    '''
    Return a string representing an interval defined by two
    numpy.datetime64 values, such as an interval of data defined by
    start and end times.
    
    st: start time

    et: end time

    Returns: Compact string representation. Seconds are included only
        if required.Fractional seconds not currently supported.
    '''

    # If st or et have units of months or years then convert to dates,
    # has no effects if smaller date units in use
    st += np.timedelta64(0, 'D')
    et += np.timedelta64(0, 'D')
    
    day = np.timedelta64(1, 'D')
    if st == floor(st, day) and et == floor(st, day):
        # Start and end on same date. Always return at least hours and
        # minutes to get instantly recognisable values. Only includes
        # seconds if necessary.
        if get_second(st) or get_second(et):
            strftime(st, '%Y-%m-%d %H:%M:%S - ') +  \
                strftime(et, '%H:%M:%S')
        else:
            return strftime(st, '%Y-%m-%d %H:%M - ') +  \
                strftime(et, '%H:%M')
    elif st == floor(st, day) and et == st + day:
        # Entire day
        return strftime(st, '%Y-%m-%d')
    else:
        # Start and end time on different dates
        if get_second(st) or get_second(et):
            return strftime(st, '%Y-%m-%d %H:%M:%S - ') + \
                strftime(et, '%Y-%m-%d %H:%M:%S')
        else:
            return strftime(st, '%Y-%m-%d %H:%M - ') + \
                strftime(et, '%Y-%m-%d %H:%M')

def parse_datetime64(s, prec, now=None):
    '''
    Parse a numpy.datetime64() time, also accepting "yesterday",
    "today", "now" and "tomorrow", with the result given to the
    specified precision.

    s: string to parse

    prec: precision of the returned time. Must be hour ("h") or better.

    Returns: numpy.datetime64 value.
    '''

    if now is None:
        now = np.datetime64('now', 's')
    else:
        now = now.astype('datetime64[s]')
    day = np.timedelta64(1, 'D')

    s = s.strip()
    if s.startswith('yesterday'):
        t = floor(now, day) - day \
            + parse_timedelta64(' '.join(s.split()[1:]), prec)
    elif s.startswith('today'):
        t = floor(now, day) \
            + parse_timedelta64(' '.join(s.split()[1:]), prec)
    elif s.startswith('now'):
        t = now + parse_timedelta64(' '.join(s.split()[1:]), prec)
    elif s.startswith('tomorrow'):
        t = floor(now, day) + day \
            + parse_timedelta64(' '.join(s.split()[1:]), prec)
    elif s.startswith('overmorrow'):
        t = floor(now, day) + 2 * day \
            + parse_timedelta64(' '.join(s.split()[1:]), prec)
    else:
        t = np.datetime64(s)
    
    return astype(t, prec)

def parse_timedelta64(s, prec):
    r = np.timedelta64(0, prec)
    for w in s.split():
        m = re.match('^(-?[0-9]+)(as|fs|ps|ns|us|ms|s|m|h|D|W|M|Y)$', w)
        if m is None:
            raise ValueError('unknown value/unit (%s)' % w)
        v, u = m.groups()
        r += np.timedelta64(int(v), u)
    return astype(r, units=prec)


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
            # if axis_data.type == np.datetime64:
            axes.xaxis.set_major_formatter(Datetime64Formatter())

            # plt.xticks(rotation=-25)

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
            # if axis_data.type == np.datetime64:
            axes.yaxis.set_major_formatter(Datetime64Formatter())

            plt.yticks(rotation=-25)  

        # Transform y to suitable units
        yy = dt64_to(y, axis_data.units)
    else:
        yy = y
        
    if plot_func is None:
        r = plt.plot(xx, yy, **kwargs)
    else:
        r = plot_func(xx, yy, **kwargs)
        
    axes.tick_params(direction='out')
        
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


def xlim_dt64(xmin=None, xmax=None, ax=None):
    if ax is None:
        ax = plt.gca()
    if xmin is not None:
        xmin = dt64_to(xmin, ax.xaxis.dt64tools.units)
    if xmax is not None:
        xmax = dt64_to(xmax, ax.xaxis.dt64tools.units)
    return ax.set_xlim(xmin=xmin, xmax=xmax)


def ylim_dt64(ymin=None, ymax=None, ax=None):
    if ax is None:
        ax = plt.gca()
    if ymin is not None:
        ymin = dt64_to(ymin, ax.yaxis.dt64tools.units)
    if ymax is not None:
        ymax = dt64_to(ymax, ax.yaxis.dt64tools.units)
    return ax.set_ylim(ymin=ymin, ymax=ymax)


def get_plot_units(axis):
    return axis.dt64tools.units


class Dt64ToolsData(object):
    def __init__(self, units, type):
        self.units = units
        self.type = type

class Datetime64Locator(Locator):
    '''Locator class for numpy.datetime64'''
    def __init__(self, maxticks=8, interval=None):
        self.maxticks = maxticks
        self.interval = interval


    def __call__(self):
        '''Return the locations of the ticks'''
        if not hasattr(self.axis, 'dt64tools'):
            raise Exception('dt64tools data not found on axis')
        units = self.axis.dt64tools.units
        limits = self.axis.get_view_interval()

        if np.diff(limits) < 1:
            logger.warn('limit for tick labels reached')
            return []

        limits_dt64 = (limits.astype('m8[' + units + ']') + 
                       self.axis.dt64tools.type(0, units))

        if self.interval is not None:
            tick_interval = self.interval
        else:
            tick_interval = self.calc_interval(limits_dt64[0], limits_dt64[1])
        if tick_interval is not None:
            first_tick = ceil(limits_dt64[0], tick_interval)
            last_tick = floor(limits_dt64[1], tick_interval)
            num = np.floor(((last_tick - first_tick) / 
                            tick_interval) + 1).astype(int)
            tick_locs = ((dt64_to(tick_interval, units).astype(int) 
                          * np.arange(num))
                         + first_tick.astype(int))
            try:
                return self.raise_if_exceeds(tick_locs)
            except RuntimeError as e:
                logger.exception(e)
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
            try:
                idx = (approx_ival >= intervals).tolist().index(False)
                tick_interval = intervals[idx]
                return intervals[idx]
            except ValueError as e:
                # Not found, fall through and use 1s or larger interval
                pass
            except:
                raise
            
        if approx_ival_s < 7 * 86400: # < 3 weeks
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

                tick_years = list(range(st2_y, get_year(et) + 1, y))
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
    def __init__(self, fmt=None, autolabel=None):
        '''
        Format ticks with date/time as appropriate. 

        fmt: Manually set the strftime format string.

        autolabel: Label the axis. 
        If False no labeling will occur. 
        If True a axis label will be updated when the axis ticks are
        formatted.
        If None the axis label will be updated only if it is an empty
        string. This is the default.
        '''

        self.fmt = fmt
        self.autolabel = autolabel

    
    def __call__(self, x, pos=None):
        # This is also called to show cursor location, where pos=None
        
        if not hasattr(self.axis, 'dt64tools'):
            raise Exception('Cannot find axis time units')

        units = self.axis.dt64tools.units

        fmt = self.fmt
        if pos is None:
            # Showing cursor location
            fmt = '%Y-%m-%dT%H:%M:%SZ'
        elif fmt is None:
            xadi = self.axis.get_data_interval()
            if np.all(np.isfinite(xadi)):
                data_interval = np.timedelta64(int(np.diff(xadi)), units)
            else:
                # Nothing plotted, so use current limits for time range
                data_interval = np.timedelta64(int(np.diff(\
                            self.axis.get_view_interval())), units)
                
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
            elif tick_interval< np.timedelta64(1, 'h'):
                f.append('%H:%M')
            elif tick_interval< np.timedelta64(1, 'D'):
                f.append('%H')
            fmt = '\n'.join(f)

        t = np.datetime64(int(x), units)
        r = strftime(t, fmt) 
        if pos == 1:
            self.axis.dt64tools.tick_fmt = fmt
            if self.autolabel or (self.autolabel is None and
                                  self.axis.get_label().get_text() == ''):
                if r == strftime(t + np.timedelta64(1, 'D'), fmt):
                    # No date information present in string
                    s = 'Time' 
                else:
                    s = 'Date'

                if isinstance(self.autolabel, str):
                    s = self.autolabel % s

                self.axis.get_label().set_text(s)

        return r
    

def strftime(t, fstr):
    tf = t.flatten() # Handle all array-like possibilities
    r = np.zeros_like(tf, dtype=object)
    for n in range(tf.size):
        # If t.dtype is an object the exact type can change. Test at
        # this point.
        if isinstance(tf[n], np.datetime64):
            r[n] = _strftime_dt64(tf[n], fstr)
        elif isinstance(tf[n], np.timedelta64):
            r[n] = _strftime_td64(tf[n], fstr)
        else:
            raise TypeError('Expecting numpy.datetime64 or numpy.timedelta64')
    if t.shape == ():
        return r[0]
    else:
        return r.reshape(t.shape)


def _strftime_dt64(t, fstr, customspec=None):
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
        
            if customspec is not None and fstr[i] in customspec:
                # Insert user custom specifier
                s += customspec[fstr[i]]
                
            elif fstr[i] == '%': # percent
                s += '%'
                
            elif fstr[i] == 'a': # abbreviated day name
                s += d_name[get_week_day(t)]
            elif fstr[i] == 'A': # day name
                s += day_name[get_week_day(t)]
            elif fstr[i] == 'b': # abbreviated month name
                s += m_name[get_month(t) - 1]
            elif fstr[i] == 'B': # month name
                s += month_name[get_month(t) - 1]

            elif fstr[i] == 'd': # day [dd]
                s += '{0:02d}'.format(int(get_day_of_month(t)))
            elif fstr[i] == 'H': # hour [hh]
                s += '{0:02d}'.format(int(get_hour(t)))
            elif fstr[i] == 'I': # hour (12h format) [hh]
                h = int(get_hour(t))
                if h > 12:
                    h -= 12
                s += '{0:02d}'.format(h)
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
                s += '{0:02d}'.format(int((get_year(t)) % 100))
            elif fstr[i] == 'Y': # year [YYYY]
                s += '{0:04d}'.format(int(get_year(t)))
            # extension
            elif fstr[i] == '#': # milliseconds
                s += '{0:03d}'.format(int(np.round(np.mod(dt64_to(t, 'ms', returnfloat=True), 1000))))
            else:
                logger.warn('Unknown format specifier: ' + fstr[i])
                replacements.pop()
                
        else:
            s += fstr[i]
        
        i += 1

    return s


def _strftime_td64(td, fstr, customspec=None):
    '''
    Private strftime function. Must be called with numpy.timedelta64,
    not in an array.
    '''

    if isnat(td):
        return 'NaT'

    td2 = td.tolist() # date.timedelta object
    i = 0
    s = ''
    replacements = []
    while i < len(fstr):
        if fstr[i] == '%':
            # Look to next char
            i += 1
            replacements.append(fstr[i])
        
            if customspec is not None and fstr[i] in customspec:
                # Insert user custom specifier
                s += customspec[fstr[i]]
                
            elif fstr[i] == '%': # percent
                s += '%'
                
            elif fstr[i] == 'd': # number of whole days
                s += '{0:d}'.format(td2.days)
            elif fstr[i] == 'H': # hour [hh]
                s += '{0:02d}'.format(td2.seconds/3600)
            elif fstr[i] == 'M': #
                s += '{0:02d}'.format(np.mod(td2.seconds/60, 60))
            elif fstr[i] == 's': # total number of seconds
                s += '{0:d}'.format(td2.total_seconds())
            elif fstr[i] == 'S': # seconds [ss]
                s += '{0:02d}'.format(np.mod(td2.seconds, 60))
            # extension
            elif fstr[i] == '#': # milliseconds
                # Use np.round to get rounding to even number
                s += '{0:03d}'.format(int(np.round(td2.microseconds / 1000.0)))
            else:
                logger.warn('Unknown format specifier: ' + fstr[i])
                replacements.pop()
                
        else:
            s += fstr[i]
        
        i += 1

    return s
