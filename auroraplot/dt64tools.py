# locator class for numpy.Datetime64
"""Tools for numpy.datetime64 and numpy.timedelta64 time classes"""

import builtins
import datetime
import logging
import math
import os
import re
import unittest
from typing import List, overload, Optional, Tuple, Union

import numpy
import numpy as np
from numpy.typing import NDArray
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import Locator
from matplotlib.ticker import Formatter

import auroraplot as ap
from auroraplot.decorators import deprecated

logger = logging.getLogger(__name__)

time_label = 'Time'
date_label = 'Date'

epoch64_us = np.datetime64('1970-01-01T00:00:00Z', 'us')
epoch64_day = np.datetime64('1970-01-01T00:00:00', 'D')

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


def wrap_day(ts: Union[np.timedelta64, np.ndarray]) -> Union[np.timedelta64, np.ndarray]:
    """Wrap timedelta24 into 0 to 24 hours"""
    return ts - np.floor((ts / np.timedelta64(1, 'D'))) * np.timedelta64(1, 'D')


def get_time_type(t):
    s = t.dtype.name
    return s[:s.rindex('[')]


def get_units(t):
    # assert isinstance(t, (np.datetime64, np.timedelta64)), \
    #    'Must be of type numpy.datetime64 or numpy.timedelta64'
    match = re.search(r'\[([^]]+)\]', t.dtype.str)
    assert match is not None, f'Failed to parse units in numpy.datetime64 type: {t.dtype.str}'
    return match.groups()[0]


def smallest_unit(a):
    i = None
    for u in a:
        if u is None:
            continue
        elif isinstance(u, str):
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
    elif not isinstance(units, str):
        units = get_units(units)
    if time_type is None:
        time_type = get_time_type(t)
    elif not isinstance(time_type, str):
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


def from_ymd(year, month, day):
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


@deprecated(from_ymd)
def from_YMD(year, month, day):  # noqa
    return from_ymd(year, month, day)


def from_hms(h: int, m: int, s: int, ms: int = None) -> np.timedelta64:
    r = np.timedelta64(h, 'h') + np.timedelta64(m, 'm') + np.timedelta64(s, 's')
    if ms is not None:
        r += np.timedelta64(ms, 'ms')
    return r


def from_dhms(d: int, h: int, m: int, s: int, ms: int = None) -> np.timedelta64:
    r = np.timedelta64(d, 'D') + np.timedelta64(h, 'h') + np.timedelta64(m, 'm') + np.timedelta64(s, 's')
    if ms is not None:
        r += np.timedelta64(ms, 'ms')
    return r


def to_dhms(ts: np.timedelta64) -> tuple[int, int, int, float]:
    days = int(ts / np.timedelta64(1, 'D'))
    ts -= np.timedelta64(days, 'D')
    hours = int(ts / np.timedelta64(1, 'h'))
    ts -= np.timedelta64(hours, 'h')
    minutes = int(ts / np.timedelta64(1, 'm'))
    ts -= np.timedelta64(minutes, 'm')
    seconds = float(ts / np.timedelta64(1, 's'))
    return days, hours, minutes, seconds


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
    return np.array(x).astype('int64') == -(2 ** 63)


def get_utc_days(st, et):
    day = np.timedelta64(1, 'D')
    t1 = st + 0 * day  # Units are D or higher precision

    while t1 < et:
        t2 = astype(floor(t1 + day, day), units=t1)
        if t2 > et:
            t2 = astype(et, units=t1)
        yield t1, t2
        t1 = t2


def get_date(t: Union[np.datetime64, np.ndarray]) -> Union[np.datetime64, np.ndarray]:
    return t.astype('<M8[D]')


def get_time_of_day(t: Union[np.datetime64, np.ndarray]) -> Union[np.datetime64, np.ndarray]:
    # td64_units = t.dtype.str.lower()  # timedelta64 equivalent units
    # print(f'td64_units: {td64_units}')
    # d = np.timedelta64(24, 'h').astype(td64_units).astype('int64')
    # return np.mod(t.astype(td64_units).astype('int64'), d).astype(td64_units)
    return t - get_date(t)


def _get_tt(a, attr):
    aa = np.array(a)  # Handle all array-like possibilities
    aaf = aa.flatten()
    r = np.zeros_like(aaf, dtype=int)
    for n in range(aaf.size):
        r[n] = getattr(aaf[n].tolist().timetuple(), attr)
    if aa.shape == ():
        return int(r[0])
    else:
        return r.reshape(aa.shape).astype('int64')


def get_year(a):
    """Return year"""
    return _get_tt(a, 'tm_year')


def get_month(a):
    """Return month"""
    return _get_tt(a, 'tm_mon')


def get_day_of_year(a):
    """Return day of year in range 1 to 366"""
    return _get_tt(a, 'tm_yday')


def get_day_of_month(a):
    """Return day of month"""
    return _get_tt(a, 'tm_mday')


# Alias
get_day = get_day_of_month


def get_week_day(a):
    """Return day of week, in range 0 (Monday) to 6 (Sunday)"""
    return _get_tt(a, 'tm_wday')


def get_hour(a):
    """Return hour of day"""
    return _get_tt(a, 'tm_hour')


def get_minute(a):
    """Return minute of day"""
    return _get_tt(a, 'tm_min')


def get_second(a):
    """Return second of day"""
    return _get_tt(a, 'tm_sec')


def get_start_of_month(a):
    a = np.array(a)
    af = a.flatten()  # Handle all array-like possibilities
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


def is_leap_year(t: Union[numpy.datetime64, int]) -> bool:
    if isinstance(t, numpy.datetime64):
        if isnat(t):
            raise ValueError('time is not valid')
        return is_leap_year(get_year(t))
    if isinstance(t, int):
        if t % 4 != 0:
            return False
        if t % 400 == 0:
            return True
        if t % 100 == 0:
            return False
        return True
    else:
        raise TypeError(f'Expected numpy.datetime64 or integer, received {type(t).__name__} instead')


def get_days_in_year(t: Union[numpy.datetime64, int]) -> int:
    return 366 if is_leap_year(t) else 365


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
            raise Exception(f'Cannot use {func_name} with units of {td_units}')
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
    """
    Round numpy.datetime64 or numpy.timedelta64 to nearest interval.

    dt: numpy.datetime64 (or numpy.timedelta64) value to be rounded.

    td: numpy.timedelta64 interval to round to.

    Returns:
    numpy.datetime64 or numpy.timedelta64 rounded to the
    nearest multiple of "td".
    """
    return _round_to_func(dt, td, np.round, 'round')


def floor(dt, td):
    """
    Round numpy.datetime64 or numpy.timedelta64 to down to nearest
    interval.

    dt: numpy.datetime64 (or numpy.timedelta64) value to be rounded down.

    td: numpy.timedelta64 interval to round down to.

    Returns:
    numpy.datetime64 or numpy.timedelta64 rounded down the
    nearest multiple of "td".
    """
    return _round_to_func(dt, td, np.floor, 'floor')


def ceil(dt, td):
    """
    Round numpy.datetime64 or numpy.timedelta64 to up to nearest
    interval.

    dt: numpy.datetime64 (or numpy.timedelta64) value to be rounded up.

    td: numpy.timedelta64 interval to round up to.

    Returns: numpy.datetime64 or numpy.timedelta64 rounded up the
        nearest multiple of "td".
    """
    return _round_to_func(dt, td, np.ceil, 'ceil')


def dt64_range(start, stop, step):
    """
    Return a range of numpy.datetime64 values, similar to the Python
    2.x xrange(), or Python 3.x range(), where the values are created
    by a generator.

    start: numpy.datetime64 start time.

    stop: numpy.datetime64 stop time (exclusive).

    step: numpy.timedelta64 interval between times.

    Returns: numpy.datetime64 values.
    """
    # Initial start time to have the same resolution as successive times
    t = start + 0 * step
    while t < stop:
        yield t
        t += step


def fmt_dt64_range(st, et):
    """
    Return a string representing an interval defined by two
    numpy.datetime64 values, such as an interval of data defined by
    start and end times.

    st: start time

    et: end time

    Returns: Compact string representation. Seconds are included only
        if required.Fractional seconds not currently supported.
    """

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
            return strftime(st, '%Y-%m-%d %H:%M:%S - ') + strftime(et, '%H:%M:%S')
        else:
            return strftime(st, '%Y-%m-%d %H:%M - ') + strftime(et, '%H:%M')
    elif st == floor(st, day) and et == st + day:
        # Entire day
        return strftime(st, '%Y-%m-%d')
    else:
        # Start and end time on different dates
        if get_second(st) or get_second(et):
            return strftime(st, '%Y-%m-%d %H:%M:%S - ') + strftime(et, '%Y-%m-%d %H:%M:%S')
        else:
            return strftime(st, '%Y-%m-%d %H:%M - ') + strftime(et, '%Y-%m-%d %H:%M')


def _add_timezone(s, timezone='+0000'):
    """
    Add timezone if one is not present.

    The default timezone used is +0000 (UTC)
    """
    if s.endswith('Z') or re.search(r'[+-]\d{2}(:?\d{2})?$', s):
        # Already has timezone indicator (+00, +0000, +00:00)
        return s
    else:
        # Default to UTC
        return s + timezone


def parse_datetime64(s, prec, now=None, timezone='+0000'):
    """
    Parse a numpy.datetime64() time, also accepting "yesterday",
    "today", "now" and "tomorrow", with the result given to the
    specified precision.

    s: string to parse

    prec: precision of the returned time. Must be hour ("h") or better.

    Returns: numpy.datetime64 value.
    """

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
        t = np.datetime64(_add_timezone(s, timezone))

    return astype(t, prec)


def parse_timedelta64(s, prec):
    r = np.timedelta64(0, prec)
    value_next = True
    values = []
    units = []
    for w in s.split():
        m = re.match('^([+-]?[0-9]+)?(as|fs|ps|ns|us|ms|s|m|h|D|W|M|Y)?$', w)
        if m is None:
            raise ValueError('unknown value/unit (%s)' % w)
        v, u = m.groups()
        if v is not None and u is not None:
            # Value and unit
            if not value_next:
                raise ValueError('unit expected but found %s' % repr(w))
            values.append(v)
            units.append(u)
        elif v is None and u is not None:
            # unit only
            if value_next:
                raise ValueError('value expected but found %s' % repr(w))
            units.append(u)
            value_next = True
        elif v is not None and u is None:
            # value only
            if not value_next:
                raise ValueError('unit expected but found %s' % repr(w))
            values.append(v)
            value_next = False

    if not value_next:
        raise ValueError('Last value missing unit: %s' % repr(s))

    for n in range(len(values)):
        r += np.timedelta64(int(values[n]), units[n])
    return astype(r, units=prec)


def t_to_axis_value(t, axis, units=None, epoch=None, fmt=None):
    t_units = get_units(t)
    if hasattr(axis, 'dt64tools'):
        # Axis already has our data
        axis_data = axis.dt64tools
        assert t.dtype.type == axis_data.type, f"Cannot add {t.dtype.type} to axis using {axis_data.type}"
        assert time_units.index(t_units) >= time_units.index(
            axis_data.units), 'Cannot add time data with units ' + t_units + ' to existing plot using units ' + axis_data.units
        if axis_data.epoch is None:
            # Set and save epoch
            axis_data.epoch = axis_data.type(0, units)
            axis.dt64tools = axis_data

    else:
        if units is None:
            if t.dtype.type == np.datetime64 and ap.plot_datetime_units:
                units = ap.plot_datetime_units
            elif t.dtype.type == np.timedelta64 and ap.plot_timedelta_units:
                units = ap.plot_timedelta_units
            else:
                units = t_units

        if epoch is None:
            if t.dtype.type == np.datetime64 and ap.plot_datetime_epoch:
                epoch = ap.plot_datetime_epoch
            elif t.dtype.type == np.timedelta64 and ap.plot_timedelta_epoch:
                epoch = ap.plot_timedelta_epoch
            else:
                epoch = t.dtype.type(0, units)

        axis_data = Dt64ToolsData(units=units,
                                  type=t.dtype.type,
                                  epoch=dt64_to(epoch, units).astype('int64'))
        axis.dt64tools = axis_data
        # Set some default formatting
        # TODO: Correct locator and formatter for timedelta64
        axis.set_major_locator(Datetime64Locator())
        # if axis_data.type == np.datetime64:
        axis.set_major_formatter(Datetime64Formatter(fmt=fmt, data_type=t.dtype.type))

        # plt.xticks(rotation=-25)

    # Transform x to suitable units
    val = dt64_to(t, axis_data.units)

    if axis_data.epoch is not None:
        val -= axis_data.epoch
    return val


def plot_dt64(x, y, axes=None,
              # Our own options
              x_time_units=None, y_time_units=None, plot_func=plt.plot,
              x_time_fmt=None, y_time_fmt=None,
              x_time_epoch=None, y_time_epoch=None,
              #
              **kwargs):
    if axes is None:
        axes = plt.gca()

    xdate = False
    try:
        if x.dtype.type in (np.datetime64, np.timedelta64):
            xdate = True
    except Exception:
        pass

    ydate = False
    try:
        if y.dtype.type in (np.datetime64, np.timedelta64):
            ydate = True
    except Exception:
        pass

    if xdate:
        units = get_units(x)
        assert units in time_units, 'Unknown units'
        assert units not in ['M', 'Y'], 'Time units cannot be month or year'
        xx = t_to_axis_value(x, axes.xaxis, units=x_time_units, epoch=x_time_epoch, fmt=x_time_fmt)
    else:
        xx = x

    if ydate:
        units = get_units(y)
        assert units in time_units, 'Unknown units'
        assert units not in ['M', 'Y'], 'Time units cannot be month or year'
        yy = t_to_axis_value(y, axes.yaxis, units=y_time_units, epoch=y_time_epoch, fmt=y_time_fmt)
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
        xmin = t_to_axis_value(xmin, ax.xaxis)
    if xmax is not None:
        xmax = t_to_axis_value(xmax, ax.xaxis)
    return ax.set_xlim(xmin=xmin, xmax=xmax)


def ylim_dt64(ymin=None, ymax=None, ax=None):
    if ax is None:
        ax = plt.gca()
    if ymin is not None:
        ymin = t_to_axis_value(ymin, ax.yaxis)
    if ymax is not None:
        ymax = t_to_axis_value(ymax, ax.yaxis)
    return ax.set_ylim(ymin=ymin, ymax=ymax)


def get_plot_units(axis):
    return axis.dt64tools.units


def highlight(ax, st, et, color='y', **kwargs):
    u = get_plot_units(ax.xaxis)
    yl = ax.get_ylim()
    x1 = dt64_to(st, u)
    x2 = dt64_to(et, u)
    ax.add_patch(patches.Rectangle([x1, yl[0]], x2 - x1, yl[1] - yl[0],
                                   **kwargs))


def get_ratio_solar_to_sidereal_day() -> float:
    """Get the ratio of the duration of one solar day to one sidereal day"""
    return 1.00273790935


def get_sidereal_day(units: Union[str, np.datetime64, np.timedelta64] = 's') -> numpy.timedelta64:
    """Get the length of a sideral day"""

    if isinstance(units, (np.datetime64, np.timedelta64)):
        units = get_units(units)
    sd = (np.timedelta64(23, 'h')
          + np.timedelta64(56, 'm')
          + np.timedelta64(4, 's')
          + np.timedelta64(100, 'ms'))
    return sd.astype(np.timedelta64(0, units))


def julian_date(t: Union[np.datetime64, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert to Julian date"""

    # Julian date starts 12:00:00 1/1/4713 BC and is Julian day 0.
    # A useful calculator between UTC and julian date can be found at
    # http://www.fourmilab.ch/cgi-bin/uncgi/Earth

    # 1970-01-01T00:00:00 UTC is Julian date 2440587.50000
    epoch_in_jd = np.timedelta64(2440587, 'D') + np.timedelta64(12, 'h')
    since_epoch = t - epoch64_day
    return (epoch_in_jd + since_epoch) / np.timedelta64(1, 'D')


def get_unix_day_number(t: Union[np.datetime64, np.ndarray]) -> Union[int, np.ndarray]:
    """Get the day number since the Unix epoch (1970-01-01)"""
    return (t - epoch64_day) // np.timedelta64(1, 'D')


@overload
def gmst(t: np.datetime64) -> np.timedelta64:
    pass


@overload
def gmst(t: NDArray[np.datetime64]) -> NDArray[np.timedelta64]:
    pass


def gmst(t):
    """Get the Greenwich Mean Sidereal Time at 00:00 UT"""

    # From "The Astronomical Almanac 1996, p B6, B7"
    # t_u = interval of time, measured in Julian centuries of 36525 days of
    # universal time (mean solar days), elapsed since the epoch 2000
    # January 1d 12h UT

    # Calculate t_u based at time at start of day
    t_u = (julian_date(get_date(t)) - 2451545)/36525
    ts = 24110.54841 + 8640184.812866 * t_u + (0.093104 * t_u**2) - (6.2e-6 * t_u**3)

    if isinstance(t, np.datetime64):
        return np.timedelta64(builtins.round(ts % 86400), 's')
    else:
        # Input was an array
        ts += 0.5
        ts %= 86400
        return ts.astype('timedelta64[s]')


@overload
def utc_to_lmst(t: np.datetime64, longitude_deg: float) -> np.timedelta64:
    pass

@overload
def utc_to_lmst(t: NDArray[np.datetime64], longitude_deg: Union[float, NDArray[np.float64]]) -> NDArray[np.timedelta64]:
    pass

def utc_to_lmst(t, longitude_deg):
    """Convert a UTC time to the mean local sidereal time

    When the longitude is East of the Greenwich meridian its value should be positive.
    """
    # From "The Astronomical Almanac 1996, p B6, B7"

    # GMST at 00:00 UT
    ts = gmst(t)
    # Add the equivalent mean sidereal time from 0h to the time

    gmst2 = wrap_day(gmst(t) + (get_time_of_day(t) + np.timedelta64(0, 'ms')) * get_ratio_solar_to_sidereal_day())

    # milliseconds = (get_time_of_day(t) / np.timedelta64(1, 'ms')) * get_ratio_solar_to_sidereal_day()
    # # ts = (ts / np.timedelta64(1, 'ms')) + milliseconds
    # ts += np.timedelta64(round(milliseconds), 'ms')
    # ts = wrap_day(ts)
    # ts %= 86400  # Remove any extra days

    # ts = wrap_day(ts + np.timedelta64(int(milliseconds + 0.5), 'ms'))

    ### % gmst = Greenwich Mean Sidereal Time
    # gmst2 = np.timedelta64(builtins.round(ts), 's')
    # gmst2 = t2

    # Local mean sidereal time

    longitude_deg = ap.wrap_degrees(longitude_deg)

    # time_diff = np.timedelta64(builtins.round(86_400_000.0 * longitude_deg / 360.0), 'ms')
    time_diff = np.timedelta64(86_400_000, 'ms') * (longitude_deg / 360.0)
    return wrap_day(gmst2 + time_diff)

    # if ts < np.timedelta64(0, 's'):
    #     ts += np.timedelta64(1, 'D')


def lmst_to_utc(lmst: Union[np.timedelta64, np.ndarray], longitude_deg: Union[float, np.ndarray], d: Union[np.datetime64, np.ndarray]) -> Union[np.datetime64, np.ndarray]:
    """Convert local mean sidereal time to in UTC"""

    #  From "The Astronomical Almanac 1996, p B6, B7"
    longitude_deg = ap.wrap_degrees(longitude_deg)

    # Add west longitude (subtract east longitude)
    # NB west longitude is negative!
    t = lmst - (np.timedelta64(86400_000, 'ms') * (longitude_deg / 360))
    t = wrap_day(t)

    # Subtract Greenwich mean sidereal time at 0h UTC
    t -= gmst(d)
    t = wrap_day(t)

    # Convert to the equivalent UTC interval
    utc = t * (1 / get_ratio_solar_to_sidereal_day())
    d_day = get_date(d)
    d_next_day = d_day + np.timedelta64(1, 'D')
    r = utc + d_day
    if isinstance(r, np.ndarray):
        idx = r < get_date(d)
        r[idx] += get_sidereal_day('ms')
        idx = r >= d_next_day
        r[idx] -= get_sidereal_day('ms')
    elif isinstance(r, np.datetime64):
        if r < d_day:
            r += get_sidereal_day('ms')
        if r >= d_next_day:
            r -= get_sidereal_day('ms')
    else:
        raise TypeError(f'Unexpected return type {type(r).__name__}')
    return r


def get_sidereal_time_offset(t: Union[np.datetime64, np.ndarray],
                             longitude_deg: float,
                             st: Optional[Union[np.datetime64, np.ndarray]] = None,
                             units: str = 'ms') -> Union[Union[np.timedelta64, np.ndarray], Tuple[Union[np.timedelta64, np.ndarray], Union[int, np.ndarray]]]:
    """Get the sidereal time offset from UTC"""

    sidereal_day = get_sidereal_day(units)
    sidereal_day_ms = get_sidereal_day('ms')
    # On this day find out what time sidereal midnight is
    midnight_sidt = lmst_to_utc(np.timedelta64(0, units), longitude_deg, t)

    # If midnight is after time t then use the sidereal midnight from the previous day
    if isinstance(t, np.datetime64):
        if midnight_sidt > t:
            midnight_sidt = lmst_to_utc(np.timedelta64(0, units), longitude_deg, t - sidereal_day)
        r = t - midnight_sidt
        if r >= get_sidereal_day('ms'):  # Use the best resolution for comparison
            r -= sidereal_day
        if r < np.timedelta64(0, units):
            r += sidereal_day
    elif isinstance(t, np.ndarray):
        idx = midnight_sidt > t
        if np.any(idx):
            midnight_sidt[idx] = lmst_to_utc(np.timedelta64(0, units), longitude_deg, t[idx])
        r = t - midnight_sidt
        idx = r >= get_sidereal_day('ms')  # Use the best resolution for comparison
        if np.any(idx):
            r[idx] -= sidereal_day
        idx = r < np.timedelta64(0, units)
        if np.any(idx):
            r[idx] += sidereal_day
    else:
        raise TypeError(f'Unexpected return type {type(t).__name__}')

    if st is not None:
        # Calculate in which sidereal day each sample falls; let the first day be zero.
        sd = np.floor((t - st + get_sidereal_time_offset(st, longitude_deg)) / sidereal_day)
        if isinstance(sd, np.ndarray):
            sd = sd.astype(int)
        else:
            sd = int(sd)
        return r, sd
    else:
        return r


class Dt64ToolsData(object):
    def __init__(self, units, type, epoch):
        self.units = units
        self.type = type
        self.epoch = epoch


class Datetime64Locator(Locator):
    """Locator class for numpy.datetime64"""

    def __init__(self, maxticks=8, interval=None):
        self.maxticks = maxticks
        self.interval = interval

    def __call__(self):
        """Return the locations of the ticks"""
        if not hasattr(self.axis, 'dt64tools'):
            raise Exception('dt64tools data not found on axis')
        units = self.axis.dt64tools.units
        epoch = self.axis.dt64tools.epoch
        limits = self.axis.get_view_interval()
        if np.diff(limits) < 1:
            logger.warning('limit for tick labels reached')
            return []

        limits_dt64 = ((epoch + limits).astype('m8[' + units + ']') +
                       self.axis.dt64tools.type(0, units))

        if self.interval is not None:
            tick_interval = self.interval
        else:
            tick_interval = self.calc_interval(limits_dt64[0], limits_dt64[1])
        if tick_interval is not None:
            first_tick = ceil(limits_dt64[0], tick_interval)
            last_tick = floor(limits_dt64[1], tick_interval)
            num = np.floor(((last_tick - first_tick) /
                            tick_interval) + 1).astype('int64')
            tick_locs = ((dt64_to(tick_interval, units).astype('int64')
                          * np.arange(num))
                         + first_tick.astype('int64')) - epoch

            try:
                return self.raise_if_exceeds(tick_locs)
            except RuntimeError as e:
                logger.exception(e)
                return []

        else:
            return self.raise_if_exceeds(
                self.calc_irregular_ticks(limits_dt64[0], limits_dt64[1],
                                          units))

    def calc_interval(self, st, et):
        """
        Compute the tick location interval. Only useful for durations
        where regular ticks can be used. See also
        calc_irregular_ticks().
        """

        # Compute the interval required, as a FP number of seconds,
        # assuming the maximum number of ticks.
        approx_ival_s = dt64_to(et - st, 's', returnfloat=True) / (self.maxticks - 1)

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
                         10 ** int(np.floor(approx_ival_log10 - unit_log10))).astype('m8[' + best_unit + ']')
            approx_ival = np.timedelta64(int(approx_ival_s * 10 ** -unit_log10),
                                         best_unit)
            try:
                idx = (approx_ival >= intervals).tolist().index(False)
                tick_interval = intervals[idx]
                return intervals[idx]
            except ValueError:
                # Not found, fall through and use 1s or larger interval
                pass
            except Exception:
                raise

        if approx_ival_s < 7 * 86400:  # < 3 weeks
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

        if approx_ival_s < 365 * 86400:  # Try for up to about 6 months
            # Try using months
            som = [get_start_of_month(st)]
            while som[-1] <= et:
                som.append(get_start_of_next_month(som[-1]))

            if som[0] < st:
                som.pop(0)  # Too early, remove
            if som[-1] > et:
                som.pop()  # Too late, remove

            month0 = get_month(som) - 1

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
                if from_ymd(tick_years[0], 1, 1) < st:
                    tick_years.pop(0)
                if from_ymd(tick_years[-1], 1, 1) > et:
                    tick_years.pop()
                if len(tick_years) <= self.maxticks:
                    # Success
                    return dt64_to(from_ymd(tick_years, 1, 1), units).tolist()
            years *= 10

        raise Exception('Failed to compute correct yearly tick spacing')


class Datetime64Formatter(Formatter):
    def __init__(self, fmt=None, autolabel=None, data_type=np.datetime64):
        """
        Format ticks with date/time as appropriate.

        fmt: Manually set the strftime format string.

        autolabel: Label the axis.
        If False no labeling will occur.
        If True a axis label will be updated when the axis ticks are
        formatted.
        If None the axis label will be updated only if it is an empty
        string. This is the default.
        """

        self.fmt = fmt
        self.autolabel = autolabel
        self.data_type = data_type

    def __call__(self, x, pos=None):
        # This is also called to show cursor location, where pos=None

        if not hasattr(self.axis, 'dt64tools'):
            raise Exception('Cannot find axis time units')

        units = self.axis.dt64tools.units
        epoch = self.axis.dt64tools.epoch
        fmt = self.fmt
        if pos is None:
            # Showing cursor location
            if fmt is None:
                if self.data_type == np.timedelta64:
                    fmt = '%d %H:%M:%S'
                else:
                    fmt = '%Y-%m-%dT%H:%M:%SZ'
        elif fmt is None:
            xadi = self.axis.get_data_interval()
            if np.all(np.isfinite(xadi)):
                data_interval = np.timedelta64(int(np.diff(xadi)), units)
            else:
                # Nothing plotted, so use current limits for time range
                data_interval = np.timedelta64(int(np.diff(self.axis.get_view_interval())), units)

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
                        tick_interval >= np.timedelta64(365, 'D'):
                    f.append('%Y')
                else:
                    f.append('%Y-%m-%d')

            # Choose seconds and fractional seconds depending on current view
            if tick_interval < np.timedelta64(1, 's'):
                f.append('%H:%M:%S.%#')
            elif tick_interval < np.timedelta64(1, 'm'):
                f.append('%H:%M:%S')
            elif tick_interval < np.timedelta64(1, 'h'):
                f.append('%H:%M')
            elif tick_interval < np.timedelta64(1, 'D'):
                f.append('%H')
            fmt = '\n'.join(f)

        t = self.data_type(int(x + epoch), units)
        r = strftime(t, fmt)
        if pos == 1:
            self.axis.dt64tools.tick_fmt = fmt
            if self.autolabel or (self.autolabel is None and
                                  self.axis.get_label().get_text() == ''):
                if r == strftime(t + np.timedelta64(1, 'D'), fmt):
                    # No date information present in string
                    s = time_label
                else:
                    s = date_label

                if isinstance(self.autolabel, str):
                    s = self.autolabel % s

                self.axis.get_label().set_text(s)

        return r


def strftime(t, fstr):
    tf = t.flatten()  # Handle all array-like possibilities
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
    """
    Private strftime function. Must be called with numpy.datetime64,
    not in an array.
    """

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
    num_suffix = np.array(['th'] * 32)  # use 1-based indexing
    num_suffix[[1, 21, 31]] = 'st'
    num_suffix[[2, 22]] = 'nd'
    num_suffix[[3, 23]] = 'rd'

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

            elif fstr[i] == '%':  # percent
                s += '%'

            elif fstr[i] == 'a':  # abbreviated day name
                s += d_name[get_week_day(t)]
            elif fstr[i] == 'A':  # day name
                s += day_name[get_week_day(t)]
            elif fstr[i] == 'b':  # abbreviated month name
                s += m_name[get_month(t) - 1]
            elif fstr[i] == 'B':  # month name
                s += month_name[get_month(t) - 1]

            elif fstr[i] == 'd':  # day [dd]
                s += '{0:02d}'.format(int(get_day_of_month(t)))
            elif fstr[i] == 'H':  # hour [hh]
                s += '{0:02d}'.format(int(get_hour(t)))
            elif fstr[i] == 'I':  # hour (12h format) [hh]
                h = int(get_hour(t))
                if h > 12:
                    h -= 12
                s += '{0:02d}'.format(h)
            elif fstr[i] == 'j':  # day of year [jjj]
                s += '{0:03d}'.format(int(get_day_of_year(t)))
            elif fstr[i] == 'm':  #
                s += '{0:02d}'.format(int(get_month(t)))
            elif fstr[i] == 'M':  #
                s += '{0:02d}'.format(int(get_minute(t)))
            elif fstr[i] == 'q':  # quarter (of year)
                s += '{0:01d}'.format(int(get_month(t)) // 4 + 1)
            elif fstr[i] == 's':  # seconds since unix epoch
                s += '{0:d}'.format(int(dt64_to(t, 's')))
            elif fstr[i] == 'S':  # seconds [ss]
                s += '{0:02d}'.format(int(get_second(t)))
            # extension
            elif fstr[i] == 't':  # st, nd, rd or th day number suffix
                s += num_suffix[get_day_of_month(t)]
            elif fstr[i] == 'y':  # year [yy]
                s += '{0:02d}'.format(int((get_year(t)) % 100))
            elif fstr[i] == 'Y':  # year [YYYY]
                s += '{0:04d}'.format(int(get_year(t)))
            # extension
            elif fstr[i] == '#':  # milliseconds
                s += '{0:03d}'.format(int(np.round(np.mod(dt64_to(t, 'ms', returnfloat=True), 1000))))
            else:
                logger.warning('Unknown format specifier: ' + fstr[i])
                replacements.pop()

        else:
            s += fstr[i]

        i += 1

    return s


def _strftime_td64(td, fstr, customspec=None):
    """
    Private strftime function. Must be called with numpy.timedelta64,
    not in an array.
    """

    if isnat(td):
        return 'NaT'

    td2 = td.tolist()  # date.timedelta object
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

            elif fstr[i] == '%':  # percent
                s += '%'

            elif fstr[i] == 'd':  # number of whole days
                s += '{0:d}'.format(int(td2.days))
            elif fstr[i] == 'H':  # hour [hh]
                s += '{0:02d}'.format(int(td2.seconds / 3600))
            elif fstr[i] == 'M':  #
                s += '{0:02d}'.format(int(np.mod(td2.seconds / 60, 60)))
            elif fstr[i] == 's':  # total number of seconds
                s += '{0:d}'.format(int(td2.total_seconds()))
            elif fstr[i] == 'S':  # seconds [ss]
                s += '{0:02d}'.format(int(np.mod(td2.seconds, 60)))
            # extension
            elif fstr[i] == '#':  # milliseconds
                # Use np.round to get rounding to even number
                s += '{0:03d}'.format(int(np.round(td2.microseconds / 1000.0)))
            else:
                logger.warning('Unknown format specifier: ' + fstr[i])
                replacements.pop()

        else:
            s += fstr[i]

        i += 1

    return s
