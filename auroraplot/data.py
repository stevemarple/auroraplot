import copy
import logging
import os
import pickle
import six
import traceback

# Python 2/3 compatibility
try:
    from urllib.request import urlopen
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse
    from urllib import urlopen

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import scipy.interpolate
import scipy.stats
import scipy.special

import auroraplot as ap
import auroraplot.tools
import auroraplot.dt64tools as dt64


logger = logging.getLogger(__name__)


def leastsq_error(p, obj, ref, channel):
    """
    Error function used with Data.fit() for least squares error fitting.
    """
    adj, = p
    err = obj.data[obj.get_channel_index(channel)[0]] - (ref.data[ref.get_channel_index(channel)[0]] + adj)
    return np.nan_to_num(err)


def _generic_load_converter(file_name, archive_data, 
                            project, site, data_type, start_time,
                            end_time, archive, channels, **kwargs):
    """
    A generic load converter.

    Expects the archive information to contain:

    constructor: Function reference to the class constructor

    timestamp_method: one of 'unixtime', 'offset0', 'offset1'
    """

    assert 'constructor' in archive_data, 'archive data must indicate constructor to use'
    assert archive_data['constructor'].__name__ == data_type, 'data type does not match'
    assert 'timestamp_method' in archive_data, 'archive data must indicate the timestamp method used'
    
    chan_tup = tuple(archive_data['channels'])

    try:
        if file_name.startswith('/'):
            uh = urlopen('file:' + file_name)
        else:
            uh = urlopen(file_name)
        try:
            kwargs = {}
            for s in ('comments', 'delimiter', 'converters', 'skiprows'):
                if s in archive_data:
                    kwargs[s] = archive_data[s]
                      
            data = np.loadtxt(uh, unpack=True, **kwargs)

            sample_time_units = dt64.get_units(archive_data['nominal_cadence'])
            if archive_data['timestamp_method'] == 'unixtime':
                sst = np.round(data[0] / dt64.multipliers[sample_time_units])
                sample_start_time = sst.astype('datetime64[' + sample_time_units + ']')
                col_offset = 1                
            elif archive_data['timestamp_method'] == 'offset0':
                sample_start_time = (data[0].astype('int64') * archive_data['nominal_cadence']) + start_time
                col_offset = 1
            elif archive_data['timestamp_method'] == 'offset1':
                sample_start_time = ((data[0].astype('int64') - 1) * archive_data['nominal_cadence']) + start_time
                col_offset = 1

            elif archive_data['timestamp_method'] in ('Y', 'YM', 'YMD', 'YMDh', 'YMDhm', 'YMDhms'):
                col_offset = len(archive_data['timestamp_method'])
                sample_start_time = (data[0]-1970).astype('datetime64[Y]')
                for n in range(1, col_offset):
                    tu = archive_data['timestamp_method'][n]
                    if tu in ('M', 'D'):
                        # Cannot use add-assign here
                        sample_start_time = sample_start_time + (data[n]-1).astype('timedelta64[%s]' % tu)
                    else:
                        # Cannot use add-assign here
                        sample_start_time = sample_start_time + data[n].astype('timedelta64[%s]' % tu)
            elif archive_data['timestamp_method'].startswith('YMDhms.'):
                tu = archive_data['timestamp_method'].replace('YMDhms.', '')
                sample_start_time = (data[0] - 1970).astype('datetime64[Y]') + \
                                    (data[1] - 1).astype('timedelta64[M]') + \
                                    (data[2] - 1).astype('timedelta64[D]') + \
                                    data[3].astype('timedelta64[h]') + \
                                    data[4].astype('timedelta64[m]') + \
                                    np.round(data[5] / dt64.multipliers[tu]).astype('timedelta64[%s]' % tu)
                col_offset = 6              
            elif archive_data['timestamp_method'] in ('h', 'hm', 'hms'):
                col_offset = len(archive_data['timestamp_method'])
                sample_start_time = start_time
                for n in range(col_offset):
                    # Cannot use add-assign here
                    sample_start_time = sample_start_time + \
                                        (data[n].astype('timedelta64[%s]' % archive_data['timestamp_method'][n]))

            else:
                raise ValueError('unknown value for timestamp_method')

            sample_end_time = sample_start_time + archive_data['nominal_cadence']
            integration_interval = np.tile(archive_data['nominal_cadence'],
                                           [np.size(channels), 
                                            np.size(sample_start_time)])

            col_idx = []
            for c in channels:
                col_idx.append(col_offset + chan_tup.index(c))


            data = np.reshape(data[col_idx],
                              [len(col_idx), np.size(sample_start_time)])
            
            if archive_data['data_multiplier']:
                data /= archive_data['data_multiplier']

            if 'valid_range' in archive_data:
                data[np.logical_or(data < archive_data['valid_range'][0], data > archive_data['valid_range'][1])] \
                    = np.nan

            tu = dt64.get_units(archive_data['nominal_cadence'])
            r = archive_data['constructor'](project=project,
                                            site=site,
                                            channels=channels,
                                            start_time=dt64.astype(start_time, tu),
                                            end_time=dt64.astype(end_time, tu),
                                            sample_start_time=dt64.astype(sample_start_time, tu),
                                            sample_end_time=dt64.astype(sample_end_time, tu),
                                            integration_interval=integration_interval,
                                            nominal_cadence=archive_data['nominal_cadence'],
                                            data=data,
                                            units=archive_data['units'],
                                            sort=archive_data.get('sort', False))
            return r

        except Exception as e:
            if 'raise_all' in kwargs and kwargs['raise_all']:
                raise
            logger.info('Could not read ' + file_name)
            logger.debug(str(e))
            logger.debug(traceback.format_exc())

        finally:
            uh.close()
    except Exception as e:
        if 'raise_all' in kwargs and kwargs['raise_all']:
            raise
        logger.info('Could not open ' + file_name)
        logger.debug(str(e))

    return None


def _generic_save_converter(d, file_name, archive_data):
    if d.data.shape[0] != np.size(d.channels):
        raise ValueError('data shape incorrect for number of channels')      
    if d.data.shape[1] != np.size(d.sample_start_time):
        raise ValueError('data shape incorrect for number of samples')

    # Force sample start time to be units of nominal_cadence (or better)
    sst = d.sample_start_time + np.timedelta64(0, dt64.get_units(d.nominal_cadence))
    
    if archive_data['timestamp_method'] == 'unixtime':
        # Force to day or better resolution
        sst += np.timedelta64(0, 'D')
        col_offset = 1
        data = np.empty([col_offset + np.size(d.channels),
                         np.size(d.sample_start_time)])
        data[0] = sst.astype('float') * dt64.multipliers[dt64.get_units(sst)]

    elif archive_data['timestamp_method'] == 's':
        # Works for either numpy.datetime64 or numpy.timedelta64
        col_offset = 1
        data = np.empty([col_offset + np.size(d.channels),
                         np.size(d.sample_start_time)])
        data[0] = dt64.astype(d.sample_start_time, 's').astype('int')

    elif archive_data['timestamp_method'] == 'offset0':
        col_offset = 1
        data = np.empty([col_offset + np.size(d.channels),
                         np.size(d.sample_start_time)])
        data[0] = (sst - d.start_time) / d.nominal_cadence

    elif archive_data['timestamp_method'] == 'offset1':
        col_offset = 1
        data = np.empty([col_offset + np.size(d.channels),
                         np.size(d.sample_start_time)])
        data[0] = (sst - d.start_time) / d.nominal_cadence
        data[0] += 1

    elif archive_data['timestamp_method'] in ('Y', 'YM', 'YMD', 'YMDh', 'YMDhm', 'YMDhms'):
        col_offset = len(archive_data['timestamp_method'])
        data = np.empty([col_offset + np.size(d.channels),
                         np.size(d.sample_start_time)])
        # Force to sst to appropriate resolution
        sst += np.timedelta64(0, archive_data['timestamp_method'][-1])
        get_part = [dt64.get_year, dt64.get_month, dt64.get_day,
                    dt64.get_hour, dt64.get_minute, dt64.get_second]
        for n in range(col_offset):
            data[n] = get_part[n](sst)

    elif archive_data['timestamp_method'].startswith('YMDhms.'):
        tu = archive_data['timestamp_method'].replace('YMDhms.', '')
        mul = dt64.multipliers[tu]
        if mul > 1:
            raise ValueError('illegal time unit (%s)' % tu)

        col_offset = 6
        data = np.empty([col_offset + np.size(d.channels),
                         np.size(d.sample_start_time)])
        sst += np.timedelta64(0, 'tu')
        data[0] = dt64.get_year(sst)
        data[1] = dt64.get_month(sst)
        data[2] = dt64.get_day(sst)
        data[3] = dt64.get_hour(sst)
        data[4] = dt64.get_minute(sst)
        # Get just seconds and the fractional seconds
        data[5] = np.mod(sst.astype('int'), int(np.round(60 / mul))) * mul

    else:
        raise ValueError('unknown value for timestamp_method')

    data[col_offset:] = d.data

    if archive_data['data_multiplier']:
        data[col_offset:] = d.data * archive_data['data_multiplier']
    else:
        data[col_offset:] = d.data

    if 'delimiter' in archive_data:
        delimiter = archive_data['delimiter']
    elif os.path.splitext(file_name)[1].lower() == '.csv':
        delimiter = ','
    else:
        delimiter = '\t'
    with ap.tools.smart_open(file_name, 'w') as fh:
        np.savetxt(fh, data.T, delimiter=delimiter, fmt=archive_data['fmt'])


def first_non_nan(data_list):
    """
    Return the first non-NaN data from a list of data objects.
    :param data_list:
    :return Data:
    """
    d = copy.copy(data_list)
    r = copy.deepcopy(d.pop(0))
    while len(d):
        if not np.any(np.isnan(r.data)):
            break
        a = d.pop(0)
        nans = np.isnan(r.data)
        r.data[nans] = a.data[nans]
    return r


class Data(object):
    """
    Base class for time-series data.
    """

    def __init__(self, 
                 project=None,
                 site=None,
                 channels=None,
                 start_time=None,
                 end_time=None,
                 sample_start_time=np.array([], dtype='datetime64[s]'), 
                 sample_end_time=np.array([], dtype='datetime64[s]'), 
                 integration_interval=None, 
                 nominal_cadence=None,
                 data=None,
                 units=None,
                 sort=False):
        self.project = project
        self.site = site
        if isinstance(channels, six.string_types):
            self.channels = np.array([channels])
        elif channels is None:
            self.channels = np.array([]).reshape([0, 0])
        else:
            self.channels = np.array(channels)
        self.start_time = start_time
        self.end_time = end_time
        self.sample_start_time = np.reshape(sample_start_time,
                                            [np.size(sample_start_time)])
        self.sample_end_time = np.reshape(sample_end_time,
                                          [np.size(sample_end_time)])
        self.integration_interval = integration_interval
        self.nominal_cadence = nominal_cadence
        if data is None:
            self.data = np.tile(np.nan, [np.size(self.channels),
                                         np.size(self.sample_start_time)])
        else:
            self.data = data
            
        self.units = units
        if sort is None:
            sort = np.size(self.data) != 0
        if sort:
            self.sort(inplace=True)

    def __repr__(self):
        units = self.units
        if units == u'\N{DEGREE SIGN}C':
            units = 'degrees C'
        elif units == u'\N{DEGREE SIGN}':
            units = 'degrees'

        return (type(self).__name__ + ':\n' +
                '          project : ' + str(self.project) + '\n' +
                '             site : ' + str(self.site) + '\n' +
                '         channels : ' + str(self.channels) + '\n' +
                '       start_time : ' + str(self.start_time) + '\n' +
                '         end_time : ' + str(self.end_time) + '\n' +
                'sample_start_time : ' + repr(self.sample_start_time) + '\n' + 
                '  sample_end_time : ' + repr(self.sample_end_time) + '\n' + 
                'integration intv. : ' + repr(self.integration_interval) + '\n' +
                '   nominal cadence: ' + str(self.nominal_cadence) + '\n' +
                '             data : ' + repr(self.data) + '\n' + 
                '            units : ' + str(units))

    def __format__(self, fmt):
        """Custom format for Data auroraplot.data.Data objects.

        Time formatting:

            '+': Format the start_time. The characters after '+' are
                passed to auroraplot.dt64tools.strftime().
 
            'start_time', 'end_time': Output the start or end time
                conforming to ISO8601 standard
                ('%Y-%m-%dT%H:%M:%S+0000'). A custom format specifier
                may be specified similar to the string format
                function(), append a colon and then the format string
                to 'start_time' or 'end_time' (e.g.,
                'start_time:%H:%M:%S %d %B %Y').

        String or numerical formatting:

            'project_lc': Output the project abbreviation in lower
                case.

            'site_lc': Output the site abbreviation in lower case.

            'project:info': Output project information, info should be
                replaced with the name of the project information to
                be inserted, e.g., 'project:url'.

            'site:info': Output site information, info should be
                replaced with the name of the site information to be
                inserted, e.g., 'site:location'.
                
            Field names which return string or numerical values can be
            appended by an optional format specifier which is passed
            to the str.format() function, e.g., 'site:elevation:.1f'.

            The format string can be preceded by a conversion type,
            where 'a' is for ascii() (Python 3 only), 'r' for repr(),
            and 's' for str(). In addition the following non-standard
            conversions are also provided: 'c' for str().capitalize(),
            capitalize(), 'l' for str().lower(), 't' for str().title()
            and 'u' for str().upper().

            If a conversion type or format specifier is not provided
            the value is converted to a string by calling str().
        """

        if fmt.startswith('!'):
            conversion, _, fmt = fmt[1:].partition(':')
        else:
            conversion = None

        if fmt.startswith('+'):
            # Unix date command uses + to mark the format specifier
            r = dt64.strftime(self.start_time, fmt[1:])
            fmt2 = ''

        else:
            # Cases below are of form 'field' or 'field:fmt2'
            # Use str.format() to do any additional format conversion
            f, sep, fmt2 = fmt.partition(':') 

            if f in ('start_time', 'end_time'):
                # Custom formatting by strftime
                if fmt2:
                    r = dt64.strftime(getattr(self, f), fmt2)
                else:
                    r = dt64.strftime(getattr(self, f), 
                                      '%Y-%m-%dT%H:%M:%S+0000')

            elif f in ('project_lc', 'site_lc'):
                r = getattr(self, f.split('_')[0])
                r = 'none' if r is None else r.lower()
            elif f == 'description':
                r = self.data_description()
            elif f == 'date_range':
                r = dt64.fmt_dt64_range(self.start_time, self.end_time)
            elif f == 'site':
                if sep:
                    # Takes form 'site:info' or site:info:fmt2'
                    info, _, fmt2 = fmt2.partition(':')
                    r = self.get_site_info(info)
                else:
                    # Just site
                    r = self.site
            elif f == 'project':
                if sep:
                    # Takes form 'project:info' or project:info:fmt2'
                    info, _, fmt2 = fmt2.partition(':')
                    r = self.get_project_info(info)
                else:
                    r = self.project
            else:
                raise ValueError("Invalid format ('%s') for type '%s'" % 
                                 (fmt, self.__class__.__name__))

        if conversion is None:
            r = str(r)
        elif conversion == 'a':
            r = ascii(r)  # Python3 only
        elif conversion == 'c':
            r = str(r).capitalize()
        elif conversion == 'l':
            r = str(r).lower()
        elif conversion == 'r':
            r = repr(r)
        elif conversion == 's':
            r = str(r)
        elif conversion == 't':
            r = str(r).title()
        elif conversion == 'u':
            r = str(r).upper()
        else:
            raise ValueError("Unknown conversion ('%s') for type '%s'" %
                             (fmt, self.__class__.__name__))

        if fmt2:
            r = ('{0:%s}' % fmt2).format(r)        

        return r

    def data_description(self):
        return 'Data'

    def assert_valid(self):
        import re
        try:
            for n in ('project', 'site', 'channels', 'start_time', 'end_time', 
                      'sample_start_time', 'sample_end_time',
                      'nominal_cadence',
                      'data', 'units'):
                attr = getattr(self, n)
                assert (attr is not None and (not isinstance(attr, six.string_types) or attr != '')), \
                    n + ' not set'

            assert re.match('^[-A-Z0-9_]+$', self.project), 'Bad value for project'
            assert re.match('^[-A-Z0-9_]+$', self.site), 'Bad value for site'

            num_channels = len(self.channels)
            num_samples = len(self.sample_start_time)

            assert self.start_time <= self.end_time, 'start_time must not be after end_time'
            assert len(self.sample_end_time) == num_samples, 'lengths of sample_start_time and sample_end_time differ'

            if self.integration_interval is not None:
                assert self.integration_interval.shape == (num_channels, num_samples), \
                    'integration_interval incorrect shape'
                assert self.data.shape == (num_channels, num_samples), 'data incorrect shape'

            assert (dt64.get_units(self.start_time) == dt64.get_units(self.nominal_cadence)), \
                'start_time units do not match cadence'
            assert (dt64.get_units(self.end_time) == dt64.get_units(self.nominal_cadence)), \
                'end_time units do not match cadence'
            assert (dt64.get_units(self.sample_start_time) == dt64.get_units(self.nominal_cadence)), \
                'sample_start_time units do not match cadence'
            assert (dt64.get_units(self.sample_end_time) == dt64.get_units(self.nominal_cadence)), \
                'sample_end_time units do not match cadence'
        except AssertionError as e:
            logger.debug(str(self))
            logger.debug(traceback.format_exc())
            raise
        
        return True

    def get_channel_index(self, channels):
        """
        Find the location of the listed channels in the objects channel list.
        
        channels: list of channels to find.

        return list of integers corresponding to location in the
        object's channels attribute.
        """
        chan_tup = tuple(self.channels)
        cidx = []
        for c in np.array(channels).flatten():
            cidx.append(chan_tup.index(c))
        return cidx

    def get_site_info(self, info=None):
        return ap.get_site_info(self.project, self.site, info)

    def get_project_info(self, info=None):
        return ap.get_project_info(self.project, info)

    def get_mean_sample_time(self):
        return dt64.mean(self.sample_start_time, self.sample_end_time)

    def pickle(self, filename):
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def extract(self, start_time=None, end_time=None, channels=None, 
                inplace=False, 
                straddle_boundary=False, trim_straddles=False):
        if start_time is not None:
            start_time = dt64.astype(start_time, units=self.nominal_cadence)
        if end_time is not None:
            end_time = dt64.astype(end_time, units=self.nominal_cadence)

        if inplace:
            r = self
        else:
            r = copy.deepcopy(self)
        if channels is not None:
            chan_tup = tuple(self.channels)
            cidx = []
            for c in channels:
                cidx.append(chan_tup.index(c))
            r.channels = r.channels[cidx]
            r.data = r.data[cidx, :]
            if r.integration_interval is not None:
                r.integration_interval = r.integration_interval[cidx, :]
            
        if start_time is not None or end_time is not None:
            if start_time is None:
                start_time = self.start_time
            if end_time is None:
                end_time = self.end_time

            if straddle_boundary:
                tidx = (self.sample_end_time > start_time) & \
                    (self.sample_start_time < end_time)
            else:
                tidx = (self.sample_start_time >= start_time) & \
                    (self.sample_end_time <= end_time)

            r.start_time = start_time
            r.end_time = end_time
            r.sample_start_time = r.sample_start_time[tidx]
            r.sample_end_time = r.sample_end_time[tidx]
            if straddle_boundary and trim_straddles:
                idx = r.sample_start_time < start_time
                r.sample_start_time[idx] = start_time 
                r.sample_end_time[r.sample_end_time > end_time] \
                    = end_time

            r.data = r.data[:, tidx]
            if r.integration_interval is not None:
                r.integration_interval = r.integration_interval[:, tidx]

        # elif channels is None and not inplace:
        #     # data matrix has not been copied
        #     r.data = r.data.copy()
        return r

    def sort(self, inplace=False, keep='last'):
        idx = np.unique(self.sample_start_time, return_index=True)[1]
        if np.size(idx) != np.size(self.sample_start_time):
            if keep is None:
                pass
            elif keep in ('first', 'last'):
                # unique does not specify which repeated entry is kept
                # and it seems to vary according to input
                d = {}
                i = 0
                for s in self.sample_start_time:
                    if s not in d or keep == 'last':
                        d[s] = i
                    i += 1

                idx = []
                for s in sorted(d.keys()):
                    idx.append(d[s])
            else:
                raise ValueError('Unknown value for keep')

        if inplace:
            r = self
        else:
            r = copy.deepcopy(self)

        r.sample_start_time = r.sample_start_time[idx]
        r.sample_end_time = r.sample_end_time[idx]

        # integration_interval may be None, or an array
        if self.integration_interval is not None \
           and self.integration_interval.size:
            r.integration_interval = self.integration_interval[:, idx]

        r.data = r.data[:, idx]
        return r

    def split(self, interval):
        r = []
        t1 = self.start_time
        while t1 < self.end_time:
            t2 = t1 + interval
            r.append(self.extract(start_time=t1, end_time=t2))
            t1 = t2
        return r

    def format_project_site(self):
        return ap.format_project_site(self.project, self.site)

    def make_title(self, subtitle=None, start_time=None, end_time=None):
        s = [self.format_project_site()]
        if subtitle:
            s.append(subtitle)
        else:
            s.append(self.data_description())
            
        s.append(dt64.fmt_dt64_range(start_time or self.start_time, 
                                     end_time or self.end_time))
        return '\n'.join(s)

    def plot(self, channels=None, figure=None, axes=None, subplot=None,
             units_prefix=None, title=None, 
             # Our own options
             start_time=None, end_time=None, time_units=None, add_legend=None,
             **kwargs):

        def bracket_units(units):
            if units is None or units == '':
                return ''
            else:
                return ' (' + units + ')'

        if channels is None:
            channels = self.channels
        elif isinstance(channels, six.string_types):
            channels=[channels]
        else:
            try:
                iterator = iter(channels)
            except TypeError:
                channels = [channels]
        
        new_figure = False
        if axes is not None:
            axes2 = axes
            if not hasattr(axes2, '__iter__'):
                axes2 = [axes2]
            if len(axes2) == 1:
                axes2 *= len(channels)
            else:
                assert len(axes2) == len(channels), \
                    'axes and channels must be same length'
        elif figure is None:
            figure=plt.figure()
            new_figure = True
        else:
            if isinstance(figure, mpl.figure.Figure):
                plt.figure(figure.number)
            else:
                plt.figure(figure)

        if subplot is not None:
            subplot2 = copy.copy(subplot)
            if not hasattr(subplot2, '__iter__'):
                subplot2 = [subplot2]
            if len(subplot2) == 1:
                subplot2 *= len(channels)
            else:
                assert len(subplot2) == len(channels), \
                    'subplot and channels must be same length'

        if start_time is None:
            start_time = self.start_time
        if end_time is None:
            end_time = self.end_time
        if time_units is None:
            time_units = 'us'

        chan_tup = tuple(self.channels)

        # Label the Y axis with channel if possible
        need_legend = not(len(channels) == 1 or
                          (hasattr(axes, '__iter__') and len(axes) > 1) or
                          (hasattr(subplot, '__iter__') and len(subplot) > 1))
        if need_legend:
            # Precompute the units information (prefix, multiplier etc)
            # since it must be the same for all channels
            allcidx = []
            for c in channels:
                allcidx.append(chan_tup.index(c))
                cu = ap.str_units(np.nanmax(np.abs(self.data[allcidx])), 
                                  self.units, prefix=units_prefix,
                                  ascii=False, wantstr=False)

        first_axes = None
        r = []
        for n in range(len(channels)):
            if axes is not None:
                ax = plt.axes(axes2[n])
            elif subplot is not None:
                ax = plt.subplot(subplot2[n], sharex=first_axes)
            else:
                ax = plt.gca()
            ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useOffset=False))
            cidx = chan_tup.index(channels[n])
            u = ap.str_units(np.nanmax(np.abs(self.data[cidx])), self.units, 
                             prefix=units_prefix, ascii=False, wantstr=False)
            xdata = dt64.mean(self.sample_start_time, self.sample_end_time)
            if u['mul'] == 1:
                # Can avoid a copy
                ydata = self.data[cidx]
            else:
                ydata = self.data[cidx] / u['mul']

            if 'label' in kwargs:
                r.append(dt64.plot_dt64(xdata, ydata,
                                        x_time_units=time_units,
                                        **kwargs)[0])
            else:
                r.append(dt64.plot_dt64(xdata, ydata,
                                        x_time_units=time_units,
                                        label=channels[n], **kwargs)[0])

            dt64.xlim_dt64(xmin=start_time, xmax=end_time, ax=ax)
            
            if not need_legend:
                # Lines plotted on different axes
                plt.ylabel(self.channels[cidx] + bracket_units(u['fmtunit']))
            
            if n == 0:
                first_axes = ax
                
        if add_legend or (add_legend is None and need_legend):
            lh = plt.legend(self.channels[allcidx], loc='best', fancybox=True)
            lh.get_frame().set_alpha(0.6)
            # Label Y axis
            plt.ylabel(str(self.data_description()) + 
                       bracket_units(cu['fmtunit']))

        if new_figure:
            # Add title
            plt.axes(first_axes)
            if title is None:
                plt.title(self.make_title(start_time=start_time, 
                                          end_time=end_time))
            else:
                plt.title(title)
            # plt.subplots_adjust(top=0.85)

        return r

    def get_cadence(self):
        if len(self.sample_start_time) < 2:
            return None
        sstd = np.diff(self.sample_start_time)
        if not np.all(sstd[0] == sstd):
            return None
        if len(self.sample_end_time):
            # Check end times too if they exist
            setd = np.diff(self.sample_end_time)
            if not np.all(setd[0] == setd):
                return None
            if sstd[0] != setd[0]:
                return None
        
        return sstd[0]

    def set_cadence(self, cadence, ignore_nan=True,
                    offset_interval=None, inplace=False,
                    aggregate=None):
        tu = dt64.get_units(cadence)

        if offset_interval is None:
            offset_interval = np.timedelta64(0, tu)
        if aggregate is None:
            aggregate = scipy.average

        if cadence > self.nominal_cadence:
            sam_st = dt64.astype(np.arange(dt64.ceil(self.start_time, cadence) 
                                           + offset_interval, 
                                           self.end_time, 
                                           cadence),
                                 time_type=self.start_time,
                                 units=tu)
            sam_et = sam_st + cadence

            sample_time = dt64.mean(self.sample_start_time, 
                                    self.sample_end_time)
            d = np.ones([len(self.channels), len(sam_st)]) * ap.NaN
            if self.integration_interval is not None:
                assert not np.any(dt64.isnat(self.integration_interval)), \
                    'integration_interval must not contain NaT'
                integ_intv = np.zeros([len(self.channels), len(sam_st)], 
                                      dtype=self.integration_interval.dtype)
            else:
                integ_intv = None
                
            keep_integ_intv = True
            for sn in range(len(sam_st)):
                tidx = np.where(np.logical_and(sample_time >= sam_st[sn],
                                               sample_time <= sam_et[sn]))[0]
                for cn in range(len(self.channels)):
                    if ignore_nan:
                        notnanidx = np.where(np.logical_not(np.isnan(self.data[cn, tidx])))[0]
                        tidx2 = tidx[notnanidx]
                    else:
                        tidx2 = tidx
                    
                    if self.integration_interval is not None:
                        # Update integration_interval
                        if len(tidx2) != 0:
                            integ_intv[cn, sn] = np.sum(self.integration_interval[cn, tidx2])
                        
                    if len(tidx2) != 0:
                        # Update data. 
                        if self.integration_interval is None:
                            d[cn, sn] = aggregate(self.data[cn, tidx2])

                        else:
                            # Weight the mean according to
                            # integration_interval if possible.
                            weights = self.integration_interval[cn, tidx2]
                            weights[dt64.isnat(weights)] = 0
                            try:
                                d[cn, sn] = aggregate(self.data[cn, tidx2],
                                                     weights=weights)
                            except TypeError as e:
                                keep_integ_intv = False
                                d[cn,sn] = aggregate(self.data[cn, tidx2])

            if inplace:
                r = self
            else:
                r = copy.copy(self)
                for k in (set(self.__dict__.keys())
                          - set(['sample_start_time', 'sample_end_time', 
                                 'integration_interval', 'nominal_cadence',
                                 'data'])):
                    setattr(r, k, copy.deepcopy(getattr(self, k)))

            r.start_time = dt64.astype(r.start_time, units=tu)
            r.end_time = dt64.astype(r.end_time, units=tu)
            r.sample_start_time = sam_st
            r.sample_end_time = sam_et
            if keep_integ_intv:
                r.integration_interval = integ_intv
            else:
                r.integration_interval = None
            r.nominal_cadence = cadence.copy()
            r.data = d

        elif cadence < self.nominal_cadence:
            raise Exception('Interpolation to reduce cadence not implemented')
        else:
            if offset_interval != np.timedelta64(0, tu):
                raise ValueError('cannot set offset_interval for same cadence')
            if inplace:
                r = self
            else:
                r = copy.deepcopy(self)

            for k in ('start_time', 'end_time', 
                      'sample_start_time', 'sample_end_time',
                      'nominal_cadence'):
                setattr(r, k, dt64.astype(getattr(r, k), units=tu))
        
        r.assert_valid()
        return r

    # TODO: fix inplace option which does not work
    def mark_missing_data(self, cadence=None,  # inplace=False
                          start_time=None, end_time=None):
        trim = False
        if cadence is None:
            cadence = self.nominal_cadence
        if start_time is None:
            start_time = self.start_time
        elif start_time > self.start_time:
            trim = True
        if end_time is None:
            end_time = self.end_time
        elif end_time < self.end_time:
            trim = True
            
        if trim:
            r = self.extract(start_time=start_time, end_time=end_time)
        # elif inplace:
        #     r = self
        else:
            r = copy.deepcopy(self)

        num_channels = len(r.channels)
        sample_time = dt64.mean(r.sample_start_time, 
                                r.sample_end_time)
        idx = np.where(np.diff(sample_time) > cadence)[0]
        
        sort = False
        obj_list = [r]

        if len(idx):
            sort = True
            data = np.ones([num_channels, len(idx)]) * ap.NaN
            if r.integration_interval is None:
                ii = None
            else:
                ii = np.zeros([num_channels, len(idx)]).astype(r.integration_interval.dtype)
                
            missing = type(self)(project=r.project,
                                 site=r.site,
                                 channels=r.channels,
                                 start_time=r.start_time,
                                 end_time=r.end_time,
                                 sample_start_time=r.sample_end_time[idx],
                                 sample_end_time=r.sample_start_time[idx+1],
                                 integration_interval=ii,
                                 nominal_cadence=r.nominal_cadence,
                                 data=data,
                                 units=r.units,
                                 sort=False)
            obj_list.append(missing)

        if r.integration_interval is None:
            ii = None
        else:
            ii = np.zeros([num_channels, 1]).astype(r.integration_interval.dtype)

        # Check for missing data at start
        if np.size(r.sample_start_time):
            if r.sample_start_time[0] - start_time > cadence:
                # Insert at start of list to avoid requiring a sort
                obj_list.insert(0, type(self)(project=r.project,
                                              site=r.site,
                                              channels=r.channels,
                                              start_time=start_time,
                                              end_time=r.sample_start_time[0],
                                              sample_start_time=np.array([start_time]),
                                              sample_end_time=r.sample_start_time[:1],
                                              integration_interval=ii,
                                              nominal_cadence=r.nominal_cadence,
                                              data=np.ones([num_channels, 1]) * ap.NaN,
                                              units=r.units,
                                              sort=False))

            # Check for missing data at end
            if end_time - r.sample_end_time[-1] > cadence:
                obj_list.append(type(self)(project=r.project,
                                           site=r.site,
                                           channels=r.channels,
                                           start_time=r.sample_end_time[-1],
                                           end_time=end_time,
                                           sample_start_time=r.sample_end_time[-1:],
                                           sample_end_time=np.array([end_time]),
                                           integration_interval=ii,
                                           nominal_cadence=r.nominal_cadence,
                                           data=np.ones([num_channels, 1]) * ap.NaN,
                                           units=r.units,
                                           sort=False))
        else:
            # No data at all
            obj_list.append(type(self)(project=r.project,
                                       site=r.site,
                                       channels=r.channels,
                                       start_time=start_time,
                                       end_time=end_time,
                                       sample_start_time=np.array([start_time]),
                                       sample_end_time=np.array([end_time]),
                                       integration_interval=ii,
                                       nominal_cadence=r.nominal_cadence,
                                       data=np.ones([num_channels, 1]) * ap.NaN,
                                       units=r.units,
                                       sort=False))
    
        if len(obj_list) == 1:
            r = obj_list[0]
            if sort:
                r.sort(inplace=True)
        else:
            r = ap.concatenate(obj_list, sort=sort)
        # if inplace:
        #     self = r
        return r

    def interp(self, sample_start_time, sample_end_time, kind='linear'):
        # Find the time class to promote to
        tu = dt64.smallest_unit([dt64.get_units(self.sample_start_time),
                                 dt64.get_units(self.sample_end_time),
                                 dt64.get_units(sample_start_time),
                                 dt64.get_units(sample_end_time)])
        tc = dt64.get_time_type(sample_start_time) + '[' + tu + ']'

        r = copy.deepcopy(self)
        xi = dt64.mean(self.sample_start_time, self.sample_end_time).astype(tc).astype('int')
        r.sample_start_time = sample_start_time
        r.sample_end_time = sample_end_time
        xo = dt64.mean(sample_start_time, sample_end_time).astype(tc).astype('int')
        r.data = scipy.interpolate.interp1d(xi, r.data, kind=kind,
                                            bounds_error=False)(xo)
        r.integration_interval = None
        return r

    def space_regularly(self, cadence, start_time=None, end_time=None,
                        missing_cadence=None, kind='linear'):
        if start_time is None:
            start_time = self.start_time
        if end_time is None:
            end_time = self.end_time
        if missing_cadence is None:
            missing_cadence = 1.5 * self.nominal_cadence

        # Ensure start/end time units match with cadence. Remember
        # some datasets use timedelta64 not datetime64 (eg quiet day
        # curves)
        cad_units = dt64.get_units(cadence)
        tc = dt64.get_time_type(start_time)
        start_time = start_time.astype(tc + '[' + cad_units + ']')
        end_time = end_time.astype(tc + '[' + cad_units + ']')

        s = self.mark_missing_data(start_time=start_time, end_time=end_time,
                                   cadence=missing_cadence)
        sam_st = np.arange(start_time, end_time, cadence)
        sam_et = sam_st + cadence
        r = s.interp(sam_st, sam_et, kind='linear')
        r.nominal_cadence = cadence
        return r

    def least_squares_fit(self, ref, err_func=leastsq_error, 
                          inplace=False, full_output=False, plot_fit=False):
        """
        Fit a dataset to a reference dataset by applying an offset to
        find the least-squared error. Uses scipy.optimize.leastsq()

        ref: reference object, of instance Data.
        
        err_func: reference to function which computes the errors. See
            leastsq_error() and sign_error(().

        inplace: if True modify self, otherwise modify a copy.

        full_output: if True return optional outputs too. See
            scipy.optimize.leastsq()

        Returns self (or a copy if 'inplace' is False) with an
            adjustment applied for best fit.
        """

        # See http://www.tau.ac.il/~kineret/amit/scipy_tutorial/ for
        # helpful tutorial on using leastsq().
        import scipy.optimize

        for c in self.channels:
            assert c in ref.channels, 'Channel not in reference object'

        # err_func = self._leastsq_residuals
        
        if inplace:
            r = self
        else:
            r = copy.deepcopy(self)

        # Fit each channel separately
        errors = []
        fit_info = []
        for c in self.channels:
            channel = self.channels[0]
            p0 = [0.0]
            fi = scipy.optimize.leastsq(err_func, p0, 
                                        full_output=True,
                                        args=(self, ref, channel))
            errors.append(fi[0][0])
            fit_info.append(fi)
            r.data[self.get_channel_index(c)] -= fi[0]

        if plot_fit:
            lh = r.plot()
            ref.plot(axes=lh[0].axes)

        if full_output:
            return (r, errors, fit_info)
        else:
            return r

    def minimise_sign_error_fit(self, ref, inplace=False, full_output=False, 
                                plot_fit=False, **kwargs):
        
        # There ought to be a way to do this with
        # scipy.optimize. Port existing Matlab code by Steve Marple.
        for c in self.channels:
            assert c in ref.channels, 'Channel not in reference object'
        if inplace:
            r = self
        else:
            r = copy.deepcopy(self)

        # Fit each channel separately
        errors = []
        fit_info = []
        for c in self.channels:
            if plot_fit:
                # Plot original data before fitting in case of
                # inplace=True
                ref.plot(label='Reference')
                ax = plt.gca()
                self.plot(axes=ax, label='Data')

            r.data[r.get_channel_index(c)], err, fi = ap.tools.fit_data(self.data[self.get_channel_index(c)[0]],
                                                                        ref.data[ref.get_channel_index(c)[0]],
                                                                        err_func=ap.tools.minimise_sign_error,
                                                                        full_output=True,
                                                                        **kwargs)
            errors.append(err)
            fit_info.append(fi)

            if plot_fit:
                r.plot(axes=ax, label='Fitted data')
                lh = plt.legend(loc='best', fancybox=True)
                lh.get_frame().set_alpha(0.6)

        if full_output:
            return (r, errors, fit_info)
        else:
            return r

    def save(self,
             archive=None,
             path=None,
             merge=None,
             overwrite=True,
             save_converter=None):
        assert archive is not None or path is not None, 'archive or path must be defined'

        an, ai = ap.get_archive_info(self.project,
                                     self.site,
                                     self.__class__.__name__,
                                     archive=archive)
        if save_converter is None:
            save_converter = ai['save_converter']
            if save_converter is None:
                raise Exception('Cannot save, save_converter not defined')

        if merge is None:
            merge = path is None

        if path is None:
            # Save to a default location
            path = ai['path']
            duration = ai['duration']
            t = dt64.dt64_range(dt64.floor(self.start_time, duration),
                                dt64.ceil(self.end_time, duration),
                                duration)
        else:
            # Save to single file
            t = [self.start_time]
            duration = self.end_time - self.start_time


        for t1 in t:
            t2 = t1 + duration
            file_name = dt64.strftime(t1, path)
            d = self.extract(t1, t2)
            d.set_time_units(ai['nominal_cadence'], inplace=True)
            if merge:
                # Load existing data and merge before saving
                tmp = ap.load_data(self.project,
                                   self.site,
                                   self.__class__.__name__,
                                   t1,
                                   t2,
                                   archive=archive)
                if tmp is not None:
                    if overwrite:
                        d = ap.concatenate([tmp, d], sort=True)
                    else:
                        d = ap.concatenate([d, tmp], sort=True)

            elif not overwrite and os.path.exists(file_name):
                raise Exception('File already exists')

            dir_name = os.path.dirname(file_name)
            if not os.path.exists(dir_name):
                logger.debug('making directory %s', dir_name)
                os.makedirs(dir_name)
            logger.info('saving to %s',
                        file_name.name if isinstance(file_name, file)
                        else file_name)
            save_converter(d, file_name, ai)

    def set_time_units(self, units, inplace=False):
        """Set time units of start/end times and cadence.

        The units cannot be larger than any units in use
        """

        if inplace:
            r = self
        else:
            r = copy.deepcopy(self)
            
        attr_list = ['start_time', 'end_time', 'nominal_cadence',
                     'sample_start_time', 'sample_end_time']
        if r.integration_interval is not None:
            attr_list.append('integration_interval')
        for a in attr_list:
            t1 = getattr(r, a)
            t2 = dt64.astype(t1, units)
            if (t1 != t2).any():
                raise ValueError('cannot set units to %s, precision lost in %s',
                                 units, a)
            setattr(r, a, t2)
        return r

    def chauvenet_criterion(self, window, inplace=False, want_outliers=False):

        data = self.data.copy()
        samt = dt64.mean(self.sample_start_time, self.sample_end_time)
        r_outliers = np.zeros_like(data, dtype=bool)
        for win in np.nditer(window):
            outliers = np.zeros_like(data, dtype=bool)
            for t1 in dt64.dt64_range(self.start_time, self.end_time, win):
                t2 = t1 + win
                for n in range(data.shape[0]):
                    idx = np.logical_and(samt >= t1, samt < t2)
                    N = np.count_nonzero(np.isfinite(data[n, idx]))
                    if not N:
                        continue
                    mean = ap.nanmean(data[n, idx])
                    std = ap.nanstd(data[n, idx])
                    criterion = 1.0 / (2*N)
                    d = np.abs(data[n, idx] - mean) / std
                    d /= 2.0 ** 0.5
                    prob = scipy.special.erfc(d)
                    outliers[n, idx] = (prob < criterion)

            r_outliers = np.logical_or(r_outliers, outliers)
            
        data[r_outliers] = np.nan

        if want_outliers:
            return r_outliers
        if inplace:
            r = self
        else:
            r = copy.deepcopy(self)

        r.data = data
        return r

    def remove_spikes_chauvenet(self,
                                inplace=False,
                                want_outliers=False,
                                link_channels=True,
                                savgol_window=np.timedelta64(5, 'm'),
                                chauvenet_window=np.array([89,79]).astype('timedelta64[s]')):
        """Remove spikes.

        Cron jobs can cause spikes in the data from magnetometers in the
        BGS Schools magnetometer network (most likely by a drop in supply
        voltage). This function removes most of the spikes."""

        # Regularly space the data prior to Savitzky Golay smoothing
        self2 = self.space_regularly(self.nominal_cadence,
                                     missing_cadence=self.nominal_cadence * 4)

        # Smooth the regularly-spaced data
        window_length = 2 * int(savgol_window / (2 * self.nominal_cadence)) + 1
        for n in range(self2.data.shape[0]):
            self2.data[n] = ap.tools.savitzky_golay(self2.data[n], 
                                                    window_length, 3)

        # Interpolate the smoothed data back to the original sample times
        smoothed = self2.interp(self.sample_start_time, self.sample_end_time)

        # Calculate the difference between the original data and the
        # smoothed version. Hopefully it is normally-distributed and
        # suitable to use Chauvenet's criterion to remove outliers.
        residuals = copy.deepcopy(self)
        residuals.data -= smoothed.data

        outliers = residuals.chauvenet_criterion(window=chauvenet_window,
                                                 want_outliers=True)

        if link_channels:
            for n in range(1, outliers.shape[0]):
                outliers[0] = np.logical_or(outliers[0], outliers[n])
                outliers[1:] = outliers[0]

        if want_outliers:
            return outliers

        if inplace:
            r = self
        else:
            r = copy.deepcopy(self)

        # r.data[outliers] = np.nan
        good_data = np.logical_not(outliers[0])
        r.sample_start_time = r.sample_start_time[good_data]
        r.sample_end_time = r.sample_end_time[good_data]
        r.integration_interval = r.integration_interval[:, good_data]
        r.data = r.data[:,good_data]
        r = r.mark_missing_data(3*r.nominal_cadence)
        
        return r

    def sliding_window(self, func, window, start_time=None, end_time=None):
        if start_time is None:
            start_time = self.start_time
        if end_time is None:
            end_time = self.end_time
        w2 = window / 2
        r = copy.deepcopy(self)
        tidx = np.nonzero(np.logical_and(self.sample_start_time >= start_time, self.sample_end_time <= end_time))[0]
        mst = self.get_mean_sample_time()
        for i in tidx:
            idx2 = np.nonzero(np.logical_and(self.sample_start_time >= mst[i] - w2,
                                             self.sample_end_time <= mst[i] + w2))[0]
            for j in range(self.data.shape[0]):
                r.data[j][i] = func(self.data[j][idx2])
        return r
