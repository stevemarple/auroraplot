import copy
import logging
import re

# Python 2/3 compatibility
import six
try:
    from urllib.request import urlopen
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse
    from urllib import urlopen

import numpy as np
import numpy.fft
import matplotlib.pyplot as plt

import auroraplot as ap
from auroraplot.data import Data
import auroraplot.dt64tools as dt64
import auroraplot.tools
from scipy.stats import nanmean
import scipy.interpolate
import warnings

logger = logging.getLogger(__name__)


def load_iaga_2000(file_name):
    try:
        if file_name.startswith('/'):
            uh = urlopen('file:' + file_name)
        else:
            uh = urlopen(file_name)

        try:
            lines = [a.rstrip().decode('ascii') for a in uh.readlines()]
            header = {}
            comments = []
            
            # Parse header
            while lines[0][0] == ' ':
                s = lines.pop(0)
                if s[1] == '#':
                    # Comment
                    c = s[2:-1].strip()
                    if c != '':
                        comments.append(c)
                else:
                    # Header line
                    key = s[:24].strip()
                    value = s[24:-1].strip()
                    header[key] = value
            
            # Parse data header and store column positions. 
            s = lines.pop(0)
            cols = s[:-1].split()
            col_pos = [[0, 0], ]
            last_pos = len(s)
            data = [[]]
            col_number = {cols[0]: 0}
            for n in range(1, len(cols)):
                pos = s.find(cols[n])
                col_pos.insert(n, [pos, 0])
                col_pos[n-1][1] = pos
                col_number[cols[n]] = n
                data.insert(n, [])
            col_pos[-1][1] = len(s)
            
            # Parse data. The most flexible and tolerant method is to
            # split the line into data columns using whitespace. This
            # approach allows files to be read where the data headers
            # don't exactly align with the data columns (as observed
            # with University of Tromso Tristan da Cunha
            # data). However if the columns are so wide that no
            # whitespace occurs this approach will fail so fall back
            # to using the data header positions to determine where
            # each data column should start.
            for s in lines:
                # Remove any trailing '|'
                s = s.strip('|')

                # Can the line be split correctly based on whitespace?
                data_cols = s.split()
                if len(data_cols) == len(cols):
                    # Yes, number of data columns matches
                    for n in range(len(cols)):
                        data[n].append(data_cols[n])
                else:
                    # No, number of data columns differs so use
                    # position to determine data
                    for n in range(len(cols)):
                        data[n].append(s[col_pos[n][0]:col_pos[n][1]].strip())

            # Convert date and time
            if 'DATE' in col_number and 'TIME' in col_number:
                dn = col_number['DATE']
                tn = col_number['TIME']
                sample_time = []
                for n in range(len(data[0])):
                    sample_time.append(np.datetime64(data[dn][n] + 'T' +
                                                     data[tn][n] + 'Z')\
                                           .astype('<M8[us]'))
                sample_time = np.array(sample_time)
            else:
                sample_time = None

            r = { 'header': header,
                  'comments': comments,
                  'data': data,
                  'columns': cols,
                  'column_number': col_number,
                  'sample_time': sample_time
                  }
            return r

        except Exception as e:
            logger.info('Could not read ' + file_name)
            logger.debug(str(e))

        finally:
            uh.close()
                
    except Exception as e:
        logger.info('Could not open ' + file_name)
        logger.debug(str(e))
    return None


def load_qdc(project, 
             site, 
             time, 
             archive=None, 
             channels=None,
             path=None,
             tries=1,
             realtime=False,
             load_function=None,
             full_output=False):
    '''Load quiet-day curve. 
    project: name of the project (upper case)

    site: site abbreviation (upper case)

    time: a time within the quiet-day curve period
    
    The following optional parameters are recognised: 
    
    archive: name of the archive. Required if more than one archive is
        present and there is not an archive called "default".

    channels: data channel(s) to load. All are loaded if not specified.

    tries: The number of attempts to load a quiet-day curve. If >1 and
        the first attempt is not successful then an attempt will be
        made to load the previous QDC.
 
    path: URL or file path, specified as a strftime format specifier.
        Alternatively can be a function reference which is passed the
        time and returns the filename. If given this overrides the
        standard load path.

    load_function: Pass responsibility for loading the data to the given
        function reference, after validating the input parameters.
        
    '''

    data_type = 'MagQDC'
    archive, ad = ap.get_archive_info(project, site, data_type, 
                                      archive=archive)
    if channels is not None:
        # Ensure it is a 1D numpy array
        channels = np.array(channels).flatten()
        for c in channels:
            if c not in ad['channels']:
                raise ValueError('Unknown channel (%s)' % str(c))
    else:
        channels = ad['channels']

    if path is None:
        path = ad['path']

    if load_function is None:
        load_function = ad.get('load_function', None)

    if tries is None:
        tries = 1
       
    if load_function:
        # Pass responsibility for loading to some other
        # function. Parameters have already been checked.
        return load_function(project, 
                             site, 
                             data_type, 
                             time, 
                             archive=archive,
                             channels=channels,
                             path=path,
                             tries=tries,
                             realtime=realtime,
                             full_output=full_output)
    data = []

    t = dt64.get_start_of_month(time)

    if realtime:
        # For realtime use the QDC for the month is (was) not
        # available, so use the previous month's QDC
        t = dt64.get_start_of_previous_month(t)
        
        # Early in the month the previous motnh's QDC was probably not
        # computed, so use the month before that
        qdc_rollover_day = ad.get('qdc_rollover_day', 4)
        if dt64.get_day_of_month(time) < qdc_rollover_day:
            t = dt64.get_start_of_previous_month(t)

    for n in range(tries):
        try:
            if hasattr(path, '__call__'):
                # Function: call it with relevant information to get the path
                file_name = path(t, project=project, site=site, 
                                 data_type=data_type, archive=archive,
                                 channels=channels)
            else:
                file_name = dt64.strftime(t, path)

            logger.info('loading ' + file_name)

            r = ad['load_converter'](file_name, 
                                     ad,
                                     project=project,
                                     site=site, 
                                     data_type=data_type, 
                                     start_time=np.timedelta64(0, 'h'), 
                                     end_time=np.timedelta64(24, 'h'),
                                     archive=archive,
                                     channels=channels,
                                     path=path)
            if r is not None:
                r.extract(inplace=True, 
                          channels=channels)
                if full_output:
                    r2 = {'magqdc': r,
                          'tries': n + 1,
                          'maxtries': tries}
                    return r2
                else:
                    return r
                
        finally:
            # Go to start of previous month
            t = dt64.get_start_of_month(t - np.timedelta64(24, 'h'))

    return None


def load_qdc_data(file_name, archive_data, 
                  project, site, data_type, channels, start_time, 
                  end_time, **kwargs):
    '''Convert AuroraWatchNet QDC file to match standard data type.

    This function can be used by datasets as the load_converter for
    MagQDC data. It is not intended to be called directly by user
    programs, use load_qdc instead.

    archive: name of archive from which data was loaded
    archive_info: archive metadata
    '''

    assert data_type == 'MagQDC', 'Illegal data_type'
    chan_tup = tuple(archive_data['channels'])
    col_idx = []
    for c in channels:
        col_idx.append(chan_tup.index(c) + 1)
    try:
        if file_name.startswith('/'):
            uh = urlopen('file:' + file_name)
        else:
            uh = urlopen(file_name)
        try:
            data = np.loadtxt(uh, unpack=True)
            sample_start_time = (np.timedelta64(1000000, 'us') * data[0])
            sample_end_time = sample_start_time \
                + archive_data['nominal_cadence']
            
            integration_interval = None
            data = data[col_idx] * 1e-9 # Stored as nT
            r = MagQDC(project=project,
                       site=site,
                       channels=np.array(channels),
                       start_time=start_time,
                       end_time=end_time,
                       sample_start_time=sample_start_time, 
                       sample_end_time=sample_end_time,
                       integration_interval=integration_interval,
                       nominal_cadence=archive_data['nominal_cadence'],
                       data=data,
                       units=archive_data['units'],
                       sort=False)
            return r

        except Exception as e:
            logger.info('Could not read ' + file_name)
            logger.debug(str(e))

        finally:
            uh.close()
    except Exception as e:
        logger.info('Could not open ' + file_name)
        logger.debug(str(e))
    return None


def _save_baseline_data(md, file_name, archive_data):
    assert isinstance(md, MagData), 'Data is wrong type'
    assert md.units == 'T', 'Data units incorrect'
    assert md.data.shape[0] == np.size(md.channels), \
        'data shape incorrect for number of channels'
    assert md.data.shape[1] == np.size(md.sample_start_time), \
        'data shape incorrect for number of samples'
    data = np.empty([1 + np.size(md.channels), np.size(md.sample_start_time)])
    data[0] = (md.sample_end_time - md.start_time) / md.nominal_cadence
    data[1:] = md.data * 1e9
    fmt = ['%d']
    fmt.extend(['%.2f'] * np.size(md.channels))
    np.savetxt(file_name, data.T, delimiter='\t', fmt=fmt)


def stack_plot(data_array, offset, channel=None, 
               start_time=None, end_time=None,
               sort=True, scale_bar=True,
                **kwargs):
    '''
    Plot multiple MagData objects on a single axes. Magnetometer
    stackplots have a distinct meaning and should not be confused with
    the matplotlib.stackplot which implements a different type of
    stacked plot.

    The variation in each MagData object (measured from the median
    value) is displayed. Each object displayed offset from the
    previous. If "sort" is True then plots are ordered North (top) to
    South (bottom).
    
    data_array: A sequence of MagData objects. All objects must use
        the same data units.
    
    offset: The distance between each trace, given in the data units
        (usually tesla).
       
    channel: The data channel to use. If None given defaults to the
        first channel in the first object of the data_array sequence.
        If channel is a string then plot only the selected channel,
        ignore MagData objects which do not contain the channel. If
        channel is a list then the first matching channel from each
        MagData object is plotted; if the object does not contain any
        of the channels nothing is plotted.

    start_time: The start time for the x axis. If None then the
        earliest start time from all datasets.
 
    end_time: The end time for the x axis. If None then the latest end
        time from all datasets.

    sort: Flag to indicate if datasets should be sorted by geographic
    latitude before plotting.
    '''

    da = np.array(data_array).flatten()
    
    for n in range(1, da.size):
        assert da[0].units == da[n].units, 'units differ'
        
    if channel is None or len(channel) == 0:
        channel = da[0].channels[0]
    elif not isinstance(channel, six.string_types) and len(channel) == 1:
        # Treat lists of single item as string
        channel = channel[0]

    if sort:
        latitude = np.zeros_like(da, dtype=float)
        for n in range(da.size):
            latitude[n] = da[n].get_site_info('latitude')
        sort_idx = np.argsort(latitude)
        da = da[sort_idx]

    r = []
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.hold(True)
    tick_locs = np.arange(da.size) * offset
    tick_labels = []
    st = da[0].start_time
    et = da[0].end_time

    # To estimate range of data, ignoring a few extreme cases.
    ydiff = np.zeros(da.size)
    max5 = ap.tools.NthLargest(5)
    min5 = ap.tools.NthLargest(5, smallest=True)

    plot_count = 0

    # Plot each data set
    for n in range(da.size):        
        label = da[n].format_project_site()

        if isinstance(channel, six.string_types):
            cidx = da[n].get_channel_index(channel)
        else:
            cidx = None
            for c in channel:
                if c in da[n].channels:
                   cidx = da[n].get_channel_index(c)
                   label += ' (%s)' % c
                   break
            if cidx is None:
                # None found
                continue

        
        # Keep track of earliest latest times, but only for plotted
        # datasets
        st = np.min([st, da[n].start_time])
        et = np.max([et, da[n].end_time])

        d = da[n].mark_missing_data(cadence=da[n].nominal_cadence*2)
        y = (d.data[cidx] - scipy.stats.nanmedian(d.data[cidx], axis=-1) \
                 + (plot_count * offset)).flatten()
        ydiff[n] = max5(y) - min5(y)

        kwargs = {}
        try:
            kwargs['color'] = da[n].get_site_info('line_color')
        except KeyError as e:
            pass
        lh = dt64.plot_dt64(d.get_mean_sample_time(), 
                            y,
                            label=d.format_project_site(),
                            **kwargs)
        plot_count += 1
        r.extend(lh)
        # tick_labels.append(d.project + '\n' + d.site)
        tick_labels.append(label.replace('/', '\n'))


    if start_time is not None:
        st = start_time
    if end_time is not None:
        et = end_time
    xlim = (dt64.dt64_to(st, ax.xaxis.dt64tools.units),
            dt64.dt64_to(et, ax.xaxis.dt64tools.units))
    ax.set_xlim(xlim)

    # Calculate some reasonable ylimits. Should show large activity
    # seen by most sites but exclude cases when the one site has a
    # major disturbance.  Use median disturbance as estimate of what
    # to realistically expect.
    if ydiff.size == 2:
        ydisturb = np.nanmin(ydiff) # Median would include any disturbance!
    else:
        ydisturb = scipy.stats.nanmedian(ydiff)

    if offset:
        # Round up to multiple of offset for aesthetic reasons.
        ydisturb = np.ceil(ydisturb / offset) * offset

        ylim = [-offset, da.size * offset]
        if ax.yaxis.get_data_interval()[0] < ylim[0]:
            ylim[0] = -ydisturb
        if ax.yaxis.get_data_interval()[1] > ylim[1]:
            ylim[1] = ((da.size - 1) * offset) + ydisturb
        ax.set_ylim(ylim)
    
    ax.yaxis.set_ticks(tick_locs)
    ax.yaxis.set_ticklabels(tick_labels)
    ax.set_title('\n'.join(['Magnetometer stackplot', 
                            dt64.fmt_dt64_range(st, et)]))

    if scale_bar:
        if offset:
            scale_bar = offset
        else:
            # Need method to compute the length of the bar
            scale_bar = False
            
    if scale_bar:
        draw_scale_marker(ax, offset/2, offset, 
                          text=ap.str_units(offset, da[0].units, 'n'))

    return r


def draw_scale_marker(ax, y, length, 
                      color='black', 
                      linewidth=2, 
                      text=None):
    xrel = 0.1
    ends_len_2 = 0.01
    xlim = ax.get_xlim()
    x = np.array((np.diff(xlim)* xrel + xlim[0]).repeat(2))
    x2 = x + np.diff(xlim) * np.array([-ends_len_2, ends_len_2])
    y2 = np.array([y-length/2, y+length/2])
    r = []
    ax.plot(x, y2, color=color, linewidth=linewidth) # Main line
    ax.plot(x2, y2[[0,0]], color=color, linewidth=linewidth) # Bottom bar
    ax.plot(x2, y2[[1,1]], color=color, linewidth=linewidth) # Top bar
    if text:
        ax.text(x2[1], y, text, color=color,
                horizontalalignment='left',
                verticalalignment='center')

    return r

class MagData(Data):
    '''Class to manipulate and display magnetometer data.'''

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
                 sort=None):
        Data.__init__(self,
                      project=project,
                      site=site,
                      channels=channels,
                      start_time=start_time,
                      end_time=end_time,
                      sample_start_time=sample_start_time,
                      sample_end_time=sample_end_time,
                      integration_interval=integration_interval,
                      nominal_cadence=nominal_cadence,
                      data=data,
                      units=units,
                      sort=sort)

    def data_description(self):
        return 'Magnetic field'


    def savetxt(self, filename, fmt=None):
        a = np.zeros([self.channels.size+1, self.data.shape[-1]],
                     dtype='float')
        a[0] = dt64.dt64_to(self.sample_start_time, 'us') / 1e6
        if self.units == 'T':
            # Convert to nT
            a[1:] = self.data * 1e9
        else:
            warnings.warn('Unknown units')
            a[1:] = self.data 
        logger.info('saving to ' + filename)
        kwargs = {}
        if fmt is not None:
            kwargs['fmt'] = fmt
        np.savetxt(filename,  a.transpose(), **kwargs)


    def plot(self, channels=None, figure=None, axes=None,
             subplot=None, units_prefix=None, title=None, 
             start_time=None, end_time=None, time_units=None, **kwargs):
        if channels is None:
            channels = self.channels

        if axes is not None:
            subplot2 = None
        elif subplot is None:
            subplot2 = []
            for n in range(1, len(channels) + 1):
                subplot2.append(len(channels)*100 + 10 + n)
        else:
            subplot2=subplot

        if units_prefix is None and self.units == 'T':
            # Use nT for plotting
            units_prefix = 'n'
        
        r = Data.plot(self, channels=channels, figure=figure, axes=axes,
                      subplot=subplot2, units_prefix=units_prefix,
                      title=title, 
                      start_time=start_time, end_time=end_time, 
                      time_units=time_units, **kwargs)
        return r


    def plot_with_qdc(self, qdc, fit_err_func=None, **kwargs):
        self.plot(**kwargs)
        if qdc is not None:
            qdc.align(self, fit_err_func=fit_err_func).plot(axes=plt.gca(), 
                                                            **kwargs)


    def variometer_plot(self, channels=None, qdc=None, axes=None, **kwargs):
        if channels is None:
            channels = self.channels
        
        colors = {'H': 'b', 'X': 'b',
                  'D': 'r', 'E': 'r', 'Y': 'r',
                  'Z': 'g'}
        kwargs['axes'] = axes
        md = copy.deepcopy(self)
        qdc2 = copy.deepcopy(qdc)
        for chan in channels:
            chan_num = list(md.channels).index(chan)
            adj = -np.median(md.data[chan_num])
            md.data[chan_num] += adj
            if chan.upper() in colors:
                kwargs['color'] = colors[chan.upper()]
            kwargs['alpha'] = 1
            kwargs['zorder'] = -chan_num
            md.plot(channels=chan, **kwargs)
            kwargs['axes'] = plt.gca()
         

            if qdc2 is not None:
                qdc2.data[list(qdc2.channels).index(chan)] += adj
                kwargs['alpha'] = 0.4
                kwargs['zorder'] -= 0.5
                qdc2.align(md).plot(channels=chan, label=chan + ' QDC', 
                                    **kwargs)
            
        leg = plt.legend(prop={'size': 'medium'})
        leg.get_frame().set_alpha(0.7)


    def get_quiet_days(self, nquiet=5, channels=None, 
                       cadence=np.timedelta64(5, 's').astype('m8[us]'),
                       method=None):
        '''
        nquiet: number of quiet days
        
        channels: channels used in calculations. Defaults to first
        channel only
        
        cadence: cadence used for calculation, and of the returned data
                
        returns: data from nquiet quietest days

        Adapted from algorithm originally developed by Andrew Senior.
        '''
        
        if channels is None:
            # Default to using H or X (ie first channel)
            cidx = [0]
            channels = self.channels[0]
        else:
            cidx = self.get_channel_index(channels)

        if method is None:
            method = 'monthly_mean'

        day = np.timedelta64(24, 'h')
        st = dt64.floor(self.start_time, day)
        et = dt64.ceil(self.end_time, day)
        s = self.space_regularly(cadence, start_time=st, end_time=et,
                                 missing_cadence=self.nominal_cadence * 2)

        num_days = int(np.round((et - st) / day))
        daily_data = s.split(day)
        daily_act = np.zeros(num_days)

        # Compute monthly mean for each selected channel
        monthly_means = nanmean(s.data[cidx], axis=1)

        if method == 'monthly_mean':
            for n in range(num_days):
                # Estimate daily activity based on RMS departure from
                # monthly mean
                daily_act[n] = \
                    nanmean(np.sqrt(nanmean((daily_data[n].data[cidx] \
                                                 .transpose() - 
                                             monthly_means)**2, axis=1)))

        elif method == 'daily_mean':
            for n in range(num_days):
                # Estimate daily activity based on RMS departure from
                # daily mean
                daily_means = nanmean(daily_data[n].data[cidx], axis=1)
                daily_act[n] = \
                    nanmean(np.sqrt(nanmean((daily_data[n].data[cidx] \
                                                 .transpose() - \
                                                 daily_means)**2, axis=1)))
            
                # Shift the data by the difference between the monthly
                # and daily means
                daily_data[n].data += (monthly_means - daily_means)
            
        elif method == 'linear_fit':
            x = self.get_mean_sample_time().astype('m8[us]').astype('int64')
            fits = []
            for cn in range(len(cidx)):
                fits.append(np.polyfit(x, self.data[cidx[cn]], 1))
                
            for n in range(num_days):
                # Estimate daily activity based on RMS departure from
                # linear fit to dataset
                daily_x = daily_data[n].get_mean_sample_time() \
                    .astype('m8[us]').astype('int64')
                tmp_act = np.zeros([1, len(cidx)])
                for cn in range(len(cidx)):
                    daily_y = fits[cn][0]*daily_x + fits[cn][1]
                    tmp_act[cn] = nanmean((daily_data[n].data[cidx[cn]]\
                                               .transpose() - daily_y)**2)
                    
                    # Shift the data by the difference between the
                    # monthly mean and the fit.
                    daily_data[n].data[cidx[cn]] += \
                        (monthly_means[cn] - daily_y)


                daily_act[n] = nanmean(np.sqrt(nanmean(tmp_act)))



        else:
            raise Exception('Unknown method')

        # Don't use days where more than 25% of data is missing
        for n in range(num_days):
            if np.mean(np.isnan(daily_data[n].data[cidx])) > 0.25:
                daily_act[n] = np.inf

        # Sort into ascending order of activity. Nans are put last.
        idx = np.argsort(daily_act)
        r = []
        for n in range(nquiet):
            r.append(daily_data[idx[n]])
        return r
        
    
    def make_qdc(self, nquiet=5, channels=None, 
                 cadence=np.timedelta64(5, 's').astype('m8[us]'),
                 quiet_days_method=None, smooth=True):
        qd = self.get_quiet_days(nquiet=nquiet, channels=channels,
                                 cadence=cadence, method=quiet_days_method)

        sam_st = np.arange(np.timedelta64(0, 's').astype('m8[us]'),
                           np.timedelta64(24, 'h').astype('m8[us]'),
                           cadence)
        sam_et = sam_st + cadence


        qdc_data = np.zeros([len(qd[0].channels), len(sam_st)])
        count = np.zeros_like(qdc_data)
        for n in range(nquiet):
            not_nan = np.logical_not(np.isnan(qd[n].data))
            qdc_data[not_nan] += qd[n].data[not_nan]
            count[not_nan] += 1

        qdc_data /= count
            
        qdc = MagQDC(project=self.project,
                     site=self.site,
                     channels=qd[0].channels,
                     start_time=np.timedelta64(0, 'h'),
                     end_time=np.timedelta64(24, 'h'),
                     sample_start_time=sam_st,
                     sample_end_time=sam_et,
                     integration_interval=None,
                     nominal_cadence=cadence,
                      data=qdc_data,
                     units=self.units,
                     sort=False)
        
        if smooth:
            qdc.smooth(inplace=True)

        return qdc

class MagQDC(MagData):
    '''Class to load and manipulate magnetometer quiet-day curves (QDC).'''
    def __init__(self,
                 project=None,
                 site=None,
                 channels=None,
                 start_time=None,
                 end_time=None,
                 sample_start_time=np.array([]),
                 sample_end_time=np.array([]),
                 integration_interval=np.array([]),
                 nominal_cadence=None,
                 data=np.array([]),
                 units=None,
                 sort=None):
        MagData.__init__(self,
                         project=project,
                         site=site,
                         channels=channels,
                         start_time=start_time,
                         end_time=end_time,
                         sample_start_time=sample_start_time,
                         sample_end_time=sample_end_time,
                         integration_interval=integration_interval,
                         nominal_cadence=nominal_cadence,
                         data=data,
                         units=units,
                         sort=sort)

    def data_description(self):
        return 'Magnetic field QDC'


    def smooth(self, fit_order=5, inplace=False):
        if inplace:
            r = self
        else:
            r = copy.deepcopy(self)
            
        coeffs = np.fft.fft(r.data)
        coeffs[:, range(fit_order+1, coeffs.shape[1]-fit_order)] = 0
        r.data = np.fft.ifft(coeffs).real
        return r


    def plot(self, channels=None, figure=None, axes=None,
             subplot=None, units_prefix=None, title=None, 
             start_time=None, end_time=None, time_units=None, **kwargs):
        
        if start_time is None:
            start_time = self.start_time
        if end_time is None:
            end_time = self.end_time

        r = MagData.plot(self, channels=channels, figure=figure, axes=axes,
                      subplot=subplot, units_prefix=units_prefix,
                      title=title, 
                         start_time=start_time, end_time=end_time, 
                         time_units=time_units, **kwargs)
        return r

    
    def align(self, ref, fit=None, **fit_kwargs):
        day = np.timedelta64(24, 'h')
        if isinstance(ref, MagData):
            r = copy.deepcopy(ref)
            interp_times = ref.get_mean_sample_time()
        else:
            # ref parameter must contain sample timestamps
            assert not fit, 'Cannot fit without reference data'
            ta = np.sort(np.array(ref).flatten())
            if len(ta) >= 2:
                # Guess the nominal cadence
                nc = np.median(np.diff(ta))
            else:
                nc = None
            r = MagData(project=self.project,
                        site=self.site,
                        channels=copy.copy(self.channels),
                        start_time=ta[0],
                        end_time=ta[-1],
                        sample_start_time=ta,
                        sample_end_time=copy.copy(ta),
                        integration_interval=None,
                        nominal_cadence=nc,
                        data=None,
                        units=self.units,
                        sort=False)
            interp_times = ta

        # Create array with room for additional entries at start and end
        xi = np.zeros([len(self.sample_start_time) + 2], dtype='m8[us]')
        xi[1:-1] = self.get_mean_sample_time().astype('m8[us]')
        yi = np.zeros([len(self.channels), self.data.shape[1] + 2],
                      dtype=self.data.dtype)
        yi[:,1:-1] = self.data
        # Extend so that interpolation can happen correctly near midnight
        xi[0] = xi[-2] - day
        xi[-1] = xi[1] + day
        yi[:, 0] = yi[:, -2]
        yi[:, -1] = yi[:, 1]

        xo = dt64.get_time_of_day(interp_times).astype('m8[us]')
        r.data = scipy.interpolate.interp1d(xi.astype('int64'), yi)\
            (xo.astype('int64'))

        if fit:
            if hasattr(fit, '__call__'):
                return fit(r, ref, inplace=True, **fit_kwargs)
            elif fit in ('baseline', 'realtime_baseline'):
                # QDC must be zero-mean for baseline adjustment,
                # correct any offset
                for n in range(r.data.shape[0]):
                    r.data[n] -= nanmean(self.data[n])
                for d in np.unique(dt64.get_date(r.sample_start_time)):
                    bl = ap.load_data(r.project, r.site, 'MagData',
                                      d, d + np.timedelta64(1, 'D'),
                                      channels=r.channels,
                                      archive=fit)
                    if bl is None or bl.data.size == 0:
                        raise Exception('No %s MagData for %s' 
                                        % (fit, str(d)))
                    r.data[:, d == dt64.get_date(r.sample_start_time)] \
                        += bl.data
            else:
                raise ValueError('Unknown fit method %s' % repr(fit))
        return r
        
