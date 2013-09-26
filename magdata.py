import copy

import numpy as np
import numpy.fft
import matplotlib.pyplot as plt

import auroraplot as ap
from auroraplot.data import Data
import auroraplot.dt64tools as dt64
from scipy.stats import nanmean
import scipy.interpolate
import warnings

def load_qdc(network, site, time, **kwargs):
    '''Load quiet-day curve. 
    network: name of the network (upper case)

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
        
    verbose: flag to indicate if verbose messages should be
        printed. If None then the global verbose parameter is checked.

    '''
    
    data_type = 'MagQDC'
    archive, ad = ap.get_archive_details(network, site, data_type, **kwargs)
    channels = kwargs.get('channels')
    if channels:
        # Could be as single channel name or a list of channels
        if isinstance(channels, basestring):
            if channels not in ad['channels']:
                raise Exception('Unknown channel')
        else:
            for c in channels:
                if c not in ad['channels']:
                    raise Exception('Unknown channel')
    else:
        channels = ad['channels']

    verbose = kwargs.get('verbose', ap.verbose)
    path = kwargs.get('path', ad['path'])

    load_function = kwargs.get('load_function', ad.get('load_function'))
    tries = kwargs.get('tries', 1)

    kwargs2 = kwargs.copy()
    kwargs2['archive'] = archive
    kwargs2['channels'] = channels
    kwargs2['load_function'] = load_function
    kwargs2['path'] = path
    kwargs2['verbose'] = verbose
        
    if load_function:
        # Pass responsibility for loading to some other
        # function. Parameters have already been checked and verbose
        # is set to the value the user desires.
        return load_function(network, site, data_type, start_time, 
                             end_time, **kwargs2)

    data = []
    t = dt64.get_start_of_month(time)
    for n in range(tries):
        try:
            if hasattr(path, '__call__'):
                # Function: call it with relevant information to get the path
                file_name = path(t, network=network, site=site, 
                                 data_type=data_type, archive=archive,
                                 channels=channels)
            else:
                file_name = dt64.strftime(t, path)

            if verbose:
                print('loading ' + file_name)

            r = ad['converter'](file_name, 
                                ad,
                                network=network,
                                site=site, 
                                data_type=data_type, 
                                start_time=np.timedelta64(0, 'h'), 
                                end_time=np.timedelta64(24, 'h'),
                                **kwargs2)
              
            if r is not None:
                r.extract(inplace=True, 
                          channels=channels)
                return r
        finally:
            # Go to start of previous month
            t = dt64.get_start_of_month(t - np.timedelta64(24, 'h'))

    return None


def stack_plot(data_array, offset, channel=None, 
                start_time=None, end_time=None,
                sort=True,
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
    
    offset: The offset applied when displayed each data object. This i
        given in the data units (usually tesla).
       
    channel: The data channel to use. If None given defaults to the
        first channel in the first object of the data_array sequence.

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
        
    if channel is None:
        channel = da[0].channels[0]

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
    for n in range(da.size):
        st = np.min([st, da[n].start_time])
        et = np.max([et, da[n].end_time])
        cidx = da[n].get_channel_index(channel)
        d = da[n].mark_missing_data(cadence=da[n].nominal_cadence*2)
        y = (d.data[cidx] - scipy.stats.nanmedian(d.data[cidx], axis=-1) \
                 + (n * offset))
        lh = dt64.plot_dt64(d.get_mean_sample_time(), y.flatten())
        r.extend(lh)
        tick_labels.append(d.network + '\n' + d.site)


    if start_time is None:
        start_time = st
    if end_time is None:
        end_time = et
    ax.set_xlim(dt64.dt64_to(start_time, ax.xaxis.dt64tools.units),
                dt64.dt64_to(end_time, ax.xaxis.dt64tools.units))
    ax.yaxis.set_ticks(tick_locs)
    ax.yaxis.set_ticklabels(tick_labels)
    ax.set_title('\n'.join(['Magnetometer stackplot', 
                            dt64.fmt_dt64_range(start_time, end_time)]))
    plt.draw()

    return r

    
class MagData(Data):
    '''Class to manipulate and display magnetometer data.'''

    def __init__(self,
                 network=None,
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
                 sort=True):
        Data.__init__(self,
                      network=network,
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


    def savetxt(self, filename, verbose=None):
        if verbose is None:
            verbose = ap.verbose
        a = np.zeros([self.channels.size+1, self.data.shape[-1]], 
                     dtype='float')
        a[0] = dt64.dt64_to(self.sample_start_time, 'us') / 1e6
        if self.units == 'T':
            # Convert to nT
            a[1:] = self.data * 1e9
        else:
            warnings.warn('Unknown units')
            a[1:] = self.data 
        print('saving to ' + filename)
        np.savetxt(filename,  a.transpose())


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


    def plot_with_qdc(self, qdc, lsq_fit=False, **kwargs):
        self.plot(**kwargs)
        qdc.align(self, lsq_fit=lsq_fit).plot(axes=plt.gca(), **kwargs)


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
            qdc_data[not_nan] += qd[n].data[:,not_nan]
            count[not_nan] += 1
            
        qdc_data /= count
            
        qdc = MagQDC(network=self.network,
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
                 network=None,
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
                 sort=True):
        MagData.__init__(self,
                         network=network,
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
        r.data = np.abs(np.fft.ifft(coeffs))
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

    
    def align(self, ref, lsq_fit=False, full_output=False):
        day = np.timedelta64(24, 'h')
        if isinstance(ref, MagData):
            r = copy.deepcopy(ref)
            interp_times = ref.get_mean_sample_time()
        else:
            # Must be sample times
            assert lsq_fit == False, \
                'Cannot do least squares fit without reference data'
            ta = np.sort(np.array(ref).flatten())
            if len(ta) >= 2:
                # Guess the nominal cadence
                nc = np.median(np.diff(ta))
            else:
                nc = None
            r = MagData(network=self.network,
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

        # xo = dt64.get_time_of_day(ref.get_mean_sample_time()).astype('m8[us]')
        xo = dt64.get_time_of_day(interp_times).astype('m8[us]')
        r.data = scipy.interpolate.interp1d(xi.astype('int64'), yi)\
            (xo.astype('int64'))

        if lsq_fit:
            return r.least_squares_fit(ref, inplace=True, 
                                       full_output=full_output)
        
        return r
        
