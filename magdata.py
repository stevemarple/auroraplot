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

def load_qdc(network, site, start_time, end_time, verbose=None, **kwargs):
    if not networks.has_key(network):
        raise Exception('Unknown network')
    elif not networks[network].has_key(site):
        raise Exception('Unknown site')
    elif not networks[network][site]['data_types'].has_key(data_type):
        raise Exception('Unknown data_type')

    if kwargs.get('archive') is None:
        if len(networks[network][site]['data_types'][data_type]) == 1:
            # Only one archive, so default is implicit
            archive = networks[network][site]['data_types'][data_type].keys()[0]
        elif networks[network][site]['data_types'][data_type].has_key('default'):
            # Use explicit default
            archive = 'default'
        else:
            raise Exception('archive must be specified')
    else:
        archive = kwargs.get('archive')

    if not networks[network][site]['data_types'][data_type].has_key(archive):
        raise Exception('Unknown archive')

    # archive data
    ad = networks[network][site]['data_types'][data_type][archive] 
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

    if verbose is None:
        verbose = ap.verbose
    tries = kwargs.get('tries', 1)

    t = dt64.get_start_of_month(dt64.mean(start_time, end_time))
    for n in range(tries):
        try:
            r = load_data(network, site, 'MagQDC', t, t, verbose=verbose)
               
            if r is not None:
                return r
        finally:
            # Go to start of previous month
            t = dt64.get_start_of_month(t - np.timedelta64(1, 'D'))

    return None


    
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


    def savetxt(self, filename):
        a = np.zeros([self.channels.size+1, self.data.shape[-1]], 
                     dtype='float')
        a[0] = dt64.dt64_to(self.sample_start_time, 'us') / 1e6
        if self.units == 'T':
            # Convert to nT
            a[1:] = self.data * 1e9
        else:
            warnings.warn('Unknown units')
            a[1:] = self.data 
        np.savetxt(filename,  a.transpose())


    def plot(self, channels=None, figure=None, axes=None,
             subplot=None, units_prefix=None, subtitle=None, 
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
                      subtitle=subtitle, 
                      start_time=start_time, end_time=end_time, 
                      time_units=time_units, **kwargs)
        return r

    def plot_with_qdc(self, qdc, **kwargs):
        self.plot(**kwargs)
        qdc.align(self).plot(axes=plt.gca())

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

        day = np.timedelta64(1, 'D').astype('m8[us]')
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
                 quiet_days_method=None):
        qd = self.get_quiet_days(nquiet=nquiet, channels=channels,
                                 cadence=cadence, method=quiet_days_method)


        sam_st = np.arange(np.timedelta64(0, 's').astype('m8[us]'),
                           np.timedelta64(1, 'D').astype('m8[us]'),
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
                     start_time=self.start_time,
                     end_time=self.end_time,
                     sample_start_time=sam_st,
                     sample_end_time=sam_et,
                     integration_interval=None,
                     nominal_cadence=cadence,
                      data=qdc_data,
                     units=self.units,
                     sort=False)
        
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
             subplot=None, units_prefix=None, subtitle=None, 
             start_time=None, end_time=None, time_units=None, **kwargs):
        
        if start_time is None:
            start_time = np.timedelta64(0, 'us')
        if end_time is None:
            end_time = np.timedelta64(1, 'D').astype('m8[us]')

        r = MagData.plot(self, channels=channels, figure=figure, axes=axes,
                      subplot=subplot, units_prefix=units_prefix,
                      subtitle=subtitle, 
                         start_time=start_time, end_time=end_time, 
                         time_units=time_units, **kwargs)
        return r

    
    def align(self, md, lsq_fit=False):
        # assert isinstance(md, MagData), 'Incorrect data type'
        day = np.timedelta64(1, 'D').astype('m8[us]')
        r = copy.deepcopy(md)

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

        xo = dt64.get_time_of_day(md.get_mean_sample_time()).astype('m8[us]')
        r.data = scipy.interpolate.interp1d(xi.astype('int64'), yi)\
            (xo.astype('int64'))

        if lsq_fit:
            r.least_squares_fit(md, inplace=True)
        
        return r
        
