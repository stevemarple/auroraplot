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
import scipy.interpolate
import warnings

logger = logging.getLogger(__name__)


def load_qdc(project, 
             site, 
             time, 
             archive=None, 
             channels=None,
             path=None,
             tries=1,
             realtime=False,
             load_function=None,
             full_output=False,
             duration=None):
    '''Load quiet-day curve. 
    project: name of the project (upper case)

    site: site abbreviation (upper case)

    time: a time within the quiet-day curve period
   
    duration: the length of the quiet-day curve period 
              (default 14 days, unless specified in archive)
 
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

    data_type = 'RioQDC'
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

    if duration is None:
        if 'duration' in ad.keys():
            duration = ad['duration']
        else:
            duration = np.timedelta64(14,'D')

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

    t = dt64.floor(time, duration)

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
                    r2 = {'rioqdc': r,
                          'tries': n + 1,
                          'maxtries': tries}
                    return r2
                else:
                    return r
                
        finally:
            # Go to start of previous month
            t = t - duration

    return None


def load_qdc_data(file_name, archive_data, 
                  project, site, data_type, channels, start_time, 
                  end_time, **kwargs):
    '''Convert RiometerNet QDC file to match standard data type.

    This function can be used by datasets as the load_converter for
    RioQDC data. It is not intended to be called directly by user
    programs, use load_qdc instead.

    archive: name of archive from which data was loaded
    archive_info: archive metadata
    '''

    assert data_type == 'RioQDC', 'Illegal data_type'
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
            data = ap.loadtxt(uh)
            sample_start_time = (np.timedelta64(1000000, 'us') * data[0])
            sample_end_time = sample_start_time \
                + archive_data['nominal_cadence']
    
            integration_interval = None
            data = data[col_idx]
            r = RioQDC(project=project,
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
                       sort=False,
                       processing=[])
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
    assert isinstance(md, RioData), 'Data is wrong type'
    assert md.units == 'dB', 'Data units incorrect'
    assert md.data.shape[0] == np.size(md.channels), \
        'data shape incorrect for number of channels'
    assert md.data.shape[1] == np.size(md.sample_start_time), \
        'data shape incorrect for number of samples'
    data = np.empty([1 + np.size(md.channels),
                     np.size(md.sample_start_time)])
    data[0] = md.sample_start_time.astype('m8[s]').astype('int64')
    data[1:] = md.data
    fmt = ['%d']
    fmt.extend(['%.2f'] * np.size(md.channels))
    np.savetxt(file_name, data.T, delimiter='\t', fmt=fmt)


class RioData(Data):
    '''Class to manipulate and display riometer data.'''

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
                 processing=[],
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
                      processing=processing,
                      sort=sort)


    def data_description(self):
        return 'Riometer data'

    def savetxt(self, filename, fmt=None):
        a = np.zeros([self.channels.size+1, self.data.shape[-1]],
                     dtype='float')
        a[0] = dt64.dt64_to(self.sample_start_time, 'us') / 1e6
        if self.units == 'dB':
            # Convert to dB here, if conversion needed
            a[1:] = self.data
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
            xch = np.int(np.ceil(np.sqrt(len(channels))))
            ych = np.int(np.ceil(np.float(len(channels))/xch))
            for n in range(1, len(channels) + 1):
                subplot2.append((xch,ych,n))
        else:
            subplot2=subplot

        r = Data.plot(self, channels=channels, figure=figure, axes=axes,
                      subplot=subplot2, units_prefix=units_prefix,
                      title=title, 
                      start_time=start_time, end_time=end_time, 
                      time_units=time_units, **kwargs)
        return r



class RioPower(RioData):
    '''Class to manipulate and display riometer power data.'''

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
                 processing=[],
                 sort=None):
        RioData.__init__(self,
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
                      processing=processing,
                      sort=sort)


    def data_description(self):
        return 'Power'



    def plot_with_qdc(self, qdc, fit_err_func=None,channels=None,**kwargs):
        if 'apply QDC' in self.processing:
            logger.warn('Plotting QDC with data, but QDC already '
                        'applied to the data.')
        if qdc is not None:
            if channels is None:
                channels = self.channels
            # Only plot channels in QDC and data, in order
            channels = list(set(channels)&set(qdc.channels))
            channels = [c for c in self.channels if c in channels]
            self.plot(channels=channels,**kwargs)
            qdc.align(self, fit_err_func=fit_err_func).plot(figure=plt.gcf(),
                                                            channels=channels,
                                                            **kwargs)

    def apply_qdc(self, qdc, fit_err_func=None, inplace=False):
        if any([c not in qdc.channels for c in self.channels]):
            logger.warn('Not all data channels in QDC.')
        if 'apply QDC' in self.processing:
            logger.warn('QDC already applied to the data.')
        channels = [c for c in self.channels if c in qdc.channels]
        cidx = self.get_channel_index(channels)
        if inplace:
            r = self
            r.channels = r.channels[cidx]
            r.data = r.data[cidx]
            if r.integration_interval is not None:
                r.integration_interval = r.integration_interval[cidx]
        else:
            r = copy.copy(self)
            for k in (set(self.__dict__.keys())
                      - set(['channels','data','integration_interval'])):
                setattr(r, k, copy.deepcopy(getattr(self, k)))
            r.channels = copy.copy(self.channels[cidx])
            r.data = copy.copy(self.data[cidx])
            if r.integration_interval is not None:
                r.integration_interval = copy.copy(r.integration_interval[cidx])
        if qdc is not None:
            qdc_data = qdc.align(self, fit_err_func=fit_err_func).data
            for chan in channels:
                rcidx = r.get_channel_index(chan)
                qcidx = qdc.get_channel_index(chan)
                r.data[rcidx] -= qdc_data[qcidx]
                r.data[rcidx] *= -1
            r.processing.append('apply QDC')
        return r

    def make_qdc(self,
                 channels=None,
                 base_qdc=None,
                 cadence=np.timedelta64(5, 's').astype('m8[us]'),
                 upper_envelope_rows=[2,3],
                 smooth=True):

        if 'apply QDC' in self.processing:
            logger.warn('A QDC has already been applied to the data.')

        if channels is None:
            channels = copy.copy(self.channels)
        channels = [c for c in self.channels if c in channels]

        lon = ap.get_site_info(self.project,self.site)['longitude']

        # Create array *qdc_arr_st* that is shape:
        #      [number sidereal days, num samples in a day]
        # it contains sidereal start times of samples in sidereal time
        # from the 2000 epoch. A corresponding array *data_arr* of the
        # same size will be filled with data interpolated for these times
        # using tsintegrate.so will be implemented in future.
        sid_start_day = np.floor(dt64.get_sidereal_day(
                                 self.sample_start_time[0],lon)
                                ).astype('int64')
        sid_end_day = np.ceil(dt64.get_sidereal_day(
                              self.sample_end_time[-1],lon)
                              ).astype('int64')
        num_sid_days = sid_end_day - sid_start_day
        # QDC times are in sidereal units, 0-24 sidereal hrs, not 0-23:56
        qdc_sam_st = np.arange(np.timedelta64(0, 's').astype('m8[us]'),
                           np.timedelta64(24, 'h').astype('m8[us]'),
                           cadence)
        qdc_sam_et = qdc_sam_st + cadence
        num_qdc_sam = qdc_sam_st.size
        sid_days = np.arange(sid_start_day,sid_end_day
                            ).astype('m8[D]').astype('m8[us]')
        qdc_arr_st = np.repeat(sid_days,num_qdc_sam).reshape(num_sid_days,
                                                            num_qdc_sam)+\
                     np.tile(qdc_sam_st,num_sid_days).reshape(num_sid_days,
                                                            num_qdc_sam)
        qdc_arr_et = qdc_arr_st + cadence

 
        # Calculate the sidereal times of the recorded data samples
        sid_sam_st = dt64.get_sidereal_time(self.sample_start_time,lon,
                                            time_units='us',
                                            time_of_day = False,
                                            sidereal_units=True)
        sid_sam_et = dt64.get_sidereal_time(self.sample_end_time,lon,
                                            time_units='us',
                                            time_of_day = False,
                                            sidereal_units=True)

        r = self.resample_data(qdc_arr_st,qdc_arr_et,channels=channels,
                               alt_sample_start_time = sid_sam_st,
                               alt_sample_end_time = sid_sam_et,
                               inplace=False)

        if base_qdc is None:
            qdc_channels = ap.get_archive_info(r.project,
                                               r.site,
                                               'RioPower')[1]['channels']
            if not all([c in qdc_channels for c in channels]):
                logger.warn('inconsistent channels in data and default archive')
                qdc_channels = np.union1d(qdc_channels,channels)
        else:
            qdc_channels = copy(base_qdc.channels)
            if not all([c in qdc_channels for c in channels]):
                logger.warn('inconsistent channels in data and base QDC')
                qdc_channels = np.union1d(qdc_channels,channels)

        
        #cidx = r.get_channel_index(channels)
        qdc_data = np.zeros([len(qdc_channels), num_qdc_sam])*np.nan 
        for c in channels:
            ## interp1d needs mark_missing_data before. Replacing with 
            ## resample_data be better, but depends on compiled code
            #xi = sid_sam_st + (sid_sam_et-sid_sam_st)/2
            #xo = qdc_arr_st + (qdc_arr_et-qdc_arr_st)/2
            #data_arr = scipy.interpolate.interp1d(xi.astype('int64'),
            #                                     self.data[n],kind='nearest',
            #                                     bounds_error=False,
            #                                     fill_value=np.nan)\
            #                                     (xo.astype('int64'))
            cidx =  r.get_channel_index(c)
            data_arr = r.data[cidx].reshape(num_sid_days,num_qdc_sam)
            ########
            # Here is the QDC calculation code.
            # Upper envelope: sort the values, then select the index...
            upper_idx = - np.array(upper_envelope_rows)
            # NaNs get sorted to the top, want them at the bottom
            data_arr[np.isnan(data_arr)] = -np.inf
            data_arr = np.sort(data_arr,axis=0)
            data_arr[np.isinf(data_arr)] = np.nan
            if data_arr.shape[0] >= (np.max(upper_idx)+1):
                cidx = np.where(qdc_channels==c)[0][0]
                qdc_data[cidx,:] = ap.nanmean(data_arr[upper_idx,:],axis=0)


        qdc = RioQDC(project=copy.copy(r.project),
                     site=copy.copy(r.site),
                     channels=qdc_channels,
                     start_time=np.timedelta64(0, 'h'),
                     end_time=np.timedelta64(24, 'h'),
                     sample_start_time=qdc_sam_st,
                     sample_end_time=qdc_sam_et,
                     integration_interval=None,
                     nominal_cadence=cadence,
                     data=qdc_data,
                     units=copy.copy(r.units),
                     sort=False,
                     processing=copy.copy(r.processing))
        qdc.processing.append('make QDC')
        if smooth:
            qdc.smooth(inplace=True)
        return qdc

class RioAbs(RioData):
    '''Class to manipulate and display riometer absorption data.'''

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
                 processing=[],
                 sort=None):
        RioData.__init__(self,
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
                      processing=processing,
                      sort=sort)

    def data_description(self):
        return 'Absorption'

    def plot_keogram(self, channels=None, figure=None, axes=None,
                     start_time=None, end_time=None, time_units=None,
                     **kwargs):
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
        if channels is None:
            channels = self.channels
        ax = plt.gca()
        ax.pcolormesh(self.data,shading='flat')





class RioQDC(RioData):
    '''Class to load and manipulate Riometer quiet-day curves (QDC).'''
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
                 sort=None,
                 processing=[]):
        RioData.__init__(self,
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
                         sort=sort,
                         processing=processing)

    def data_description(self):
        return 'Riometer QDC'

    def qsave(self,t,archive=None,path=None,save_converter=None):
        assert ((archive is not None and path is None) or 
                (archive is None and path is not None)), \
            'archive or path must be defined (and not both)'
        an, ai = ap.get_archive_info(self.project,
                                     self.site,
                                     self.__class__.__name__,
                                     archive=archive)
        if path is None:
            path = ai['path']
            t0 = np.datetime64('1900') # A Monday, best time for boundaries
            t = dt64.floor(t-t0, ai['duration'])+t0
        if save_converter is None:
            save_converter = ai['save_converter']

        file_name = dt64.strftime(t, path)
        q = self.set_time_units(ai['nominal_cadence'], inplace=False)
        import os
        dir_name = os.path.dirname(file_name)
        if not os.path.exists(dir_name):
            logger.debug('making directory %s', dir_name)
            os.makedirs(dir_name)
        logger.info('saving to %s', file_name)
        save_converter(q, file_name, ai)

    def smooth(self, fit_order=20, inplace=False):
        if inplace:
            r = self
        else:
            r = copy.deepcopy(self)
            
        coeffs = np.fft.fft(r.data)
        coeffs[:, range(fit_order+1, coeffs.shape[1]-fit_order)] = 0
        r.data = np.fft.ifft(coeffs).real
        r.processing.append('smooth')
        return r


    def plot(self, channels=None, figure=None, axes=None,
             subplot=None, units_prefix=None, title=None, 
             start_time=None, end_time=None, time_units=None, **kwargs):
        
        if start_time is None:
            start_time = self.start_time
        if end_time is None:
            end_time = self.end_time

        r = RioData.plot(self, channels=channels, figure=figure, axes=axes,
                         subplot=subplot, units_prefix=units_prefix,
                         title=title, 
                         start_time=start_time, end_time=end_time, 
                         time_units=time_units, **kwargs)
        return r

    
    def align(self, ref, fit=None, **fit_kwargs):
        day = np.timedelta64(24, 'h')
        if isinstance(ref, RioData):
            sample_start_time = copy.copy(ref.sample_start_time)
            sample_end_time = copy.copy(ref.sample_end_time)
            nominal_cadence = copy.copy(ref.nominal_cadence)
        else:
            # ref parameter must contain sample timestamps
            assert not fit, 'Cannot fit without reference data'
            sample_start_time = np.sort(np.array(ref).flatten())
            sample_end_time = copy.copy(sample_start_time)
            if len(sample_start_time) >= 2:
                # Guess the nominal cadence
                nominal_cadence = np.median(np.diff(sample_start_time))
            else:
                nominal_cadence = None
        start_time = sample_start_time[0]
        end_time = sample_end_time[-1]
        r = RioData(project=copy.copy(self.project),
                    site=copy.copy(self.site),
                    channels=copy.copy(self.channels),
                    start_time=start_time,
                    end_time=end_time,
                    sample_start_time=sample_start_time,
                    sample_end_time=sample_end_time,
                    integration_interval=None,
                    nominal_cadence=nominal_cadence,
                    data=None,
                    units=copy.copy(self.units),
                    sort=False)
        interp_times = r.get_mean_sample_time()

        # Create array with room for additional entries at start and end
        xi = np.zeros([len(self.sample_start_time) + 2], dtype='m8[us]')
        xi[1:-1] = self.get_mean_sample_time().astype('m8[us]')
        # Extend so that interpolation can happen correctly near midnight
        xi[0] = xi[-2] - day
        xi[-1] = xi[1] + day

        lon = ap.get_site_info(self.project,self.site)['longitude']
        xo = dt64.get_sidereal_time(interp_times,lon,time_of_day=True
                                    ).astype('m8[us]')


        yi = np.zeros(self.data.shape[1] + 2, dtype=self.data.dtype)
        r.data = np.zeros([len(r.channels), len(xo)], dtype=self.data.dtype)
        for c in r.channels:
            cidx = self.get_channel_index(c)
            yi[1:-1] = self.data[cidx]
            yi[0] = yi[-2]
            yi[-1] = yi[1]
            cidx = r.get_channel_index(c)
            r.data[cidx] = scipy.interpolate.interp1d(xi.astype('int64'), yi)\
                                                     (xo.astype('int64'))

        if fit:
            if hasattr(fit, '__call__'):
                return fit(r, ref, inplace=True, **fit_kwargs)
            elif fit in ('baseline', 'realtime_baseline'):
                # QDC must be zero-mean for baseline adjustment,
                # correct any offset
                for n in range(r.data.shape[0]):
                    r.data[n] -= ap.nanmean(self.data[n])
                for d in np.unique(dt64.get_date(r.sample_start_time)):
                    bl = ap.load_data(r.project, r.site, 'RioData',
                                      d, d + np.timedelta64(1, 'D'),
                                      channels=r.channels,
                                      archive=fit)
                    if bl is None or bl.data.size == 0:
                        raise Exception('No %s RioData for %s/%s %s' 
                                        % (fit, self.project, self.site,
                                           str(d)))
                    r.data[:, d == dt64.get_date(r.sample_start_time)] \
                        += bl.data
            else:
                raise ValueError('Unknown fit method %s' % repr(fit))
        return r
        
