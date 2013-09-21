import copy
# import pickle
import cPickle as pickle

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import scipy.interpolate
# import scipy.stats

import auroraplot as ap
import auroraplot.dt64tools as dt64

class Data(object):
    '''Base class for time-series data.'''

    def __init__(self, 
                 network=None,
                 site=None,
                 channels=None,
                 start_time=None,
                 end_time=None,
                 sample_start_time=np.array([]), 
                 sample_end_time=np.array([]), 
                 integration_interval=None, 
                 nominal_cadence=None,
                 data=np.array([]),
                 units=None,
                 sort=True):
        self.network = network
        self.site = site
        self.channels = np.array(channels)
        self.start_time = start_time
        self.end_time = end_time
        self.sample_start_time = sample_start_time
        self.sample_end_time = sample_end_time
        self.integration_interval=integration_interval
        self.nominal_cadence = nominal_cadence
        self.data = data
        self.units = units
        if sort:
            self.sort(inplace=True)

    def __repr__(self):
        units = self.units
        if units == u'\N{DEGREE SIGN}C':
            units = 'degrees C'
        elif units == u'\N{DEGREE SIGN}':
            units = 'degrees'

        return (type(self).__name__ + ':\n' +
                '          network : ' + str(self.network) + '\n' +
                '             site : ' + str(self.site) + '\n' +
                '         channels : ' + str(self.channels) + '\n' +
                '       start_time : ' + str(self.start_time) + '\n' +
                '         end_time : ' + str(self.end_time) + '\n' +
                'sample_start_time : ' + repr(self.sample_start_time) + '\n' + 
                '  sample_end_time : ' + repr(self.sample_end_time) + '\n' + 
                'integration intv. : ' + repr(self.integration_interval)+'\n'+
                '   nominal cadence: ' + str(self.nominal_cadence) + '\n' +
                '             data : ' + repr(self.data) + '\n' + 
                '            units : ' + str(units))

    def data_description(self):
        return 'Data'

    def assert_valid(self):
        import re
        for n in ('network', 'site', 'channels', 'start_time', 'end_time', 
                  'sample_start_time', 'sample_end_time',
                  'nominal_cadence',
                  'data', 'units'):
            attr = getattr(self, n)
            assert (attr is not None and 
                    (not isinstance(attr, basestring) or attr != '')), \
                    n + ' not set'
        assert re.match('^[-A-Z0-9]+$', self.network), 'Bad value for network'
        assert re.match('^[-A-Z0-9]+$', self.site), 'Bad value for site'

        num_channels = len(self.channels)
        num_samples = len(self.sample_start_time)

        assert self.start_time <= self.end_time, \
            'start_time must not be after end_time'
        assert len(self.sample_end_time) == num_samples, \
            'lengths of sample_start_time and sample_end_time differ'
       
        if self.integration_interval is not None:
            assert self.integration_interval.shape == \
                (num_channels, num_samples), \
                'integration_interval incorrect shape'
        assert self.data.shape == (num_channels, num_samples), \
            'data incorrect shape'

        return True


    def get_channel_index(self, channels):
        '''
        Find the location of the listed channels in the objects channel list.
        
        channels: list of channels to find.

        return list of integers correpsonging to location in the
        object's channels attribute.
        '''
        chan_tup = tuple(self.channels)
        cidx = []
        for c in channels:
            cidx.append(chan_tup.index(c))
        return cidx

    def get_mean_sample_time(self):
        return dt64.mean(self.sample_start_time, self.sample_end_time)

    def pickle(self, filename):
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)


    def extract(self, start_time=None, end_time=None, channels=None, 
                inplace=False):
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

            ### TODO: Add option to accept sample which straddle
            ### boundary conditions.
            tidx = (self.sample_start_time >= start_time) & \
                (self.sample_end_time <= end_time)
            r.start_time = start_time
            r.end_time = end_time
            r.sample_start_time = r.sample_start_time[tidx]
            r.sample_end_time = r.sample_end_time[tidx]
            r.data = r.data[:, tidx]
            if r.integration_interval is not None:
                r.integration_interval = r.integration_interval[:, tidx]

        # elif channels is None and not inplace:
        #     # data matrix has not been copied
        #     r.data = r.data.copy()
        return r

    def sort(self, inplace=False):
        idx = np.argsort(self.sample_start_time)
        if inplace:
            new = self
        else:
            new = copy.deepcopy(self)

        new.sample_start_time = new.sample_start_time[idx]
        new.sample_end_time = new.sample_end_time[idx]

        # integration_interval may be None, or an array
        if self.integration_interval is not None:
            new.integration_interval = new.integration_interval[:, idx]

        new.data = new.data[:, idx]
        return new
            

    def split(self, interval):
        r = []
        t1 = self.start_time
        while t1 < self.end_time:
            t2 = t1 + interval
            r.append(self.extract(start_time=t1, end_time=t2))
            t1 = t2
        return r


    def plot(self, channels=None, figure=None, axes=None, subplot=None,
             units_prefix=None, subtitle=None, 
             # Our own options
             start_time=None, end_time=None, time_units=None, 
             **kwargs):
        if channels is None:
            channels=self.channels
        elif isinstance(channels, basestring):
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
                axes2 * len(channels)
            else:
                assert len(axes2) == len(channels), \
                    'axes and channels must be same length'
        elif figure is None:
            figure=plt.figure()
            new_figure = True
        else:
            plt.figure(figure)
        

        if subplot is not None:
            subplot2 = copy.copy(subplot)
            if not hasattr(subplot2, '__iter__'):
                subplot2 = [subplot2]
            if len(subplot2) == 1:
                subplot2 * len(channels)
            else:
                assert len(subplot2) == len(channels), \
                    'subplot and channels must be same length'

        if subtitle is None:
            subtitle = self.data_description()

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
        for n in range(len(channels)):
            if axes is not None:
                ax = plt.axes(axes2[n])
            elif subplot is not None:
                ax = plt.subplot(subplot2[n])
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

            if kwargs.has_key('label'):
                r = dt64.plot_dt64(xdata, ydata, **kwargs)
            else:
                r = dt64.plot_dt64(xdata, ydata, label=channels[n], **kwargs)

            ax.set_xlim(dt64.dt64_to(start_time, ax.xaxis.dt64tools.units),
                        dt64.dt64_to(end_time, ax.xaxis.dt64tools.units))
            
            if not need_legend:
                # Lines plotted on different axes
                plt.ylabel(self.channels[cidx] + ' (' + u['fmtunit'] + ')')
            
            if n == 0:
                first_axes = ax
                
        if need_legend:
            lh = plt.legend(self.channels[allcidx], loc='best', fancybox=True)
            lh.get_frame().set_alpha(0.6)
            # Label Y axis
            plt.ylabel(str(subtitle) + ' (' + cu['fmtunit'] + ')')

        if new_figure:
            # Add title
            plt.axes(first_axes)
            tstr = [self.network + ' / ' + self.site]
            if subtitle:
                tstr.append(subtitle)
                plt.subplots_adjust(top=0.85)
            tstr.append(dt64.fmt_dt64_range(start_time, end_time))

            plt.title('\n'.join(tstr))

        return r

    def set_cadence(self, cadence, ignore_nan=True,
                    offset_interval=np.timedelta64(0, 'us'), inplace=False,
                    aggregate=scipy.average):
        if cadence > self.nominal_cadence:
            sam_st = np.arange(dt64.ceil(self.start_time, cadence) 
                               + offset_interval, self.end_time, cadence)
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
                            integ_intv[cn,sn] = \
                                np.sum(self.integration_interval[cn, tidx2])
                        
                    if len(tidx2) != 0:
                        # Update data. Weight the mean according to
                        # integration_interval if possible.
                        if self.integration_interval is None:
                            weights = None
                        else:
                            weights = self.integration_interval[cn, tidx2]
                            weights[dt64.isnat(weights)] = 0

                        d[cn,sn] = aggregate(self.data[cn, tidx2], 
                                             weights=weights)

            if inplace:
                r = self
            else:
                r = copy.copy(self)
                for k in (set(self.__dict__.keys())
                          - set(['sample_start_time', 'sample_end_time', 
                                 'integration_interval', 'nominal_cadence',
                                 'data'])):
                    setattr(r, k, copy.deepcopy(getattr(self, k)))

            r.sample_start_time = sam_st
            r.sample_end_time = sam_et
            r.integration_interval = integ_intv
            r.nominal_cadence = copy.copy(cadence)
            r.data = d
            return r

        elif cadence < self.nominal_cadence:
            raise Exception('Interpolation to reduce cadence not implemented')
        else:
            # raise Warning('Cadence already at ' + str(cadence))
            warnings.warn('Cadence already at ' + str(cadence))
            if inplace:
                return self
            else:
                return copy.deepcopy(self)
        
        
    # TODO: fix inplace option which does not work
    def mark_missing_data(self, cadence=None, # inplace=False
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
            trim = true
            
        if trim:
            r = self.extract(start_time=start_time, end_time=end_time)
        # elif inplace:
        #     r = self
        else:
            r = copy.deepcopy(self)


        sample_time = dt64.mean(r.sample_start_time, 
                                r.sample_end_time)
        idx = np.where(np.diff(sample_time) > cadence)[0]
        
        
        num_channels = len(r.channels)
        data = np.ones([num_channels, len(idx)]) * ap.NaN
        if r.integration_interval is None:
            ii = None
        else:
            ii = np.zeros([len(r.channels), len(idx)]).astype('m8[us]')

        missing = type(self)(network=r.network,
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
        obj_list = [r, missing]

        # Check for missing data at start
        if r.sample_start_time[0] - start_time > cadence:
            obj_list.append(type(self)(
                    network=r.network,
                    site=r.site,
                    channels=r.channels,
                    start_time=start_time,
                    end_time=r.sample_start_time[0],
                    sample_start_time=np.array([start_time]),
                    sample_end_time=r.sample_start_time[:1],
                    integration_interval=np.zeros([num_channels, 1]).astype('m8[us]'),
                    nominal_cadence=r.nominal_cadence,
                    data=np.ones([num_channels, 1]) * ap.NaN,
                    units=r.units,
                    sort=False))

        # Check for missing data at end
        if end_time - r.sample_end_time[-1] > cadence:
            obj_list.append(type(self)(
                        network=r.network,
                        site=r.site,
                        channels=r.channels,
                        start_time=r.sample_end_time[-1],
                        end_time=end_time,
                        sample_start_time=r.sample_end_time[-1:],
                        sample_end_time=np.array([end_time]),
                        integration_interval=np.zeros([num_channels, 1]).astype('m8[us]'),
                        nominal_cadence=r.nominal_cadence,
                        data=np.ones([num_channels, 1]) * ap.NaN,
                        units=r.units,
                        sort=False))
        r = ap.concatenate(obj_list)
        # if inplace:
        #     self = r
        return r

    def interp(self, sample_start_time, sample_end_time, kind='linear'):
        one_us = np.timedelta64(1, 'us')
        r = copy.deepcopy(self)
        xi = dt64.mean(self.sample_start_time, self.sample_end_time).astype('m8[us]') / one_us

        r.sample_start_time = sample_start_time
        r.sample_end_time = sample_end_time
        xo = dt64.mean(sample_start_time, sample_end_time).astype('m8[us]') / one_us
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
            missing_cadence = self.cadence
        s = self.mark_missing_data(start_time=start_time, end_time=end_time,
                                   cadence=missing_cadence)
        one_us = np.timedelta64(1, 'us')
        
        sam_st = start_time + np.arange(0, (end_time - start_time) / one_us, 
                                        cadence / one_us).astype('m8[us]')
        sam_et = sam_st + cadence
        return s.interp(sam_st, sam_et, kind='linear')


    def _leastsq_residuals(self, p, obj, channel):
        adj, = p
        err = self.data[self.get_channel_index(channel)[0]] \
            - (obj.data[obj.get_channel_index(channel)[0]] + adj)
        return np.nan_to_num(err)


    def least_squares_fit(self, obj, inplace=False):
        '''
        Fit a dataset to a reference dataset by applying an offset to
        find the least-squared error. Uses scipy.optimize.leastsq()

        obj: reference object, of instance Data.
        
        inplace: if True modify self, otherwise modify a copy.

        Returns self (or a copy if 'inplace' is False) with an
        adjustment applied for best fit.
        '''

        # See http://www.tau.ac.il/~kineret/amit/scipy_tutorial/ for
        # helpful tutorial on using leastsq().
        import scipy.optimize

        for c in self.channels:
            assert c in obj.channels, 'Channel not in reference object'

        err_func = self._leastsq_residuals
        
        if inplace:
            r = self
        else:
            r = copy.deepcopy(self)

        # Fit each channel separately
        for c in self.channels:
            channel = self.channels[0]
            p0 = [0.0]
            plsq = scipy.optimize.leastsq(err_func, p0, 
                                          args=(obj, channel))
            r.data[self.get_channel_index(c)] -= plsq[0]

        return r

