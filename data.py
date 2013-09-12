import copy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats

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
                '            units : ' + units)

    def data_description(self):
        return 'Data'

    def assert_valid(self):
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
            tidx = (self.sample_start_time >= start_time) & \
                (self.sample_end_time < end_time)
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
            

    def plot(self, channels=None, figure=None, axes=None, subplot=None,
             units_prefix=None, subtitle=None):
        if channels is None:
            channels=self.channels
        elif isinstance(channels, basestring):
            channels=[channels]
        else:
            try:
                iterator = iter(channels)
            except TypeError:
                channels = [channels]
        
        if axes is not None:
            axes2 = copy.copy(axes)
            if not hasattr(axes2, '__iter__'):
                axes2 = [axes2]
            if len(axes2) == 1:
                axes2 * len(channels)
            else:
                assert len(axes2) == len(channels), \
                    'axes and channels must be same length'
        elif figure is None:
            figure=plt.figure()
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

            r = dt64.plot_dt64(xdata, ydata)
            
            plt.xlim(xmin=dt64.dt64_to(self.start_time, 'ns'),
                     xmax=dt64.dt64_to(self.end_time, 'ns'))

            if not need_legend:
                # Lines plotted on different axes
                # u['channel'] = self.channels[cidx]
                # plt.ylabel('%(channel)s (%(prefix)s%(unit)s)' % u)
                plt.ylabel(self.channels[cidx] + ' (' + u['fmtunit'] + ')')
            
            if n == 0:
                first_axes = ax
                
        if need_legend:
            lh = plt.legend(self.channels[allcidx], loc='best', fancybox=True)
            lh.get_frame().set_alpha(0.7)
            # Label Y axis
            plt.ylabel(str(subtitle) + ' (' + cu['fmtunit'] + ')')

        # Add title
        plt.axes(first_axes)
        tstr = self.network + ' / ' + self.site
        # if len(channels) == 1:
        #    tstr += '\n' + self.channels[0]
        if subtitle:
            tstr += '\n' + subtitle
        # tstr += '\n' + dt64.fmt_dt64_range(self.start_time, self.end_time)
        plt.title(tstr)
        return r

    def set_cadence(self, cadence, ignore_nan=True,
                    offset_interval=np.timedelta64(0, 'ns'), inplace=False):
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
                        if self.integration_interval is not None:
                            weights = self.integration_interval[cn, tidx2]
                            weights[dt64.isnat(weights)] = 0
                            d[cn,sn] = \
                                scipy.average(np.mean(self.data[cn, tidx2]))
                        else:
                            d[cn,sn] = np.mean(self.data[cn, tidx2])
                                    

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
        
        
    def mark_missing_data(self, cadence=None, inplace=False):
        if cadence is None:
            cadence = self.nominal_cadence
        sample_time = dt64.mean(self.sample_start_time, 
                                self.sample_end_time)
        idx = np.where(np.diff(sample_time) > cadence)[0]
        
        if inplace:
            r = self
        else:
            r = copy.deepcopy(self)
        
        data = np.ones([len(self.channels), len(idx)]) * NaN
        if self.integration_interval is None:
            ii = None
        else:
            ii = np.ones([len(self.channels), len(idx)]) * NaT

        missing = type(self)(network=self.network,
                             site=self.site,
                             channels=self.channels,
                             start_time=self.start_time,
                             end_time=self.end_time,
                             sample_start_time=self.sample_end_time[idx],
                             sample_end_time=self.sample_start_time[idx+1],
                             integration_interval=ii,
                             nominal_cadence=self.nominal_cadence,
                             data=data,
                             units=self.units,
                             sort=False)
        
        r = concatenate([self, missing])
        if inplace:
            self = r
        return r
