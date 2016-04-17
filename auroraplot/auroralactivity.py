import copy
import logging
import numpy as np
import matplotlib as mpl
import matplotlib.ticker
import matplotlib.pyplot as plt

import auroraplot as ap
import auroraplot.data
import auroraplot.magdata
from auroraplot.data import Data
import auroraplot.dt64tools as dt64
import auroraplot.tools

from scipy.stats import nanmean

logger = logging.getLogger(__name__)

    
class AuroraWatchActivity(Data):
    '''
    Class to manipulate and display geomagnetic activity as way used
    by AuroraWatch UK.
    '''

    
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
                 sort=False,
                 magdata=None,
                 magqdc=None, 
                 thresholds=None,
                 colors=None,
                 fit=None,
                 fit_params={},
                 input_channels=None,
                 range_=False):
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
        
        if magdata is not None and magqdc is not None:
            assert magdata.units == magqdc.units, 'Units must match'
            cadence = np.timedelta64(60, 'm')
            cad_units = dt64.get_units(cadence)
            if isinstance(magqdc, ap.magdata.MagQDC):
                aligned = magqdc.align(magdata, fit=fit, **fit_params)
            else:
                aligned = magqdc
            
            if input_channels is None:
                input_channels = []
                if 'H' in magdata.channels:
                    input_channels.append('H')
                elif 'X' in magdata.channels:
                    input_channels.append('X')
                if 'D' in magdata.channels:
                    input_channels.append('D')
                elif 'E' in magdata.channels:
                    input_channels.append('E') # IAGA notation for D when in nT
                elif 'Y' in magdata.channels:
                    input_channels.append('Y')
            
            input_channels = np.array(input_channels).flatten()
            cidx = magdata.get_channel_index(input_channels)
            if magdata.nominal_cadence <= np.timedelta64(5, 's'):
                # Throw away ~30 seconds
                n = int(np.timedelta64(30, 's') / magdata.nominal_cadence)
            else:
                # Throw away up to 2.5 minutes
                n = int(np.timedelta64(150, 's') / magdata.nominal_cadence)

            nth_largest = ap.tools.NthLargest(n)

            disturbance = magdata.extract(channels=input_channels)
            disturbance.data = np.abs(magdata.data[cidx] -
                                      aligned.data[cidx])
            disturbance.set_cadence(cadence,
                                    inplace= True, 
                                    aggregate=nth_largest)
            data = disturbance.data.copy()

            if range_:
                range_high = magdata.extract(channels=input_channels)
                range_high.set_cadence(cadence,
                                       inplace= True, 
                                       aggregate=nth_largest)
                nth_smallest = ap.tools.NthLargest(n, smallest=True)
                range_low = magdata.extract(channels=input_channels)
                range_low.set_cadence(cadence,
                                      inplace= True, 
                                      aggregate=nth_smallest)
                hourly_range = copy.deepcopy(range_high)
                hourly_range.data = range_high.data - range_low.data
                assert np.all(hourly_range.sample_start_time ==
                              disturbance.sample_start_time), \
                              'sample start times differ'
                data = np.vstack([data, hourly_range.data])              
                
            self.project = magdata.project
            self.site = magdata.site
            if range_:
                self.channels = np.array(['Activity'])
            else:
                self.channels = np.array(['Disturbance'])
            self.start_time = disturbance.start_time.astype('datetime64[%s]' % cad_units)
            self.end_time = disturbance.end_time.astype(self.start_time.dtype)
            self.sample_start_time = disturbance.sample_start_time
            self.sample_end_time = disturbance.sample_end_time
            self.integration_interval = None
            self.nominal_cadence = cadence
            # Take the maximum value for all 
            self.data = np.max(data, axis=0).reshape([1, data.shape[1]])
            self.units = magdata.units
            if sort:
                self.sort(inplace=True)

            if self.nominal_cadence <= np.timedelta64(5, 's'):
                # Throw away ~30 seconds
                n = int(np.timedelta64(30, 's') / self.nominal_cadence)
             # elif self.nominal_cadence < np.timedelta64(2, 'm'):
            else:
                # Throw away up to 2.5 minutes
                n = int(np.timedelta64(150, 's') / self.nominal_cadence)

            nth_largest = ap.tools.NthLargest(n)
            self.set_cadence(np.timedelta64(60, 'm'),
                             inplace= True, aggregate=nth_largest)

        if thresholds is None:
            self.thresholds = self.get_site_info('activity_thresholds')
        else:
            self.thresholds = thresholds
        if colors is None:
            self.colors = self.get_site_info('activity_colors')
        else:
            self.colors = colors



    def data_description(self):
        return 'Geomagnetic activity'

    def get_color(self):
        assert self.data.shape[0] == 1, 'Can only get color for single channel'
        assert np.all(self.thresholds == np.sort(self.thresholds)), \
            'thresholds not in ascending order'
        
        col = np.zeros([self.data.shape[-1], 3])
        col[:] = self.colors[0]

        data_no_nans = self.data.copy()
        data_no_nans[np.logical_not(np.isfinite(self.data))] = -1
        for n in range(1, len(self.thresholds)):
            col[data_no_nans[0] >= self.thresholds[n]] = self.colors[n]
        return col
            
    def get_level(self):
        '''Return numerical level

        Values return the indices in the thresholds, with -1 indicating
        missing data.
        '''

        r = np.zeros(self.data.shape, dtype=int)
        data = self.data.copy()
        isnan = np.logical_not(np.isfinite(self.data))
        r[isnan] = -1
        data[isnan] = -1
        for n in range(1, len(self.thresholds)):
            r[data >= self.thresholds[n]] = n
        return r

    def plot(self, channels=None, figure=None, axes=None,
             subplot=None, units_prefix=None, title=None, 
             # Our own options
             start_time=None, end_time=None, time_units=None, 
             plot_func=plt.bar, plot_thresholds=True,
             **kwargs):

        if units_prefix is None and self.units == 'T':
            # Use nT for plotting
            units_prefix = 'n'
        
        if plot_func == plt.bar:
            if time_units is None:
                time_units = dt64.get_units(self.sample_start_time)
            if not kwargs.has_key('width'):
                kwargs['width'] = dt64.dt64_to(self.nominal_cadence, time_units)

            if not kwargs.has_key('align'):
                kwargs['align'] = 'center'

            if not kwargs.has_key('color'):
                kwargs['color'] = self.get_color()

            if not kwargs.has_key('edgecolor'):
                kwargs['edgecolor'] = self.get_color() * .9

            if not kwargs.has_key('linewidth'):
                kwargs['linewidth'] = 0.5

            if not kwargs.has_key('zorder'):
                kwargs['zorder'] = 4

                
        r = Data.plot(self, channels=channels, figure=figure, axes=axes,
                      subplot=subplot, units_prefix=units_prefix,
                      title=title, 
                      start_time=start_time, end_time=end_time, 
                      time_units=time_units, plot_func=plot_func, **kwargs)

        if plot_thresholds:
            if axes is None:
                axes = plt.gca()
            mul = ap.str_units(0, 'T', units_prefix, wantstr=False)['mul']
            tmp = self.extract(start_time, end_time, channels=channels)
            ymax = np.nan
            ylim = axes.get_ylim()
            if np.size(tmp.data):
                ymax = np.nanmax(tmp.data) / mul
            if np.isnan(ymax):
                ymax = axes.get_ylim()[1]
            new_ylim_top = ylim[1]

            for n in range(len(self.thresholds)-1, 0, -1):
                y = self.thresholds[n] / mul
                if y >= ymax:
                    new_ylim_top = y*1.05
                axes.plot(axes.xaxis.get_view_interval(), 
                          self.thresholds[[n,n]] / mul, 
                          color=self.colors[n], 
                          linestyle='-',
                          linewidth=2,
                          zorder=3)
            axes.set_ylim(top=new_ylim_top)
        return r


class KIndex(Data):
    '''
    Class to represent the geomagnetic K index.
    '''

    def __init__(self,
                 project=None,
                 site=None,
                 channels=None,
                 start_time=None,
                 end_time=None,
                 sample_start_time=np.array([]),
                 sample_end_time=np.array([]),
                 nominal_cadence=np.timedelta64(3, 'h'),
                 integration_interval=np.array([]),
                 data=np.array([]),
                 units=None,
                 sort=False,
                 magdata=None,
                 magqdc=None, 
                 scale=None,
                 nth=1,
                 fit=None,
                 fit_params={},
                 with_qdc=None):
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

        if magdata is not None:
            self.project = magdata.project
            self.site = magdata.site
            self.channels = c = magdata.channels
            self.start_time = dt64.floor(magdata.start_time, 
                                         self.nominal_cadence)
            self.end_time = dt64.ceil(magdata.end_time,
                                      self.nominal_cadence)

            self.sample_start_time = np.arange(self.start_time, self.end_time,
                                               self.nominal_cadence)
            self.sample_end_time = self.sample_start_time + self.nominal_cadence
            self.integration_interval = None
                        
            if magqdc is None:
                logger.info('Creating KIndex object without a QDC')
                bsub = magdata.data[magdata.get_channel_index(c)]
                self.with_qdc = False
            else:
                assert magdata.units == magqdc.units, 'Units must match'
                if isinstance(magqdc, ap.magdata.MagQDC):
                    aligned = magqdc.align(magdata, fit=fit, **fit_params)
                else:
                    aligned = magqdc                            

                # Baseline subtracted data
                bsub = np.abs(magdata.data[magdata.get_channel_index(c)] -
                              aligned.data[aligned.get_channel_index(c)])
                self.with_qdc = False

            self.units = None

            if nth is None:
                if magdata.nominal_cadence <= np.timedelta64(5, 's'):
                    # Throw away ~30 seconds
                    nth = int(np.timedelta64(30, 's') \
                                  / magdata.nominal_cadence)
                else:
                    # Throw away up to 2.5 minutes
                    nth = int(np.timedelta64(150, 's') \
                                  / magdata.nominal_cadence)

            nth_largest = ap.tools.NthLargest(nth)
            nth_smallest = ap.tools.NthLargest(nth, smallest=True)
            
            self.range = np.zeros([len(self.channels), 
                                  len(self.sample_start_time)])
            for i in range(len(self.sample_start_time)):
                tidx = np.where(np.logical_and(magdata.sample_start_time >=
                                               self.sample_start_time[i],
                                               magdata.sample_end_time <=
                                               self.sample_end_time[i]))[0]
                for cn in range(len(self.channels)):
                    self.range[cn, i] 
                    nth_largest(bsub[cn, tidx])
                    nth_smallest(bsub[cn, tidx])
                    self.range[cn, i] = nth_largest(bsub[cn, tidx]) \
                        - nth_smallest(bsub[cn, tidx])

            self.data = np.tile(np.nan, self.range.shape)
            self.data[np.nonzero(np.isfinite(self.range))] = 0

            if scale is None:
                scale = self.get_site_info('k_index_scale')
                
            # K-index thresholds may be scaled but all are proportional to
            # the limits Bartels defined for Niemegk observatory.
            self.thresholds = np.array([0.00, 0.01, 0.02, 0.04,
                                        0.08, 0.14, 0.24, 0.40,
                                        0.66, 1.00]) * scale


            for i in range(1, len(self.thresholds)):
                self.data[np.nonzero(self.range >= self.thresholds[i])] = i


    def data_description(self):
        return 'K index'


    def get_color(self):
        assert self.data.shape[0] == 1, 'Can only get color for single channel'
        is_valid = np.isfinite(self.data[0])
        col = np.zeros([self.data.shape[-1], 3])
        col[:] = np.array([0.2, 1.0, 0.2]) # Green

        [0.2, 1.0, 0.2],  # green  
            #                         [1.0, 1.0, 0.0],  # yellow
            #                         [1.0, 0.6, 0.0],  # amber
            #                         [1.0, 0.0, 0.0]

        # col[self.data[0][is_valid] >= 4] = np.array([1.0, 1.0, 0.0]) # Yellow
        col[self.data[0][is_valid] >= 4] = np.array([1.0, 0.6, 0.0]) # Amber
        col[self.data[0][is_valid] >= 5] = np.array([1.0, 0.0, 0.0]) # Red
        return col


    def plot(self, channels=None, figure=None, axes=None,
             subplot=None, title=None, bottom=-0.2,
             # Our own options
             start_time=None, end_time=None, time_units=None, plot_func=plt.bar,
             **kwargs):
        
        if plot_func == plt.bar:
            data2 = copy.copy(self.data)
            
            # Make zero values visible
            data2 -= bottom
            kwargs['bottom'] = bottom
            
            self = copy.copy(self)
            self.data = data2
            if channels is None:
                self.data = np.array([np.max(self.data, axis=0)])
                self.channels = np.array([','.join(self.channels)])
                channels = self.channels
            else:
                ci = self.get_channel_index(channels)
                self.data = np.array([np.max(self.data[ci], axis=0)])
                self.channels = ','.join(self.channels[ci])
                
            if time_units is None:
                time_units = dt64.get_units(self.sample_start_time)
            if not kwargs.has_key('width'):
                kwargs['width'] = dt64.dt64_to(self.nominal_cadence, 
                                               time_units) * 0.8

            if not kwargs.has_key('align'):
                kwargs['align'] = 'center'

            if not kwargs.has_key('color'):
                kwargs['color'] = self.get_color()

            if not kwargs.has_key('linewidth'):
                kwargs['linewidth'] = 0.0

        else:
            bottom = 0
        r = Data.plot(self, channels=channels, figure=figure, axes=axes,
                      subplot=subplot, units_prefix='',
                      title=title, 
                      start_time=start_time, end_time=end_time, 
                      time_units=time_units, plot_func=plot_func, **kwargs)
        ax = plt.gca()
        if plot_func == plt.bar:
            ax.set_ylabel('Local K index (' + ','.join(channels) + ')')

        ax.set_ylim(bottom, 9)
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
        return r
