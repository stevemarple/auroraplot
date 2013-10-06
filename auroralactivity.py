import numpy as np
import matplotlib.pyplot as plt

import auroraplot as ap
import auroraplot.data
import auroraplot.magdata
from auroraplot.data import Data
import auroraplot.dt64tools as dt64
import auroraplot.tools

from scipy.stats import nanmean

    
class AuroraWatchActivity(Data):
    '''
    Class to manipulate and display geomagnetic activity as way used
    by AuroraWatch UK.
    '''

    
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
                 sort=True,
                 magdata=None,
                 magqdc=None, 
                 thresholds=None,
                 colors=None,
                 fit=None,
                 fit_params={}):
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
        
        if magdata is not None and magqdc is not None:
            self.network = magdata.network
            self.site = magdata.site
            self.channels = c = magdata.channels
            self.start_time = magdata.start_time
            self.end_time = magdata.end_time
            self.sample_start_time = magdata.sample_start_time
            self.sample_end_time = magdata.sample_end_time
            self.integration_interval = None
            self.nominal_cadence = magdata.nominal_cadence
                        
            if isinstance(magqdc, ap.magdata.MagQDC):
                aligned = magqdc.align(magdata, fit=fit, **fit_params)
            else:
                aligned = magqdc                            
            self.data = np.abs(magdata.data[magdata.get_channel_index(c)] -
                               aligned.data[aligned.get_channel_index(c)])
            assert magdata.units == magqdc.units, 'Units must match'
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
            self.set_cadence(np.timedelta64(1, 'h'),
                             inplace= True, aggregate=nth_largest)

        if thresholds is None:
            # self.thresholds = np.array([0.0, 50.0, 100.0, 200.0]) * 1e-9
            self.thresholds = self.get_site_info('activity_thresholds')
        else:
            self.thresholds = thresholds
        if colors is None:
            # self.colors = np.array([[0.2, 1.0, 0.2],  # green  
            #                         [1.0, 1.0, 0.0],  # yellow
            #                         [1.0, 0.6, 0.0],  # amber
            #                         [1.0, 0.0, 0.0]]) # red
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
        for n in range(1, len(self.thresholds)):
            col[self.data[0] >= self.thresholds[n]] = self.colors[n]
        return col
            
    def plot(self, channels=None, figure=None, axes=None,
             subplot=None, units_prefix=None, title=None, 
             # Our own options
             start_time=None, end_time=None, time_units=None, plot_func=plt.bar,
             **kwargs):
        
        if plot_func == plt.bar:
            if time_units is None:
                time_units = dt64.get_units(self.sample_start_time)
            if not kwargs.has_key('width'):
                kwargs['width'] = dt64.dt64_to(self.nominal_cadence, time_units)

            if not kwargs.has_key('align'):
                kwargs['align'] = 'center'

            if not kwargs.has_key('color'):
                kwargs['color'] = self.get_color()

            if not kwargs.has_key('linewidth'):
                kwargs['linewidth'] = 0.0
                
        r = Data.plot(self, channels=channels, figure=figure, axes=axes,
                      subplot=subplot, units_prefix=units_prefix,
                      title=title, 
                      start_time=start_time, end_time=end_time, 
                      time_units=time_units, plot_func=plot_func, **kwargs)
        return r
