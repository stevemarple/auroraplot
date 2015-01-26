import logging
import numpy as np

from auroraplot.data import Data

logger = logging.getLogger(__name__)

class TemperatureData(Data):
    '''Class to manipulate and display temperature data.'''
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
                 sort=True):
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
        return 'Temperature'

    def set_units(self, units, inplace=False):
        if inplace:
            r = self
        else:
            r = copy.deepcopy(self)
        # Accept various representations for degrees Celsius. The
        # final one uses the single-character unicode represention for
        # deg C.
        celsius = ['Celsius', 'C', u'\N{DEGREE SIGN}C', u'\u2103']
        if units in celsius:
            if r.units in celsius:
                return r
            elif r.units == 'K':
                # K -> Celsius
                r.data -= 273.15
                r.units = units
            else:
                raise Exception('Unknown units')
        elif units == 'K':
            if r.units in celsius:
                # Celsius -> K
                r.data += 273.15
                r.units = units
            elif r.units == 'K':
                return r
            else:
                raise Exception('Unknown units')
        else:
            raise Exception('Unknown units')
        return r

