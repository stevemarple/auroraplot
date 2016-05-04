import copy
import logging
import os
import traceback

# Python 2/3 compatibility
import six
try:
    from urllib.request import urlopen
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse
    from urllib import urlopen

import requests
import numpy as np

import auroraplot as ap
from auroraplot.magdata import MagData as MagData
from auroraplot.temperaturedata import TemperatureData
from auroraplot.voltagedata import VoltageData

logger = logging.getLogger(__name__)

data_dir = 'http://aurorawatch.lancs.ac.uk/data/bgs_sch'

def check_mag_data(data):
    data[np.logical_or(data < -0.0001, data > 0.0001)] = np.nan
    return data

def check_temperature(data):
    data[np.logical_or(data < -40, data > 100)] = np.nan
    return data

def check_voltage(data):
    data[np.logical_or(data < 0, data > 10)] = np.nan
    return data

def load_bgs_sch_data(file_name, archive_data, 
                      project, site, data_type, channels, start_time, 
                      end_time, **kwargs):
    '''Convert AuroraWatchNet data to match standard data type

    data: MagData or other similar format data object
    archive: name of archive from which data was loaded
    archive_info: archive metadata
    '''

    data_type_info = {
        'MagData': {
            'class': MagData,
            'col_offset': 1,
            'scaling': 1e-9,
            'data_check': check_mag_data,
            },
        'TemperatureData' : {
            'class': TemperatureData,
            'col_offset': 4,
            'scaling': 1,
            'data_check': check_temperature,
            },
        'VoltageData': {
            'class': VoltageData,
            'col_offset': 6,
            'scaling': 1,
            'data_check': check_voltage,
            },
        }
    
    assert data_type in data_type_info, 'Illegal data_type'
    chan_tup = tuple(archive_data['channels'])
    col_idx = []
    for c in channels:
        col_idx.append(data_type_info[data_type]['col_offset'] + 
                       chan_tup.index(c))
    try:
        if file_name.startswith('/'):
            uh = urlopen('file:' + file_name)
        else:
            uh = urlopen(file_name)
        try:
            ltkwa = {}
            if file_name.endswith('.csv'):
                ltkwa['delimiter'] = ','

            data = np.loadtxt(uh, unpack=True, **ltkwa)
            sample_start_time = ap.epoch64_us + \
                (np.timedelta64(1000000, 'us') * data[0])
            # end time and integration interval are guesstimates
            sample_end_time = sample_start_time + np.timedelta64(1000000, 'us')
            integration_interval = np.ones([len(channels), 
                                            len(sample_start_time)],
                                            dtype='m8[us]')
            data = data[col_idx] * data_type_info[data_type]['scaling']
            if data_type_info[data_type]['data_check']:
                data = data_type_info[data_type]['data_check'](data)
            r = data_type_info[data_type]['class']( \
                project=project,
                site=site,
                channels=channels,
                start_time=start_time,
                end_time=end_time,
                sample_start_time=sample_start_time, 
                sample_end_time=sample_end_time,
                integration_interval=integration_interval,
                nominal_cadence=archive_data['nominal_cadence'],
                data=data,
                units=archive_data['units'],
                sort=True)
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

cc3_by_nc_sa = 'This work is licensed under the Creative Commons ' + \
    'Attribution-NonCommercial-ShareAlike 3.0 Unported License. ' + \
    'To view a copy of this license, visit ' + \
    'http://creativecommons.org/licenses/by-nc-sa/3.0/deed.en_GB.'

sites = {
    'BGS3': {
        'location': 'Lancaster, UK',
        'latitude': 54.01,
        'longitude': -2.78,
        'elevation': 27,
        'start_time': np.datetime64('2015-10-19T00:00Z'),
        'end_time': None, # Still operational
        'k_index_scale': 650e-9, # Estimated
        'k_index_filter': None,
        'copyright': 'Steve Marple.',
        'license': cc3_by_nc_sa,
        'attribution':  'Space and Plasma Physics group, ' + \
            'Department of Physics, Lancaster University, UK.',
        'line_color': [1, 0, 0],
        }, # BGS3

    'BGS4': {
        'location': 'Lancaster, UK',
        'latitude': 54.0,
        'longitude': -2.78,
        'elevation': 27,
        'start_time': np.datetime64('2015-10-19T00:00Z'),
        'end_time': None, # Still operational
        'k_index_scale': 650e-9, # Estimated
        'k_index_filter': None,
        'copyright': 'Steve Marple.',
        'license': cc3_by_nc_sa,
        'attribution': 'Data provided by Steve Marple.', 
        'line_color': [0, 0.6, 0],
        }, # BGS4

    'BGS5': {
        'location': 'Lancaster, UK',
        'latitude': 54.01,
        'longitude': -2.77,
        'elevation': 93,
        'start_time': np.datetime64('2015-10-19T00:00Z'),
        'end_time': None, # Still operational
        'k_index_scale': 650e-9, # Estimated
        'k_index_filter': None,
        'copyright': 'Lancaster University.',
        'license': cc3_by_nc_sa,
        'attribution':  'Space and Plasma Physics group, ' + \
            'Department of Physics, Lancaster University, UK.',
        'line_color': [0x7b/255., 0x03/255., 0x48/255.],
        }, # BGS5

    'BGS9': {
        'location': 'Edinburgh, UK',
        'latitude': 55.952,
        'longitude': -3.190,
        'elevation': np.nan,
        'start_time': np.datetime64('2016-01-01T00:00Z'),
        'end_time': None, # Still operational
        'k_index_scale': 750e-9, # Estimated
        'k_index_filter': None,
        'copyright': 'Lancaster University.',
        'license': cc3_by_nc_sa,
        'attribution':  'British Geological Survey.',
        'line_color': [0x7b/255., 0x03/255., 0x48/255.],
        }, # BGS9

    'BGS10': {
        'location': 'Edinburgh, UK',
        'latitude': 55.952,
        'longitude': -3.190,
        'elevation': np.nan,
        'start_time': np.datetime64('2016-01-01T00:00Z'),
        'end_time': None, # Still operational
        'k_index_scale': 750e-9, # Estimated
        'k_index_filter': None,
        'copyright': 'Lancaster University.',
        'license': cc3_by_nc_sa,
        'attribution':  'British Geological Survey.',
        'line_color': [0x7b/255., 0x03/255., 0x48/255.],
        }, # BGS10


    }


# Default values for all sites
defaults = {
    'activity_thresholds': np.array([0.0, 50.0, 100.0, 200.0]) * 1e-9,
    'activity_colors':  np.array([[0.2, 1.0, 0.2],  # green  
                                  [1.0, 1.0, 0.0],  # yellow
                                  [1.0, 0.6, 0.0],  # amber
                                  [1.0, 0.0, 0.0]]), # red
    'copyright': 'Copyright ???',
    'license': cc3_by_nc_sa,
    }

default_data_types = {
    'MagData': {
        'default': 'realtime',
        'realtime': {
            'channels': np.array(['H', 'E', 'Z']),
            'path': data_dir + '/{site_lc}/%Y/%m/{site_lc}_%Y%m%d.csv',
            'duration': np.timedelta64(24, 'h'),
            'format': 'aurorawatchnet',
            'load_converter': load_bgs_sch_data,
            'nominal_cadence': np.timedelta64(5, 's'),
            'units': 'T',
            },
        },
    'MagQDC': {
        'qdc': {
            'channels': np.array(['H', 'E', 'Z']),
            'path': data_dir + '/{site_lc}/qdc/%Y/{site_lc}_qdc_%Y%m.csv',
            'duration': np.timedelta64(24, 'h'),
            'format': 'aurorawatchnet_qdc',
            'load_converter': ap.magdata.load_qdc_data,
            'nominal_cadence': np.timedelta64(5, 's'),
            'units': 'T',
            },
        },
    'TemperatureData': {
        'realtime': {
            'channels': np.array(['Sensor temperature']),
            'path': data_dir + '/{site_lc}/%Y/%m/{site_lc}_%Y%m%d.csv',
            'duration': np.timedelta64(24, 'h'),
            'format': 'aurorawatchnet',
            'load_converter': load_bgs_sch_data,
            'nominal_cadence': np.timedelta64(5, 's'),
            'units': six.u('\N{DEGREE SIGN}C'),
            },
        },
    }

for s in sites:
    sd = sites[s]
    for key,val in defaults.items():
        if key not in sd:
            sd[key] = val
            
    # Populate the data types
    if 'data_types' not in sd:
        sd['data_types'] = {}
    for dt in default_data_types:
        if dt not in sd['data_types']:
            sd['data_types'][dt] = copy.deepcopy(default_data_types[dt])
            for archive in sd['data_types'][dt]:
                if archive == 'default':
                    continue
                sd['data_types'][dt][archive]['path'] = \
                    sd['data_types'][dt][archive]['path'].format(site_lc=s.lower())
        elif sd['data_types'][dt] is None:
            del sd['data_types'][dt]

ap.add_project('BGS_SCH', sites)

