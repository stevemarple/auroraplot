import copy
from decimal import Decimal
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

def load_bgs_sch_data(file_name,
                      archive_data,
                      project,
                      site,
                      data_type,
                      channels,
                      start_time,
                      end_time,
                      invalid_raise=False,
                      **kwargs):
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
            kw = {}
            if file_name.endswith('.csv'):
                kw['delimiter'] = ','

            data = np.genfromtxt(uh, 
                                 unpack=True, 
                                 invalid_raise=invalid_raise, 
                                 **kw)
            sample_start_time = ap.epoch64_us + \
                (np.timedelta64(1000000, 'us') * data[0])
            # end time and integration interval are guesstimates
            sample_end_time = sample_start_time + np.timedelta64(1000000, 'us')
            integration_interval = np.ones([len(channels), 
                                            len(sample_start_time)],
                                            dtype='m8[us]')
            data = data[col_idx] * data_type_info[data_type]['scaling']
            if data_type == 'MagData' and archive_data.get('swap_H_E'):
                data[[0,1]] = data[[1,0]]
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


def remove_spikes(md, **kwargs):
    return md.remove_spikes_chauvenet(savgol_window=np.timedelta64(5, 'm'),
                                      chauvenet_window=np.array([89,79]).astype('timedelta64[s]'))

def temperature_compensation(md, inplace=False, cadence=None, **kwargs):
    # Set cadence to reduce noise
    cad = np.timedelta64(2, 'm')
    if cadence and cadence > cad:
        cad = cadence
    td = ap.load_data(md.project,
                      md.site,
                      'TemperatureData',
                      md.start_time,
                      md.end_time,
                      channels=['Sensor temperature'], 
                      cadence=cad)
    # Resample
    td = td.interp(md.sample_start_time, md.sample_end_time)
    mean_temp = 20
    temp_diff = td.data - mean_temp
    coeffs = [0, -25e-9, -5e-9]
    # coeffs = [0, -2.15532366e-9, -0.49559117e-9]
    md_error = np.zeros([1, md.data.shape[1]])
    # md_error += coeffs[0] # Offset
    # for n in range(1, len(coeffs)):
    for n in range(len(coeffs)):
        md_error += np.power(temp_diff, n) * coeffs[n]
        
    md_error[np.isnan(md_error)] = 0
    if inplace:
        r = md
    else:
        r = copy.deepcopy(md)
    print(md_error)
    r.data -= md_error
    return r

cc3_by_nc_sa = 'This work is licensed under the Creative Commons ' + \
    'Attribution-NonCommercial-ShareAlike 3.0 Unported License. ' + \
    'To view a copy of this license, visit ' + \
    'http://creativecommons.org/licenses/by-nc-sa/3.0/deed.en_GB.'

sites = {
    'BGS3': {
        'location': 'Lancaster, UK',
        'latitude': Decimal('54.01'),
        'longitude': Decimal('-2.78'),
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

    'LAN1': { # Formerly BGS4
        'location': 'Lancaster, UK',
        'latitude': Decimal('54.0'),
        'longitude': Decimal('-2.78'),
        'elevation': 27,
        'start_time': np.datetime64('2015-10-19T00:00Z'),
        'end_time': None, # Still operational
        'k_index_scale': 650e-9, # Estimated
        'k_index_filter': None,
        'copyright': 'Steve Marple/British Geological Survey.',
        'license': cc3_by_nc_sa,
        'attribution': 'Operated by Steve Marple.', 
        'line_color': [0, 0.6, 0],
        'data_types': {
            'MagData': {
                'default': 'realtime',
                'raw': {
                    'channels': np.array(['H', 'E', 'Z']),
                    'path': data_dir + '/lan1/%Y/%m/lan1_%Y%m%d.csv',
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'load_converter': load_bgs_sch_data,
                    'nominal_cadence': np.timedelta64(10, 's'),
                    'units': 'T',
                    },
                'realtime': {
                    'channels': np.array(['H', 'E', 'Z']),
                    'path': data_dir + '/lan1/%Y/%m/lan1_%Y%m%d.csv',
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'load_converter': load_bgs_sch_data,
                    'nominal_cadence': np.timedelta64(10, 's'),
                    'units': 'T',
                    'filter_function': remove_spikes,
                    },
                'realtime_baseline': {
                    'channels': np.array(['H', 'E', 'Z']),
                    'path': (data_dir +
                             '/baseline/realtime/lan1/lan1_%Y.txt'),
                    'duration': np.timedelta64(1, 'Y'),
                    'load_converter': ap.data._generic_load_converter,
                    'save_converter': ap.data._generic_save_converter,
                    'nominal_cadence': np.timedelta64(1, 'D'),
                    'units': 'T',
                    # Information for generic load/save 
                    'constructor': ap.magdata.MagData,
                    'sort': False,
                    'timestamp_method': 'YMD',
                    'fmt': ['%04d', '%02d', '%02d', '%.2f', '%.2f', '%.2f'],
                    'data_multiplier': 1000000000, # Store as nT values
                    # Information for making the data files
                    'qdc_fit_duration': np.timedelta64(10, 'D'),
                    'realtime_qdc': True,
                    },
                },
            'MagQDC': {
                'qdc': {
                    'channels': np.array(['H', 'E', 'Z']),
                    'path': data_dir + '/lan1/qdc/%Y/lan1_qdc_%Y%m.csv',
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet_qdc',
                    'load_converter': ap.magdata.load_qdc_data,
                    'nominal_cadence': np.timedelta64(10, 's'),
                    'units': 'T',
                    },
                },
            'TemperatureData': {
                'realtime': {
                    'channels': np.array(['Sensor temperature']),
                    'path': data_dir + '/lan1/%Y/%m/lan1_%Y%m%d.csv',
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'load_converter': load_bgs_sch_data,
                    'nominal_cadence': np.timedelta64(10, 's'),
                    'units': six.u('\N{DEGREE SIGN}C'),
                    },
                },
            },
        }, # LAN1

    'LAN2': { # Formerly BGS5
        'location': 'Lancaster, UK',
        'latitude': Decimal('54.01'),
        'longitude': Decimal('-2.77'),
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
        'data_types': {
            'MagData': {
                'default': 'realtime',
                'raw': {
                    'channels': np.array(['H', 'E', 'Z']),
                    'path': data_dir + '/lan2/%Y/%m/lan2_%Y%m%d.csv',
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'load_converter': load_bgs_sch_data,
                    'nominal_cadence': np.timedelta64(10, 's'),
                    'units': 'T',
                    },
                'realtime': {
                    'channels': np.array(['H', 'E', 'Z']),
                    'path': data_dir + '/lan2/%Y/%m/lan2_%Y%m%d.csv',
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'load_converter': load_bgs_sch_data,
                    'nominal_cadence': np.timedelta64(10, 's'),
                    'units': 'T',
                    'filter_function': remove_spikes,
                    },
                'realtime_baseline': {
                    'channels': np.array(['H', 'E', 'Z']),
                    'path': (data_dir +
                             '/baseline/realtime/lan2/lan2_%Y.txt'),
                    'duration': np.timedelta64(1, 'Y'),
                    'load_converter': ap.data._generic_load_converter,
                    'save_converter': ap.data._generic_save_converter,
                    'nominal_cadence': np.timedelta64(1, 'D'),
                    'units': 'T',
                    # Information for generic load/save 
                    'constructor': ap.magdata.MagData,
                    'sort': False,
                    'timestamp_method': 'YMD',
                    'fmt': ['%04d', '%02d', '%02d', '%.2f', '%.2f', '%.2f'],
                    'data_multiplier': 1000000000, # Store as nT values
                    # Information for making the data files
                    'qdc_fit_duration': np.timedelta64(10, 'D'),
                    'realtime_qdc': True,
                    },
                },
            'MagQDC': {
                'qdc': {
                    'channels': np.array(['H', 'E', 'Z']),
                    'path': data_dir + '/lan2/qdc/%Y/lan2_qdc_%Y%m.csv',
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet_qdc',
                    'load_converter': ap.magdata.load_qdc_data,
                    'nominal_cadence': np.timedelta64(10, 's'),
                    'units': 'T',
                    },
                },
            'TemperatureData': {
                'realtime': {
                    'channels': np.array(['Sensor temperature']),
                    'path': data_dir + '/lan2/%Y/%m/lan2_%Y%m%d.csv',
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'load_converter': load_bgs_sch_data,
                    'nominal_cadence': np.timedelta64(10, 's'),
                    'units': six.u('\N{DEGREE SIGN}C'),
                    },
                },
            },
        }, # LAN2

    'BRO': { # Formerly BGS8
        'location': 'Broxburn, UK',
        'latitude': Decimal('55.94'),
        'longitude': Decimal('-3.47'),
        'elevation': 245,
        'start_time': np.datetime64('2016-06-02T00:00Z'),
        'end_time': None, # Still operational
        'k_index_scale': 750e-9, # Estimated
        'k_index_filter': None,
        'copyright': 'British Geological Survey.',
        'license': cc3_by_nc_sa,
        'attribution':  'British Geological Survey.',
        'line_color': [0x07/255., 0x0e/255., 0x68/255.],
        'url': 'https://blogs.glowscotland.org.uk/wl/broxburnps/',
        'data_types': {
            'MagData': {
                'default': 'realtime',
                'raw': {
                    'channels': np.array(['H', 'E', 'Z']),
                    'path': data_dir + '/bgs8/%Y/%m/bgs8_%Y%m%d.csv',
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'load_converter': load_bgs_sch_data,
                    'nominal_cadence': np.timedelta64(10, 's'),
                    'units': 'T',
                    },
                'realtime': {
                    'channels': np.array(['H', 'E', 'Z']),
                    'path': data_dir + '/bgs8/%Y/%m/bgs8_%Y%m%d.csv',
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'load_converter': load_bgs_sch_data,
                    'nominal_cadence': np.timedelta64(10, 's'),
                    'units': 'T',
                    'filter_function': remove_spikes,
                    },
                'realtime_baseline': {
                    'channels': np.array(['H', 'E', 'Z']),
                    'path': (data_dir +
                             '/baseline/realtime/bro/bro_%Y.txt'),
                    'duration': np.timedelta64(1, 'Y'),
                    'load_converter': ap.data._generic_load_converter,
                    'save_converter': ap.data._generic_save_converter,
                    'nominal_cadence': np.timedelta64(1, 'D'),
                    'units': 'T',
                    # Information for generic load/save 
                    'constructor': ap.magdata.MagData,
                    'sort': False,
                    'timestamp_method': 'YMD',
                    'fmt': ['%04d', '%02d', '%02d', '%.2f', '%.2f', '%.2f'],
                    'data_multiplier': 1000000000, # Store as nT values
                    # Information for making the data files
                    'qdc_fit_duration': np.timedelta64(10, 'D'),
                    'realtime_qdc': True,
                    },
                },
            'MagQDC': {
                'qdc': {
                    'channels': np.array(['H', 'E', 'Z']),
                    'path': data_dir + '/bgs8/qdc/%Y/bgs8_qdc_%Y%m.csv',
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet_qdc',
                    'load_converter': ap.magdata.load_qdc_data,
                    'nominal_cadence': np.timedelta64(10, 's'),
                    'units': 'T',
                    },
                },
            'TemperatureData': {
                'realtime': {
                    'channels': np.array(['Sensor temperature']),
                    'path': data_dir + '/bgs8/%Y/%m/bgs8_%Y%m%d.csv',
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'load_converter': load_bgs_sch_data,
                    'nominal_cadence': np.timedelta64(10, 's'),
                    'units': six.u('\N{DEGREE SIGN}C'),
                    },
                },
            },
        }, # BRX

    'NOR': { # Previously BGS9
        'location': 'Norwich, UK',
        'latitude': Decimal('52.6316'),
        'longitude': Decimal('1.30042'),
        'elevation': np.nan,
        'start_time': np.datetime64('2016-08-16T12:30Z'),
        'end_time': None, # Still operational
        'k_index_scale': 580e-9, # Estimated
        'k_index_filter': None,
        'copyright': 'British Geological Survey.',
        'license': cc3_by_nc_sa,
        'attribution':  'British Geological Survey.',
        'line_color': [0x70/255., 0x00/255., 0x3b/255.],
        'url': 'http://www.norwich-school.org.uk/',
        'data_types': {
            'MagData': {
                'default': 'realtime',
                'raw': {
                    'channels': np.array(['H', 'E', 'Z']),
                    'path': data_dir + '/nor/%Y/%m/bgs9_%Y%m%d.csv',
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'load_converter': load_bgs_sch_data,
                    'nominal_cadence': np.timedelta64(10, 's'),
                    'units': 'T',
                    },
                'realtime': {
                    'channels': np.array(['H', 'E', 'Z']),
                    'path': data_dir + '/nor/%Y/%m/bgs9_%Y%m%d.csv',
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'load_converter': load_bgs_sch_data,
                    'nominal_cadence': np.timedelta64(10, 's'),
                    'units': 'T',
                    'filter_function': remove_spikes,
                    'swap_H_E': True,
                    },
                'realtime_baseline': {
                    'channels': np.array(['H', 'E', 'Z']),
                    'path': (data_dir +
                             '/baseline/realtime/nor/nor_%Y.txt'),
                    'duration': np.timedelta64(1, 'Y'),
                    'load_converter': ap.data._generic_load_converter,
                    'save_converter': ap.data._generic_save_converter,
                    'nominal_cadence': np.timedelta64(1, 'D'),
                    'units': 'T',
                    # Information for generic load/save 
                    'constructor': ap.magdata.MagData,
                    'sort': False,
                    'timestamp_method': 'YMD',
                    'fmt': ['%04d', '%02d', '%02d', '%.2f', '%.2f', '%.2f'],
                    'data_multiplier': 1000000000, # Store as nT values
                    # Information for making the data files
                    'qdc_fit_duration': np.timedelta64(10, 'D'),
                    'realtime_qdc': True,
                    },
                },
            'MagQDC': {
                'qdc': {
                    'channels': np.array(['H', 'E', 'Z']),
                    'path': data_dir + '/nor/qdc/%Y/bgs9_qdc_%Y%m.csv',
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet_qdc',
                    'load_converter': ap.magdata.load_qdc_data,
                    'nominal_cadence': np.timedelta64(10, 's'),
                    'units': 'T',
                    },
                },
            'TemperatureData': {
                'realtime': {
                    'channels': np.array(['Sensor temperature']),
                    'path': data_dir + '/nor/%Y/%m/bgs9_%Y%m%d.csv',
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'load_converter': load_bgs_sch_data,
                    'nominal_cadence': np.timedelta64(10, 's'),
                    'units': six.u('\N{DEGREE SIGN}C'),
                    },
                },
            },
        }, # NOR

    'OUN': { # Previously BGS10
        'location': 'Oundle, UK',
        'latitude': Decimal('52.4809'), 
        'longitude': Decimal('-0.46904'),
        'elevation': np.nan,
        'start_time': np.datetime64('2016-08-15T15:30Z'),
        'end_time': None, # Still operational
        'k_index_scale': 560e-9, # Estimated
        'k_index_filter': None,
        'copyright': 'British Geological Survey.',
        'license': cc3_by_nc_sa,
        'attribution':  'British Geological Survey.',
        'line_color': [0x05/255., 0x31/255., 0x61/255.],
        'url': 'http://www.oundleschool.org.uk/',
        'data_types': {
            'MagData': {
                'default': 'realtime',
                'raw': {
                    'channels': np.array(['H', 'E', 'Z']),
                    'path': data_dir + '/oun/%Y/%m/bgs10_%Y%m%d.csv',
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'load_converter': load_bgs_sch_data,
                    'nominal_cadence': np.timedelta64(10, 's'),
                    'units': 'T',
                    },
                'realtime': {
                    'channels': np.array(['H', 'E', 'Z']),
                    'path': data_dir + '/oun/%Y/%m/bgs10_%Y%m%d.csv',
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'load_converter': load_bgs_sch_data,
                    'nominal_cadence': np.timedelta64(10, 's'),
                    'units': 'T',
                    'filter_function': remove_spikes,
                    },
                'realtime_baseline': {
                    'channels': np.array(['H', 'E', 'Z']),
                    'path': (data_dir +
                             '/baseline/realtime/oun/oun_%Y.txt'),
                    'duration': np.timedelta64(1, 'Y'),
                    'load_converter': ap.data._generic_load_converter,
                    'save_converter': ap.data._generic_save_converter,
                    'nominal_cadence': np.timedelta64(1, 'D'),
                    'units': 'T',
                    # Information for generic load/save 
                    'constructor': ap.magdata.MagData,
                    'sort': False,
                    'timestamp_method': 'YMD',
                    'fmt': ['%04d', '%02d', '%02d', '%.2f', '%.2f', '%.2f'],
                    'data_multiplier': 1000000000, # Store as nT values
                    # Information for making the data files
                    'qdc_fit_duration': np.timedelta64(10, 'D'),
                    'realtime_qdc': True,
                    },
                },
            'MagQDC': {
                'qdc': {
                    'channels': np.array(['H', 'E', 'Z']),
                    'path': data_dir + '/oun/qdc/%Y/bgs10_qdc_%Y%m.csv',
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet_qdc',
                    'load_converter': ap.magdata.load_qdc_data,
                    'nominal_cadence': np.timedelta64(10, 's'),
                    'units': 'T',
                    },
                },
            'TemperatureData': {
                'realtime': {
                    'channels': np.array(['Sensor temperature']),
                    'path': data_dir + '/oun/%Y/%m/bgs10_%Y%m%d.csv',
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'load_converter': load_bgs_sch_data,
                    'nominal_cadence': np.timedelta64(10, 's'),
                    'units': six.u('\N{DEGREE SIGN}C'),
                    },
                },
            },
        }, # OUN

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
        'raw': {
            'channels': np.array(['H', 'E', 'Z']),
            'path': data_dir + '/{site_lc}/%Y/%m/{site_lc}_%Y%m%d.csv',
            'duration': np.timedelta64(24, 'h'),
            'format': 'aurorawatchnet',
            'load_converter': load_bgs_sch_data,
            'nominal_cadence': np.timedelta64(5, 's'),
            'units': 'T',
            },
        'realtime': {
            'channels': np.array(['H', 'E', 'Z']),
            'path': data_dir + '/{site_lc}/%Y/%m/{site_lc}_%Y%m%d.csv',
            'duration': np.timedelta64(24, 'h'),
            'format': 'aurorawatchnet',
            'load_converter': load_bgs_sch_data,
            'nominal_cadence': np.timedelta64(5, 's'),
            'units': 'T',
            'filter_function': remove_spikes,
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

project = {
    'name': 'BGS Schools Magnetometer Network',
    'abbreviation': 'BGS_SCH',
    'url': 'http://aurorawatch.lancs.ac.uk/project-info/bgs-sch/',
    'sites': sites,
}

ap.add_project('BGS_SCH', project)

