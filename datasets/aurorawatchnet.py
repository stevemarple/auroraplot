import os
import numpy as np
import urllib2
import auroraplot as ap
from auroraplot.magdata import MagData
from auroraplot.magdata import MagQDC
from auroraplot.temperaturedata import TemperatureData
from auroraplot.voltagedata import VoltageData

data_dir = '/data/aurorawatch/net'

def check_mag_data(data):
    data[np.logical_or(data < -0.0001, data > 0.0001)] = np.nan
    return data

def check_temperature(data):
    data[np.logical_or(data < -40, data > 100)] = np.nan
    return data

def check_voltage(data):
    data[np.logical_or(data < 0, data > 10)] = np.nan
    return data

def convert_awn_data(file_name, archive_data, 
                     network, site, data_type, channels, start_time, 
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
    
    assert data_type_info.has_key(data_type), 'Illegal data_type'
    chan_tup = tuple(archive_data['channels'])
    col_idx = []
    for c in channels:
        col_idx.append(data_type_info[data_type]['col_offset'] + 
                       chan_tup.index(c))
    try:
        if file_name.startswith('/'):
            uh = urllib2.urlopen('file:' + file_name)
        else:
            uh = urllib2.urlopen(file_name)
        try:
            data = np.loadtxt(uh, unpack=True)
            sample_start_time = ap.epoch64_us + \
                (np.timedelta64(1, 's') * data[0])
            # end time and integration interval are guesstimates
            sample_end_time = sample_start_time + np.timedelta64(1, 's')
            integration_interval = np.ones([len(channels), 
                                            len(sample_start_time)],
                                            dtype='m8[s]')
            data = data[col_idx] * data_type_info[data_type]['scaling']
            if data_type_info[data_type]['data_check']:
                data = data_type_info[data_type]['data_check'](data)
            r = data_type_info[data_type]['class']( \
                network=network,
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
            if kwargs.get('verbose'):
                print('Could not read ' + file_name)
                # print(str(e))

        finally:
            uh.close()
    except Exception as e:
        if kwargs.get('verbose'):
            print('Could not open ' + file_name)
            # print(str(e))
    return None


def convert_awn_qdc_data(file_name, archive_data, 
                         network, site, data_type, channels, start_time, 
                         end_time, **kwargs):
    '''Convert AuroraWatchNet data to match standard data type

    archive: name of archive from which data was loaded
    archive_info: archive metadata
    '''

    assert data_type == 'MagQDC', 'Illegal data_type'
    chan_tup = tuple(archive_data['channels'])
    col_idx = []
    for c in channels:
        col_idx.append(chan_tup.index(c) + 1)
    try:
        if file_name.startswith('/'):
            uh = urllib2.urlopen('file:' + file_name)
        else:
            uh = urllib2.urlopen(file_name)
        try:
            data = np.loadtxt(uh, unpack=True)
            sample_start_time = (np.timedelta64(1, 's') * data[0])
            sample_end_time = sample_start_time \
                + archive_data['nominal_cadence']
            
            integration_interval = None
            data = data[col_idx] * 1e-9 # Stored as nT
            r = MagQDC(network=network,
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
            if kwargs.get('verbose'):
                print('Could not read ' + file_name)
                print(str(e))

        finally:
            uh.close()
    except Exception as e:
        if kwargs.get('verbose'):
            print('Could not open ' + file_name)
            print(str(e))
    return None
    

sites = {
    'LAN1': {
        'location': 'Lancaster, UK',
        'latitude': 54.0,
        'longitude': -2.78,
        'elevation': 27,
        'data_types': {
            'MagData': {
                'realtime': {
                    'channels': np.array(['H']),
                    'path': os.path.join(data_dir,
                                         'lan1/%Y/%m/lan1_%Y%m%d.txt'),
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'converter': convert_awn_data,
                    'nominal_cadence': np.timedelta64(30, 's'),
                    'units': 'T',
                    },
                },
            'MagQDC': {
                'qdc': {
                    'channels': np.array(['H']),
                    'path': os.path.join(data_dir, 
                                         'lan1/qdc/%Y/lan1_qdc_%Y%m.txt'),
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet_qdc',
                    'converter': convert_awn_qdc_data,
                    'nominal_cadence': np.timedelta64(5, 's'),
                    'units': 'T',
                    },
                },
            'TemperatureData': {
                'realtime': {
                    'channels': np.array(['Sensor temperature', 
                                          'System temperature']),
                    'path': os.path.join(data_dir,
                                         'lan1/%Y/%m/lan1_%Y%m%d.txt'),
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'converter': convert_awn_data,
                    'nominal_cadence': np.timedelta64(30, 's'),
                    'units': u'\N{DEGREE SIGN}C',
                    #'units': 'C',
                    },
                },
            'VoltageData': {
                'realtime': {
                    'channels': np.array(['Battery voltage']),
                    'path': os.path.join(data_dir,
                                         'lan1/%Y/%m/lan1_%Y%m%d.txt'),
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'converter': convert_awn_data,
                    'nominal_cadence': np.timedelta64(30, 's'),
                    'units': 'V',
                    },
                },        
            },
        'start_time': np.datetime64('2012-12-12T17:29Z'),
        'end_time': None, # Still operational
        'acknowledgement': {'short': 'Steve Marple.'},
        },
    'LAN3': {
        'location': 'Lancaster, UK',
        'latitude': 54.01,
        'longitude': -2.77,
        'elevation': 93,
        'data_types': {
            'MagData': {
                'realtime': {
                    'channels': np.array(['H']),
                    'path': os.path.join(data_dir,
                                         'lan3/%Y/%m/lan3_%Y%m%d.txt'),
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'converter': convert_awn_data,
                    'nominal_cadence': np.timedelta64(30, 's'),
                    'units': 'T',
                    },
                },
            'MagQDC': {
                'qdc': {
                    'channels': np.array(['H']),
                    'path': os.path.join(data_dir, 
                                         'lan3/qdc/%Y/lan3_qdc_%Y%m.txt'),
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet_qdc',
                    'converter': convert_awn_qdc_data,
                    'nominal_cadence': np.timedelta64(5, 's'),
                    'units': 'T',
                    },
                },
            'TemperatureData': {
                'realtime': {
                    'channels': np.array(['Sensor temperature', 
                                          'System temperature']),
                    'path': os.path.join(data_dir,
                                         'lan3/%Y/%m/lan3_%Y%m%d.txt'),
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'converter': convert_awn_data,
                    'nominal_cadence': np.timedelta64(30, 's'),
                    'units': u'\N{DEGREE SIGN}C',
                    # 'units': 'C',
                    },
                },
            'VoltageData': {
                'realtime': {
                    'channels': np.array(['Battery voltage']),
                    'path': os.path.join(data_dir,
                                         'lan3/%Y/%m/lan3_%Y%m%d.txt'),
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'converter': convert_awn_data,
                    'nominal_cadence': np.timedelta64(30, 's'),
                    'units': 'V',
                    },
                },        
            },
        'start_time': np.datetime64('2012-12-18T00:00Z'),
        'end_time': None, # Still operational
        'acknowledgement': {'short': 'Lancaster University.'},
        },
    'METOFFICE1': {
        'location': 'Ormskirk, UK',
        'latitude': 53.569195, 
        'longitude': -2.887264,
        'elevation': None,
        'data_types': {
            'MagData': {
                'realtime': {
                    'channels': np.array(['H']),
                    'path': os.path.join(data_dir,
                                         'metoffice1/%Y/%m/metoffice1_%Y%m%d.txt'),
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'converter': convert_awn_data,
                    'nominal_cadence': np.timedelta64(30, 's'),
                    'units': 'T',
                    },
                },
            'MagQDC': {
                'qdc': {
                    'channels': np.array(['H']),
                    'path': os.path.join(data_dir, 
                                         'metoffice1/qdc/%Y/metoffice1_qdc_%Y%m.txt'),
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet_qdc',
                    'converter': convert_awn_qdc_data,
                    'nominal_cadence': np.timedelta64(5, 's'),
                    'units': 'T',
                    },
                },
            'TemperatureData': {
                'realtime': {
                    'channels': np.array(['Sensor temperature', 
                                          'System temperature']),
                    'path': os.path.join(data_dir,
                                         'metoffice1/%Y/%m/metoffice1_%Y%m%d.txt'),
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'converter': convert_awn_data,
                    'nominal_cadence': np.timedelta64(30, 's'),
                    'units': u'\N{DEGREE SIGN}C',
                    # 'units': 'C',
                    },
                },
            'VoltageData': {
                'realtime': {
                    'channels': np.array(['Battery voltage']),
                    'path': os.path.join(data_dir,
                                         'metoffice1/%Y/%m/metoffice1_%Y%m%d.txt'),
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'converter': convert_awn_data,
                    'nominal_cadence': np.timedelta64(30, 's'),
                    'units': 'V',
                    },
                },        
            },
        'start_time': np.datetime64('2013-08-01T00:00Z'),
        'end_time': None, # Still operational
        'acknowledgement': {'short': 'Lancaster University.'},
        },
    }

# Set activity color/thresholds unless already set.
default_activity_thresholds = np.array([0.0, 50.0, 100.0, 200.0]) * 1e-9
default_activity_colors = np.array([[0.2, 1.0, 0.2],  # green  
                                    [1.0, 1.0, 0.0],  # yellow
                                    [1.0, 0.6, 0.0],  # amber
                                    [1.0, 0.0, 0.0]]) # red
for s in sites:
    if not sites[s].has_key('activity_thresholds'):
        sites[s]['activity_thresholds'] = default_activity_thresholds
    if not sites[s].has_key('activity_colors'):
        sites[s]['activity_colors'] = default_activity_colors
    
ap.add_network('AURORAWATCHNET', sites)

