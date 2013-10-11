import os
import logging
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
            logging.info('Could not read ' + file_name)
            logging.debug(str(e))

        finally:
            uh.close()
    except Exception as e:
        logging.info('Could not open ' + file_name)
        logging.debug(str(e))

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
            logging.info('Could not read ' + file_name)
            logging.debug(str(e))

        finally:
            uh.close()
    except Exception as e:
        logging.info('Could not open ' + file_name)
        logging.debug(str(e))
    return None
    


cc3_by_nc_sa = 'This work is licensed under the Creative Commons ' + \
    'Attribution-NonCommercial-ShareAlike 3.0 Unported License. ' + \
    'To view a copy of this license, visit ' + \
    'http://creativecommons.org/licenses/by-nc-sa/3.0/deed.en_GB.'

sites = {
    'LAN1': {
        'location': 'Lancaster, UK',
        'latitude': 54.0,
        'longitude': -2.78,
        'elevation': 27,
        'start_time': np.datetime64('2012-12-12T17:29Z'),
        'end_time': None, # Still operational
        'copyright': 'Steve Marple.',
        'license': cc3_by_nc_sa,
        'attribution': 'Data provided by Steve Marple.', 
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
        }, # LAN1
    'LAN3': {
        'location': 'Lancaster, UK',
        'latitude': 54.01,
        'longitude': -2.77,
        'elevation': 93,
        'start_time': np.datetime64('2012-12-18T00:00Z'),
        'end_time': None, # Still operational
        'license': cc3_by_nc_sa,
        'copyright': 'Lancaster University.',
        'attribution': 'Space Plasma Environment and Radio Science group, ' + \
            'Department of Physics, Lancaster University, UK.',
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
        }, # LAN3
    'METOFFICE1': {
        'location': 'Ormskirk, UK',
        'latitude': 53.569195, 
        'longitude': -2.887264,
        'elevation': None,
        'start_time': np.datetime64('2013-08-01T00:00Z'),
        'end_time': None, # Still operational
        'license': cc3_by_nc_sa,
        'copyright': 'Lancaster University.',
        'attribution': 'Space Plasma Environment and Radio Science group, ' + \
            'Department of Physics, Lancaster University, UK.',
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
        }, # METOFFICE1
    'BRA1': {
        'location': 'Brae, Shetland, UK',
        'latitude': 60.395869,
        'longitude': -1.351124000000027,
        'elevation': 11,
        'start_time': np.datetime64('2014-01-01T00:00:00+0000'),
        'end_time': None, # Still operational
        'license': cc3_by_nc_sa,
        'copyright': 'Lancaster University.',
        'attribution': 'Space Plasma Environment and Radio Science group, ' + \
            'Department of Physics, Lancaster University, UK.',
        'data_types': {
            'MagData': {
                'realtime': {
                    'channels': np.array(['H']),
                    'path': os.path.join(data_dir,
                                         'bra1/%Y/%m/bra1_%Y%m%d.txt'),
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
                                         'bra1/qdc/%Y/bra1_qdc_%Y%m.txt'),
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
                                         'bra1/%Y/%m/bra1_%Y%m%d.txt'),
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'converter': convert_awn_data,
                    'nominal_cadence': np.timedelta64(30, 's'),
                    'units': u'\N{DEGREE SIGN}C',
                    },
                },
            'VoltageData': {
                'realtime': {
                    'channels': np.array(['Battery voltage']),
                    'path': os.path.join(data_dir,
                                         'bra1/%Y/%m/bra1_%Y%m%d.txt'),
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'converter': convert_awn_data,
                    'nominal_cadence': np.timedelta64(30, 's'),
                    'units': 'V',
                    },
                },        
            },
        }, # BRA1
    'SAN1': {
        'location': 'Sanday, UK',
        'latitude': 59.25110830191925,
        'longitude': -2.5873320735991,
        'elevation': 16.366,
        'start_time': np.datetime64('2014-01-01T00:00:00+0000'),
        'end_time': None, # Still operational
        'license': cc3_by_nc_sa,
        'copyright': 'Lancaster University.',
        'attribution': 'Space Plasma Environment and Radio Science group, ' + \
            'Department of Physics, Lancaster University, UK.',
        'data_types': {
            'MagData': {
                'realtime': {
                    'channels': np.array(['H']),
                    'path': os.path.join(data_dir,
                                         'san1/%Y/%m/san1_%Y%m%d.txt'),
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
                                         'san1/qdc/%Y/san1_qdc_%Y%m.txt'),
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
                                         'san1/%Y/%m/san1_%Y%m%d.txt'),
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'converter': convert_awn_data,
                    'nominal_cadence': np.timedelta64(30, 's'),
                    'units': u'\N{DEGREE SIGN}C',
                    },
                },
            'VoltageData': {
                'realtime': {
                    'channels': np.array(['Battery voltage']),
                    'path': os.path.join(data_dir,
                                         'san1/%Y/%m/san1_%Y%m%d.txt'),
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'converter': convert_awn_data,
                    'nominal_cadence': np.timedelta64(30, 's'),
                    'units': 'V',
                    },
                },        
            },
        }, # SAN1
    'TOB1': {
        'location': 'Tobermory, Mull, UK',
        'latitude': 56.62415194965067,
        'longitude': -6.068624798208475,
        'elevation': 43,
        'start_time': np.datetime64('2014-01-01T00:00:00+0000'),
        'end_time': None, # Still operational
        'license': cc3_by_nc_sa,
        'copyright': 'Lancaster University.',
        'attribution': 'Space Plasma Environment and Radio Science group, ' + \
            'Department of Physics, Lancaster University, UK.',
        'data_types': {
            'MagData': {
                'realtime': {
                    'channels': np.array(['H']),
                    'path': os.path.join(data_dir,
                                         'xxx1/%Y/%m/xxx1_%Y%m%d.txt'),
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
                                         'xxx1/qdc/%Y/xxx1_qdc_%Y%m.txt'),
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
                                         'xxx1/%Y/%m/xxx1_%Y%m%d.txt'),
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'converter': convert_awn_data,
                    'nominal_cadence': np.timedelta64(30, 's'),
                    'units': u'\N{DEGREE SIGN}C',
                    },
                },
            'VoltageData': {
                'realtime': {
                    'channels': np.array(['Battery voltage']),
                    'path': os.path.join(data_dir,
                                         'xxx1/%Y/%m/xxx1_%Y%m%d.txt'),
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'converter': convert_awn_data,
                    'nominal_cadence': np.timedelta64(30, 's'),
                    'units': 'V',
                    },
                },        
            },
        }, # TOB1
     'WHI1': {
        'location': 'Whitehaven, Cumbria, UK',
        'latitude': 54.543384,
        'longitude': -3.5610000000000355,
        'elevation': 132,
        'start_time': np.datetime64('2014-01-01T00:00:00+0000'),
        'end_time': None, # Still operational
        'license': cc3_by_nc_sa,
        'copyright': 'Lancaster University.',
        'attribution': 'Space Plasma Environment and Radio Science group, ' + \
            'Department of Physics, Lancaster University, UK.',
        'data_types': {
            'MagData': {
                'realtime': {
                    'channels': np.array(['H']),
                    'path': os.path.join(data_dir,
                                         'whi1/%Y/%m/whi1_%Y%m%d.txt'),
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
                                         'whi1/qdc/%Y/whi1_qdc_%Y%m.txt'),
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
                                         'whi1/%Y/%m/whi1_%Y%m%d.txt'),
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'converter': convert_awn_data,
                    'nominal_cadence': np.timedelta64(30, 's'),
                    'units': u'\N{DEGREE SIGN}C',
                    },
                },
            'VoltageData': {
                'realtime': {
                    'channels': np.array(['Battery voltage']),
                    'path': os.path.join(data_dir,
                                         'whi1/%Y/%m/whi1_%Y%m%d.txt'),
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'converter': convert_awn_data,
                    'nominal_cadence': np.timedelta64(30, 's'),
                    'units': 'V',
                    },
                },        
            },
        }, # WHI1
    'ALT1': {
        'location': 'Altrincham, UK',
        'latitude': 53.381988,
        'longitude': -2.3597310000000107,
        'elevation': 68,
        'start_time': np.datetime64('2014-01-01T00:00:00+0000'),
        'end_time': None, # Still operational
        'license': cc3_by_nc_sa,
        'copyright': 'Lancaster University.',
        'attribution': 'Space Plasma Environment and Radio Science group, ' + \
            'Department of Physics, Lancaster University, UK.',
        'data_types': {
            'MagData': {
                'realtime': {
                    'channels': np.array(['H']),
                    'path': os.path.join(data_dir,
                                         'alt1/%Y/%m/alt1_%Y%m%d.txt'),
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
                                         'alt1/qdc/%Y/alt1_qdc_%Y%m.txt'),
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
                                         'alt1/%Y/%m/alt1_%Y%m%d.txt'),
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'converter': convert_awn_data,
                    'nominal_cadence': np.timedelta64(30, 's'),
                    'units': u'\N{DEGREE SIGN}C',
                    },
                },
            'VoltageData': {
                'realtime': {
                    'channels': np.array(['Battery voltage']),
                    'path': os.path.join(data_dir,
                                         'alt1/%Y/%m/alt1_%Y%m%d.txt'),
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'converter': convert_awn_data,
                    'nominal_cadence': np.timedelta64(30, 's'),
                    'units': 'V',
                    },
                },        
            },
        }, # ALT1
    'MAL1': {
        'location': 'Malpas, UK',
        'latitude': 53.029658,
        'longitude': -2.760221999999999,
        'elevation': 122,
        'start_time': np.datetime64('2014-01-01T00:00:00+0000'),
        'end_time': None, # Still operational
        'license': cc3_by_nc_sa,
        'copyright': 'Lancaster University.',
        'attribution': 'Space Plasma Environment and Radio Science group, ' + \
            'Department of Physics, Lancaster University, UK.',
        'data_types': {
            'MagData': {
                'realtime': {
                    'channels': np.array(['H']),
                    'path': os.path.join(data_dir,
                                         'mal1/%Y/%m/mal1_%Y%m%d.txt'),
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
                                         'mal1/qdc/%Y/mal1_qdc_%Y%m.txt'),
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
                                         'mal1/%Y/%m/mal1_%Y%m%d.txt'),
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'converter': convert_awn_data,
                    'nominal_cadence': np.timedelta64(30, 's'),
                    'units': u'\N{DEGREE SIGN}C',
                    },
                },
            'VoltageData': {
                'realtime': {
                    'channels': np.array(['Battery voltage']),
                    'path': os.path.join(data_dir,
                                         'mal1/%Y/%m/mal1_%Y%m%d.txt'),
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'converter': convert_awn_data,
                    'nominal_cadence': np.timedelta64(30, 's'),
                    'units': 'V',
                    },
                },        
            },
        }, # MAL1
    'ASH1': {
        'location': 'Ashbourne, UK',
        'latitude': 53.021899,
        'longitude': -1.7287959999999885,
        'elevation': 157,
        'start_time': np.datetime64('2014-01-01T00:00:00+0000'),
        'end_time': None, # Still operational
        'license': cc3_by_nc_sa,
        'copyright': 'Lancaster University.',
        'attribution': 'Space Plasma Environment and Radio Science group, ' + \
            'Department of Physics, Lancaster University, UK.',
        'data_types': {
            'MagData': {
                'realtime': {
                    'channels': np.array(['H']),
                    'path': os.path.join(data_dir,
                                         'ash1/%Y/%m/ash1_%Y%m%d.txt'),
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
                                         'ash1/qdc/%Y/ash1_qdc_%Y%m.txt'),
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
                                         'ash1/%Y/%m/ash1_%Y%m%d.txt'),
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'converter': convert_awn_data,
                    'nominal_cadence': np.timedelta64(30, 's'),
                    'units': u'\N{DEGREE SIGN}C',
                    },
                },
            'VoltageData': {
                'realtime': {
                    'channels': np.array(['Battery voltage']),
                    'path': os.path.join(data_dir,
                                         'ash1/%Y/%m/ash1_%Y%m%d.txt'),
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'converter': convert_awn_data,
                    'nominal_cadence': np.timedelta64(30, 's'),
                    'units': 'V',
                    },
                },        
            },
        }, # ASH1
        'PEL1': {
        'location': 'Pelsall, UK',
        'latitude': 52.623806,
        'longitude': -1.9490200000000186,
        'elevation': 143,
        'start_time': np.datetime64('2014-01-01T00:00:00+0000'),
        'end_time': None, # Still operational
        'license': cc3_by_nc_sa,
        'copyright': 'Lancaster University.',
        'attribution': 'Space Plasma Environment and Radio Science group, ' + \
            'Department of Physics, Lancaster University, UK.',
        'data_types': {
            'MagData': {
                'realtime': {
                    'channels': np.array(['H']),
                    'path': os.path.join(data_dir,
                                         'pel1/%Y/%m/pel1_%Y%m%d.txt'),
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
                                         'pel1/qdc/%Y/pel1_qdc_%Y%m.txt'),
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
                                         'pel1/%Y/%m/pel1_%Y%m%d.txt'),
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'converter': convert_awn_data,
                    'nominal_cadence': np.timedelta64(30, 's'),
                    'units': u'\N{DEGREE SIGN}C',
                    },
                },
            'VoltageData': {
                'realtime': {
                    'channels': np.array(['Battery voltage']),
                    'path': os.path.join(data_dir,
                                         'pel1/%Y/%m/pel1_%Y%m%d.txt'),
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'converter': convert_awn_data,
                    'nominal_cadence': np.timedelta64(30, 's'),
                    'units': 'V',
                    },
                },        
            },
        }, # PEL1
    'BRE1': {
        'location': 'Brecon, UK',
        'latitude': 52.02851,
        'longitude': -3.2026879999999665,
        'elevation': 121,
        'start_time': np.datetime64('2014-01-01T00:00:00+0000'),
        'end_time': None, # Still operational
        'license': cc3_by_nc_sa,
        'copyright': 'Lancaster University.',
        'attribution': 'Space Plasma Environment and Radio Science group, ' + \
            'Department of Physics, Lancaster University, UK.',
        'data_types': {
            'MagData': {
                'realtime': {
                    'channels': np.array(['H']),
                    'path': os.path.join(data_dir,
                                         'bre1/%Y/%m/bre1_%Y%m%d.txt'),
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
                                         'bre1/qdc/%Y/bre1_qdc_%Y%m.txt'),
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
                                         'bre1/%Y/%m/bre1_%Y%m%d.txt'),
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'converter': convert_awn_data,
                    'nominal_cadence': np.timedelta64(30, 's'),
                    'units': u'\N{DEGREE SIGN}C',
                    },
                },
            'VoltageData': {
                'realtime': {
                    'channels': np.array(['Battery voltage']),
                    'path': os.path.join(data_dir,
                                         'bre1/%Y/%m/bre1_%Y%m%d.txt'),
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'converter': convert_awn_data,
                    'nominal_cadence': np.timedelta64(30, 's'),
                    'units': 'V',
                    },
                },        
            },
        }, # BRE1
    'CAN1': {
        'location': 'Canterbury, UK',
        'latitude': 51.260914,
        'longitude': 1.084820000000036,
        'elevation': 48,
        'start_time': np.datetime64('2014-01-01T00:00:00+0000'),
        'end_time': None, # Still operational
        'license': cc3_by_nc_sa,
        'copyright': 'Lancaster University.',
        'attribution': 'Space Plasma Environment and Radio Science group, ' + \
            'Department of Physics, Lancaster University, UK.',
        'data_types': {
            'MagData': {
                'realtime': {
                    'channels': np.array(['H']),
                    'path': os.path.join(data_dir,
                                         'can1/%Y/%m/can1_%Y%m%d.txt'),
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
                                         'can1/qdc/%Y/can1_qdc_%Y%m.txt'),
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
                                         'can1/%Y/%m/can1_%Y%m%d.txt'),
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'converter': convert_awn_data,
                    'nominal_cadence': np.timedelta64(30, 's'),
                    'units': u'\N{DEGREE SIGN}C',
                    },
                },
            'VoltageData': {
                'realtime': {
                    'channels': np.array(['Battery voltage']),
                    'path': os.path.join(data_dir,
                                         'can1/%Y/%m/can1_%Y%m%d.txt'),
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'converter': convert_awn_data,
                    'nominal_cadence': np.timedelta64(30, 's'),
                    'units': 'V',
                    },
                },        
            },
        }, # CAN1
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



