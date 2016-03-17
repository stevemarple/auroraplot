import copy
import logging
import os
import numpy as np
import urllib2
import auroraplot as ap
import auroraplot.dt64tools as dt64
from auroraplot.temperaturedata import TemperatureData
from auroraplot.voltagedata import VoltageData
from auroraplot.humiditydata import HumidityData

# Borrow load functions
import auroraplot.datasets.aurorawatchnet as awn

data_dir = '/data/aurorawatchnet'


def load_cloud_data(file_name, archive_data, 
                       project, site, data_type, channels, start_time, 
                       end_time, data_cols, **kwargs):
    # data_type_info = { }
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
                                            dtype='m8[us]')
            data = data[data_cols]
            # if data_type_info[data_type]['data_check']:
            #     data = data_type_info[data_type]['data_check'](data)
            r = globals()[data_type]( \
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
            logging.info('Could not read ' + file_name)
            logging.debug(str(e))

        finally:
            uh.close()

    except Exception as e:
        logging.info('Could not open ' + file_name)
        logging.debug(str(e))

    return None


def load_humidity_data(file_name, archive_data, 
                          project, site, data_type, channels, start_time, 
                          end_time, **kwargs):
    return load_cloud_data(file_name, archive_data, 
                              project, site, data_type, channels, 
                              start_time, end_time, 
                              data_cols=[4], **kwargs)

def load_temperature_data(file_name, archive_data, 
                             project, site, data_type, channels, start_time, 
                             end_time, **kwargs):
    awn_channels = np.array(['Sensor temperature', # Mag sensor
                             'System temperature'])
    extra_channels = np.array(['Detector temperature',
                               'Sky temperature',
                               'Ambient temperature'])
    chan_array = np.array(channels)
    cidx_a = []
    cidx_b = []
    for n in range(len(channels)):
        if channels[n] in awn_channels:
            cidx_a.append(n)
        elif channels[n] in extra_channels:
            cidx_b.append(n)


    assert len(cidx_a) + len(cidx_b) == chan_array.size
    

    if len(cidx_a):
        awn_file_name = file_name.replace('_cloud.txt', '.txt')
        a = load_cloud_data(awn_file_name,
                               archive_data, 
                               project, site, data_type, 
                               chan_array[cidx_a],
                               start_time, end_time, 
                               data_cols=[5], **kwargs)
    else:
        a = None
    
    if len(cidx_b):
        b = load_cloud_data(file_name, archive_data, 
                               project, site, data_type, 
                               chan_array[cidx_b],
                               start_time, end_time, 
                               data_cols=[1,2,3], **kwargs)
        if a is None:
            return b
        else:
            # Join a and b
            # assert 1 or a.start_time == b.start_time and \
            #     a.end_time == b.end_time and \
            #     np.all(a.sample_start_time == b.sample_end_time) and \
            #     np.all(a.sample_end_time == b.sample_end_time) and \
            #     a.nominal_cadence == b.nominal_cadence and \
            #     a.units == b.units
            r = copy.deepcopy(a)
            cidx_a.extend(cidx_b)
            r.channels = chan_array[cidx_a]

            # Bug in data recording process. Work around for now
            assert a.sample_start_time.size == b.sample_end_time.size, \
                'Different length data sets'
            
            # assert np.all(np.abs(a.sample_start_time - b.sample_start_time) \
            #                   <= np.timedelta64(1, 's'))
            # assert np.all(np.abs(a.sample_end_time - b.sample_end_time) \
            #                   <= np.timedelta64(1, 's'))
            # b.sample_start_time = a.sample_start_time
            # b.sample_end_time = a.sample_end_time
            # # -- end of hack --

            # # Find common sample times
            # common_sample_start_time = np.intersect1d(a.sample_start_time, 
            #                                           b.sample_start_time)

            # # Find where they are located in a and b
            # s_a_idx = np.nonzero(np.in1d(a.sample_start_time, 
            #                              common_sample_start_time))[0]
            # s_b_idx = np.nonzero(np.in1d(a.sample_start_time, 
            #                              common_sample_start_time))[0]
            # assert np.all(a.sample_end_time[s_a_idx] \
            #                   == b.sample_end_time[s_b_idx]), \
            #                   'Sample end times do not match'

            # ns = common_sample_start_time.size
            ns = a.sample_start_time.size
            
            r.data = np.zeros([a.channels.size + b.channels.size, ns])
            r_a_cidx = range(a.channels.size)
            r_b_cidx = range(a.channels.size, r.channels.size)
            integ_units = dt64.smallest_unit([dt64.get_units(a.integration_interval),
                                              dt64.get_units(b.integration_interval)])

            r.data[r_a_cidx] = a.data
            r.data[r_b_cidx] = b.data
            r.integration_interval = \
                np.zeros([r.channels.size, ns]).astype('m8[' + integ_units +
                                                       ']')
            r.integration_interval[r_a_cidx] = \
                dt64.dt64_to(a.integration_interval, integ_units)
            r.integration_interval[r_b_cidx] = \
                dt64.dt64_to(b.integration_interval, integ_units)
            return r            
    else:
        return a

sites = {
    'TEST2': {
        'location': 'Lancaster, UK',
        'latitude': 54.0,
        'longitude': -2.78,
        'elevation': 27,
        'data_types': {
            'TemperatureData': {
                'realtime': {
                    'channels': np.array(['System temperature',
                                          'Detector temperature',
                                          'Sky temperature',
                                          'Ambient temperature']),
                    'path': os.path.join(data_dir,
                                         'test2/%Y/%m/test2_%Y%m%d_cloud.txt'),
                    'duration': np.timedelta64(24, 'h'),
                    'load_converter': load_temperature_data,
                    'nominal_cadence': np.timedelta64(30, 's'),
                    'units': u'\N{DEGREE SIGN}C',
                    },
                },
            'VoltageData': {
                'realtime': {
                    'channels': np.array(['Battery voltage']),
                    'path': os.path.join(data_dir,
                                         'test2/%Y/%m/test2_%Y%m%d.txt'),
                    'duration': np.timedelta64(24, 'h'),
                    'load_converter': awn.load_awn_data,
                    'nominal_cadence': np.timedelta64(30, 's'),
                    'units': 'V',
                    },
                },        
            'HumidityData': {
                'realtime': {
                    'channels': np.array(['Relative humidity']),
                    'path': os.path.join(data_dir,
                                         'test2/%Y/%m/test2_%Y%m%d_cloud.txt'),
                    'duration': np.timedelta64(24, 'h'),
                    'load_converter': load_humidity_data,
                    'nominal_cadence': np.timedelta64(30, 's'),
                    'units': '%',
                    },
                },
            },
        'start_time': np.datetime64('2013-07-13T00:00Z'),
        'end_time': None, # Still operational
        'acknowledgement': {'short': 'Steve Marple.'},
        },
    
    'LAN4': {
        'location': 'Lancaster, UK',
        'latitude': 54.0,
        'longitude': -2.78,
        'elevation': 27,
        'data_types': {
            'TemperatureData': {
                'realtime': {
                    'channels': np.array(['System temperature',
                                          'Detector temperature',
                                          'Sky temperature',
                                          'Ambient temperature']),
                    'path': os.path.join(data_dir,
                                         'lan4/%Y/%m/lan4_%Y%m%d_cloud.txt'),
                    'duration': np.timedelta64(24, 'h'),
                    'load_converter': load_temperature_data,
                    'nominal_cadence': np.timedelta64(30, 's'),
                    'units': u'\N{DEGREE SIGN}C',
                    },
                },
            'VoltageData': {
                'realtime': {
                    'channels': np.array(['Battery voltage']),
                    'path': os.path.join(data_dir,
                                         'lan4/%Y/%m/lan4_%Y%m%d.txt'),
                    'duration': np.timedelta64(24, 'h'),
                    'load_converter': awn.load_awn_data,
                    'nominal_cadence': np.timedelta64(30, 's'),
                    'units': 'V',
                    },
                },        
            'HumidityData': {
                'realtime': {
                    'channels': np.array(['Relative humidity']),
                    'path': os.path.join(data_dir,
                                         'lan4/%Y/%m/lan4_%Y%m%d_cloud.txt'),
                    'duration': np.timedelta64(24, 'h'),
                    'load_converter': load_humidity_data,
                    'nominal_cadence': np.timedelta64(30, 's'),
                    'units': '%',
                    },
                },
            },
        'start_time': np.datetime64('2013-07-13T00:00Z'),
        'end_time': None, # Still operational
        'acknowledgement': {'short': 'Steve Marple.'},
        },
    
    }


ap.add_project('CLOUDWATCH', sites)
