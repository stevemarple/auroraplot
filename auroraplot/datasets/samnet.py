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
import auroraplot.data
from auroraplot.magdata import MagData as MagData
from auroraplot.temperaturedata import TemperatureData
from auroraplot.voltagedata import VoltageData

logger = logging.getLogger(__name__)

base_url = 'http://spears.lancs.ac.uk/data/samnet/'

sam_channels = ['H', 'E', 'Z']

def load_samnet_data(file_name, archive_data, 
                        project, site, data_type, channels, start_time, 
                        end_time, **kwargs):

    chan_tup = tuple(archive_data['channels'])
    col_idx = []
    for c in channels:
        col_idx.append(chan_tup.index(c))
    nominal_cadence_s = (archive_data['nominal_cadence'] / 
                         np.timedelta64(1000000, 'us'))
    try:
        try:
            conv = lambda s: (s.strip().startswith('9999.9') and np.nan) \
                or float(s.strip())
            comments = ap.get_site_info(project, site, 'samnet_code')[0]
            data = np.loadtxt(file_name, 
                              unpack=True, 
                              converters={0: conv, 1: conv, 2: conv},
                              comments=comments)
            # TODO: check correct settings for sample start/end time
            # for both 1s and 5s data. IIRC 1s is offset and 5s is
            # centred.
            sample_start_time = np.arange(0, 86400, nominal_cadence_s)\
                .astype('m8[s]') + start_time
            sample_end_time = sample_start_time + \
                archive_data['nominal_cadence']
                
            data = data[col_idx] * 1e-9
            integration_interval = np.ones_like(data) \
                * archive_data['nominal_cadence']

            r = MagData( \
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
            logger.debug(traceback.format_exc())

        finally:
            # uh.close()
            pass
    except Exception as e:
        logger.info('Could not open ' + file_name)
        logger.debug(str(e))
        logger.debug(traceback.format_exc())

    return None


def load_new_samnet_data(file_name, archive_data, 
                            project, site, data_type, channels, start_time, 
                            end_time, **kwargs):
    '''Convert new-style SAMNET data to match standard data type

    archive: name of archive from which data was loaded
    archive_info: archive metadata
    '''

    def to_tesla(x):
        if x == '9999999.999':
            return np.nan
        else:
            return float(x)/1e9

    try:
        if file_name.startswith('/'):
            fh = urlopen('file:' + file_name)
            # Pass the file name instead of a file handle.
            # fh = file_name
        else:
            # fh = urlopen(file_name)
            req = requests.get(file_name, stream=True)
            fh = req.raw
        try:
            file_data = np.loadtxt(fh, unpack=True, 
                                   comments='%',
                                   converters={3: to_tesla, 
                                               4: to_tesla, 
                                               5: to_tesla})
            assert np.all(file_data[0] >= 0) and np.all(file_data[0] < 24)
            assert np.all(file_data[1] >= 0) and np.all(file_data[1] < 60)
            assert np.all(file_data[2] >= 0) and np.all(file_data[2] < 60)

            data = file_data[3:]
            sample_start_time = start_time + \
                + file_data[0].astype('m8[h]') \
                + file_data[1].astype('m8[m]') \
                + file_data[2].astype('m8[s]') \
                + np.timedelta64(0, 'us')

            # end time and integration interval are guesstimates
            sample_end_time = sample_start_time \
                + archive_data['nominal_cadence']
            integration_interval = np.tile(archive_data['nominal_cadence'],
                                           data.shape)
            r = MagData( \
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
            logger.debug(traceback.format_exc())


        finally:
            fh.close()
    except Exception as e:
        logger.info('Could not open ' + file_name)
        logger.debug(str(e))
        logger.debug(traceback.format_exc())

    return None


def load_new_samnet_temp_volt_data(file_name, archive_data, 
                                      project, site, data_type, 
                                      channels, start_time, 
                                      end_time, **kwargs):
    '''Convert new-style SAMNET temperate/voltage data to standard type'''

    def to_float(x):
        if x == '999999.99':
            return np.nan
        else:
            return float(x)

    def to_cpu_temp(x):
        if x == '999.9':
            return np.nan
        else:
            return float(x)

    try:
        if file_name.startswith('/'):
            uh = urlopen('file:' + file_name)
        else:
            uh = urlopen(file_name)
        try:
            file_data = np.loadtxt(uh, unpack=True, 
                                   comments='%',
                                   converters={2: to_float, 
                                               3: to_float, 
                                               4: to_float, 
                                               5: to_float,
                                               6: to_float,
                                               7: to_cpu_temp})
            assert np.all(file_data[0] >= 0) and np.all(file_data[0] < 24)
            assert np.all(file_data[1] >= 0) and np.all(file_data[1] < 60)
            
            if data_type == 'TemperatureData':
                # Keep channel order same as for AuroraWatchNet
                data = file_data[[3,2,4,7]]
                class_type = TemperatureData
            elif data_type == 'VoltageData':
                data = file_data[[5]] # aux not used
                class_type = VoltageData
            sample_start_time = start_time + \
                + file_data[0].astype('m8[h]') \
                + file_data[1].astype('m8[m]') \
                + np.timedelta64(0, 'us')

            # end time and integration interval are guesstimates
            sample_end_time = sample_start_time \
                + archive_data['nominal_cadence']
            integration_interval = np.tile(archive_data['nominal_cadence'],
                                           data.shape)
            r = class_type( \
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
            logger.debug(traceback.format_exc())

        finally:
            uh.close()
    except Exception as e:
        logger.info('Could not open ' + file_name)
        logger.debug(str(e))
        logger.debug(traceback.format_exc())

    return None



def load_rt_data(file_name, archive_data,
                    project, site, data_type, channels, start_time, 
                    end_time, **kwargs):
    assert(data_type == 'MagData')
    try:
        chan_tup = tuple(archive_data['channels'])
        col_idx = []
        for c in channels:
            col_idx.append(1 + chan_tup.index(c))

        if file_name.startswith('/'):
            uh = urlopen('file:' + file_name)
        else:
            uh = urlopen(file_name)

        try:
            data = np.loadtxt(uh, unpack=True)
            sample_start_time = start_time \
                + (np.timedelta64(1000000, 'us') * data[0])
            sample_end_time = sample_start_time + np.timedelta64(1000000, 'us')
            integration_interval = np.tile(1000000, [len(channels), 
                                                 len(sample_start_time)],
                                           dtype='m8[us]')
            r = MagData(project=project,
                        site=site,
                        channels=channels,
                        start_time=start_time,
                        end_time=end_time,
                        sample_start_time=sample_start_time, 
                        sample_end_time=sample_end_time,
                        integration_interval=integration_interval,
                        nominal_cadence=archive_data['nominal_cadence'],
                        data=data[col_idx]*1e-9,
                        units='T',
                        sort=True)
            return r
        except Exception as e:
            logger.info('Could not read ' + file_name)
            logger.debug(str(e))
            logger.debug(traceback.format_exc())

        finally:
            uh.close()
    except Exception as e:
        logger.info('Could not open ' + file_name)
        logger.debug(str(e))
        logger.debug(traceback.format_exc())

    return None

sites = {
    'BOR1': {
        'location': 'Borok, CIS',
        'latitude': 58.03,
        'longitude': 38.33,
        'elevation': np.nan,
        'start_time': np.datetime64('1998-12-03T00:00:00+0000', 's'),
        'end_time': np.datetime64('2015-06-29T00:00:00+0000', 's'),
        'copyright': 'Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space and Plasma Physics group, Department of Physics, Lancaster University.',
        'samnet_code': 'bo',
        },
    'CRK1': {
        'location': 'Crooktree, UK',
        'latitude': 57.09,
        'longitude': -2.64,
        'elevation': np.nan,
        'start_time': np.datetime64('2002-05-17T00:00:00+0000', 's'),
        'end_time': np.datetime64('2013-02-26T00:00:00+0000', 's'),
        'copyright': 'Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space and Plasma Physics group, Department of Physics, Lancaster University.',
        'samnet_code': 'cr',
        },
    'CRK2': {
        'location': 'Crooktree, UK',
        'latitude': 57.09,
        'longitude': -2.64,
        'elevation': np.nan,
        'start_time': np.datetime64('2015-08-18T00:00:00+0000', 's'),
        'end_time': None,
        'k_index_scale': 800e-9, # Estimated
        'copyright': 'Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space and Plasma Physics group, Department of Physics, caster University.',
        'line_color': [1, 0, 0],
        'samnet_code': 'cr',
        'data_types': {
            'MagData': {
                'default': '1s',
                '1s': {
                    'channels': sam_channels,
                    'path': base_url + 'new/crk2/%Y/%m/%Y%m%d.txt',
                    'duration': np.timedelta64(24, 'h'),
                    'load_converter': ap.data._generic_load_converter,
#                    'load_converter': load_new_samnet_data,
                    'nominal_cadence': np.timedelta64(1, 's'),
                    'units': 'T',
                    # Generic load/save information
                    'constructor': ap.magdata.MagData,
                    'data_multiplier': 1000000000,
                    'valid_range': (-10000e-9, 10000e-9),
                    'timestamp_method': 'hms',
                    'comments': '%',
                    },
                '1min': {
                    'channels': sam_channels,
                    'path': base_url + 'new/crk2/%Y/%m/%Y%m%d.min',
                    'duration': np.timedelta64(24, 'h'),
                    'load_converter': ap.data._generic_load_converter,
                    'nominal_cadence': np.timedelta64(1, 'm'),
                    'units': 'T',
                    # Generic load/save information
                    'constructor': ap.magdata.MagData,
                    'data_multiplier': 1000000000,
                    'valid_range': (-10000e-9, 10000e-9),
                    'timestamp_method': 'hm',
                    'comments': '%',
                    },
                },
            'MagQDC': {
                'qdc': {
                    'channels': np.array(sam_channels),
                    'path': base_url + 'new/crk2/qdc/%Y/crk2_qdc_%Y%m.txt',
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet_qdc',
                    'load_converter': ap.magdata.load_qdc_data,
                    'nominal_cadence': np.timedelta64(5, 's'),
                    'units': 'T',
                    },
                },
            'TemperatureData': {
                '1min': {
                    'channels': np.array(['Sensor temperature',
                                          'System temperature',
                                          'Obsdaq temperature',
                                          'CPU temperature']),
                    'path': base_url + 'new/crk2/%Y/%m/%Y%m%d.sup',
                    'duration': np.timedelta64(24, 'h'),
                    'load_converter': load_new_samnet_temp_volt_data,
                    'nominal_cadence': np.timedelta64(1, 'm'),
                    'units': six.u('\N{DEGREE SIGN}C'),
                    },
                },
            'VoltageData': {
                '1min': {
                    'channels': np.array(['Obsdaq voltage']),
                    'path': base_url + 'new/crk2/%Y/%m/%Y%m%d.sup',
                    'duration': np.timedelta64(24, 'h'),
                    'load_converter': load_new_samnet_temp_volt_data,
                    'nominal_cadence': np.timedelta64(1, 'm'),
                    'units': 'V',                    
                    },
                }
            },
        },
    'ESK1': {
        'location': 'Eskdalemuir, UK',
        'latitude': 55.32,
        'longitude': -3.2,
        'elevation': np.nan,
        'start_time': np.datetime64('2001-01-01T00:00:00+0000', 's'),
        'end_time': None,
        'k_index_scale': 750e-9, # From BGS  Monthly Magnetic Bulletin
        'copyright': 'Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space and Plasma Physics group, Department of Physics, Lancaster University.',
        'samnet_code': 'es',
        'data_types': {
            'MagData': {
                'incoming': {
                    'channels': sam_channels,
                    'path': (base_url + 'incoming/esk/es%d%m%y.day.gz'),
                    'duration': np.timedelta64(24, 'h'),
                    'load_converter': load_samnet_data,
                    'nominal_cadence': np.timedelta64(1000000, 'us'),
                    'units': 'T',
                },
            },
        },
    },
    'FAR1': {
        'location': ', Faroes',
        'latitude': 62.05,
        'longitude': -7.02,
        'elevation': np.nan,
        'start_time': np.datetime64('1987-10-01T00:00:00+0000', 's'),
        'end_time': np.datetime64('2011-02-12T00:00:00+0000', 's'),
        'copyright': 'Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space and Plasma Physics group, Department of Physics, Lancaster University.',
        'samnet_code': 'fa',
        },
    'GML1': {
        'location': 'Glenmore Lodge, UK',
        'latitude': 57.16,
        'longitude': -3.68,
        'elevation': np.nan,
        'start_time': np.datetime64('1987-10-01T00:00:00+0000', 's'),
        'end_time': np.datetime64('2002-05-15T00:00:00+0000', 's'),
        'copyright': 'Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space and Plasma Physics group, Department of Physics, Lancaster University.',
        'samnet_code': 'gm',
        },
    'HAD1': {
        'location': 'Hartland, UK',
        'latitude': 50.99,
        'longitude': -4.48,
        'elevation': np.nan,
        'start_time': np.datetime64('2001-01-01T00:00:00+0000', 's'),
        'end_time': None,
        'k_index_scale': 500e-9, # From BGS  Monthly Magnetic Bulletin
        'copyright': 'Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space and Plasma Physics group, Department of Physics, Lancaster University.',
        'samnet_code': 'hd',
        'data_types': {
            'MagData': {
                'incoming': {
                    'channels': sam_channels,
                    'path': (base_url + 'incoming/had/hd%d%m%y.day.gz'),
                    'duration': np.timedelta64(24, 'h'),
                    'load_converter': load_samnet_data,
                    'nominal_cadence': np.timedelta64(1000000, 'us'),
                    'units': 'T',
                },
            },
        },
    },
    'HAN1': {
        'location': 'Hankasalmi, Finland',
        'latitude': 62.3,
        'longitude': 26.65,
        'elevation': np.nan,
        'start_time': np.datetime64('1997-01-03T00:00:00+0000', 's'),
        'end_time': np.datetime64('2005-05-16T00:00:00+0000', 's'),
        'copyright': 'Finnish Meteorological Institute / Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space and Plasma Physics group, Department of Physics, Lancaster University. Original data provided by the Finnish Meteorological Institute.',
        'samnet_code': 'ha',
        },
    'HAN3': {
        'location': 'Hankasalmi, Finland',
        'latitude': 62.2539,
        'longitude': 26.5967,
        'elevation': np.nan,
        'start_time': np.datetime64('2005-08-18T00:00:00+0000', 's'),
        'end_time': None,
        'copyright': 'Finnish Meteorological Institute / Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space and Plasma Physics group, Department of Physics, Lancaster University. Original data provided by the Finnish Meteorological Institute.',
        'samnet_code': 'ha',
        },
    'HLL1': {
        'location': 'Hella, Iceland',
        'latitude': 63.77,
        'longitude': -20.56,
        'elevation': np.nan,
        'start_time': np.datetime64('1998-10-06T00:00:00+0000', 's'),
        'end_time': None,
        'copyright': 'Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space and Plasma Physics group, Department of Physics, Lancaster University.',
        'samnet_code': 'hl',
        },
    'KIL1': {
        'location': 'Kilpisjarvi, Finland',
        'latitude': 69.02,
        'longitude': 20.79,
        'elevation': np.nan,
        'start_time': np.datetime64('1997-12-01T00:00:00+0000', 's'),
        'end_time': None,
        'copyright': 'Finnish Meteorological Institute / Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space and Plasma Physics group, Department of Physics, Lancaster University. Original data provided by the Finnish Meteorological Institute.',
        'samnet_code': 'ki',
        },
    'KVI1': {
        'location': 'Kvistaberg, Sweden',
        'latitude': 59.5,
        'longitude': 17.63,
        'elevation': np.nan,
        'start_time': np.datetime64('1987-10-13T00:00:00+0000', 's'),
        'end_time': np.datetime64('2003-01-01T00:00:00+0000', 's'),
        'copyright': 'Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space and Plasma Physics group, Department of Physics, Lancaster University.',
        'samnet_code': 'kv',
        },
    'LAN1': {
        'location': 'Lancaster, UK',
        'latitude': 54.01,
        'longitude': -2.77,
        'elevation': np.nan,
        'start_time': np.datetime64('2003-02-20T17:00:00+0000', 's'),
        'end_time': np.datetime64('2013-10-24T00:00:00+0000', 's'),
        'copyright': 'Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space and Plasma Physics group, Department of Physics, Lancaster University.',
        'line_color': [1, 0, 0],
        'samnet_code': 'la',
        },
    'LAN2': {
        'location': 'Lancaster, UK',
        'latitude': 54.01,
        'longitude': -2.77,
        'elevation': np.nan,
        'start_time': np.datetime64('2014-12-02T00:00:00+0000', 's'),
        'end_time': None,
        'k_index_scale': 650e-9, # Estimated
        'copyright': 'Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space and Plasma Physics group, Department of Physics, Lancaster University.',
        'line_color': [181/255., 18/255., 27/255.],
        'samnet_code': 'la',
        'data_types': {
            'MagData': {
                'default': '1s',
                '1s': {
                    'channels': sam_channels,
                    'path': base_url + 'new/lan2/%Y/%m/%Y%m%d.txt',
                    'duration': np.timedelta64(24, 'h'),
#                    'load_converter': ap.data._generic_load_converter,
                    'load_converter': load_new_samnet_data,
                    'nominal_cadence': np.timedelta64(1, 's'),
                    'units': 'T',
                    # Generic load/save information
                    'constructor': ap.magdata.MagData,
                    'data_multiplier': 1000000000,
                    'valid_range': (-10000e-9, 10000e-9),
                    'timestamp_method': 'hms',
                    'comments': '%',
                    },
                '1min': {
                    'channels': sam_channels,
                    'path': base_url + 'new/lan2/%Y/%m/%Y%m%d.min',
                    'duration': np.timedelta64(24, 'h'),
                    'load_converter': ap.data._generic_load_converter,
                    'nominal_cadence': np.timedelta64(1, 'm'),
                    'units': 'T',
                    # Generic load/save information
                    'constructor': ap.magdata.MagData,
                    'data_multiplier': 1000000000,
                    'valid_range': (-10000e-9, 10000e-9),
                    'timestamp_method': 'hm',
                    'comments': '%',
                    },
                },
            'MagQDC': {
                'qdc': {
                    'channels': np.array(sam_channels),
                    'path': base_url + 'new/lan2/qdc/%Y/lan2_qdc_%Y%m.txt',
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet_qdc',
                    'load_converter': ap.magdata.load_qdc_data,
                    'nominal_cadence': np.timedelta64(5, 's'),
                    'units': 'T',
                    },
                },
            'TemperatureData': {
                '1min': {
                    'channels': np.array(['Sensor temperature',
                                          'System temperature',
                                          'Obsdaq temperature',
                                          'CPU temperature']),
                    'path': base_url + 'new/lan2/%Y/%m/%Y%m%d.sup',
                    'duration': np.timedelta64(24, 'h'),
                    'load_converter': load_new_samnet_temp_volt_data,
                    'nominal_cadence': np.timedelta64(1, 'm'),
                    'units': six.u('\N{DEGREE SIGN}C'),
                    },
                },
            'VoltageData': {
                '1min': {
                    'channels': np.array(['Obsdaq voltage']),
                    'path': base_url + 'new/lan2/%Y/%m/%Y%m%d.sup',
                    'duration': np.timedelta64(24, 'h'),
                    'load_converter': load_new_samnet_temp_volt_data,
                    'nominal_cadence': np.timedelta64(1, 'm'),
                    'units': 'V',                    
                    },
                }
            },
        },
    'LER1': {
        'location': 'Lerwick, UK',
        'latitude': 60.13,
        'longitude': -1.18,
        'elevation': np.nan,
        'start_time': np.datetime64('2001-01-01T00:00:00+0000', 's'),
        'end_time': None,
        'k_index_scale': 1000e-9, # From BGS  Monthly Magnetic Bulletin
        'copyright': 'Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space and Plasma Physics group, Department of Physics, Lancaster University.',
        'samnet_code': 'le',
        'data_types': {
            'MagData': {
                'incoming': {
                    'channels': sam_channels,
                    'path': (base_url + 'incoming/ler/le%d%m%y.day.gz'),
                    'duration': np.timedelta64(24, 'h'),
                    'load_converter': load_samnet_data,
                    'nominal_cadence': np.timedelta64(1000000, 'us'),
                    'units': 'T',
                },
            },
        },
    },
    'LNC2': {
        'location': 'Lancaster, UK',
        'latitude': 54.01,
        'longitude': -2.77,
        'elevation': np.nan,
        'start_time': np.datetime64('2006-12-24T00:00:00+0000', 's'),
        'end_time': np.datetime64('2008-06-27T00:00:00+0000', 's'),
        'copyright': 'Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space and Plasma Physics group, Department of Physics, Lancaster University.',
        'samnet_code': 'ln',
        },
    'NOR1': {
        'location': 'Nordli, Norway',
        'latitude': 64.37,
        'longitude': 13.36,
        'elevation': np.nan,
        'start_time': np.datetime64('1987-10-01T00:00:00+0000', 's'),
        'end_time': np.datetime64('2003-03-01T00:00:00+0000', 's'),
        'copyright': 'Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space and Plasma Physics group, Department of Physics, Lancaster University.',
        'samnet_code': 'no',
        },
    'NUR1': {
        'location': 'Nurmijarvi, Finland',
        'latitude': 60.51,
        'longitude': 24.66,
        'elevation': np.nan,
        'start_time': np.datetime64('1987-10-01T00:00:00+0000', 's'),
        'end_time': np.datetime64('2003-01-01T00:00:00+0000', 's'),
        'copyright': 'Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space and Plasma Physics group, Department of Physics, Lancaster University.',
        'samnet_code': 'nu',
        },
    'NUR3': {
        'location': 'Nurmijarvi, Finland',
        'latitude': 60.5,
        'longitude': 24.65,
        'elevation': np.nan,
        'start_time': np.datetime64('2003-01-01T00:00:00+0000', 's'),
        'end_time': None,
        'copyright': 'Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space and Plasma Physics group, Department of Physics, Lancaster University.',
        'samnet_code': 'nu',
        },
    'OUJ2': {
        'location': 'Oulujarvi, Finland',
        'latitude': 64.52,
        'longitude': 27.23,
        'elevation': np.nan,
        'start_time': np.datetime64('2003-01-01T00:00:00+0000', 's'),
        'end_time': None,
        'copyright': 'Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space and Plasma Physics group, Department of Physics, Lancaster University.',
        'samnet_code': 'oj',
        },
    'OUL1': {
        'location': 'Oulu, Finland',
        'latitude': 65.1,
        'longitude': 25.85,
        'elevation': np.nan,
        'start_time': np.datetime64('1987-10-01T00:00:00+0000', 's'),
        'end_time': np.datetime64('2003-01-01T00:00:00+0000', 's'),
        'copyright': 'Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space and Plasma Physics group, Department of Physics, Lancaster University.',
        'samnet_code': 'ou',
        },
    'UPS2': {
        'location': 'Uppsala, Sweden',
        'latitude': 59.9,
        'longitude': 17.35,
        'elevation': np.nan,
        'start_time': np.datetime64('2003-01-01T00:00:00+0000', 's'),
        'end_time': None,
        'copyright': 'Geological Survey of Sweden / Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space and Plasma Physics group, Department of Physics, Lancaster University. Original data provided by Geological Survey of Sweden.',
        'samnet_code': 'up',
        },
    'YOR1': {
        'location': 'York, UK',
        'latitude': 53.95,
        'longitude': -1.05,
        'elevation': np.nan,
        'start_time': np.datetime64('1987-10-01T00:00:00+0000', 's'),
        'end_time': np.datetime64('2006-03-30T12:00:00+0000', 's'),
        'copyright': 'Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space and Plasma Physics group, Department of Physics, Lancaster University.',
        'samnet_code': 'yo',
        },
    
    }



# Set activity color/thresholds unless already set.
default_activity_thresholds = np.array([0.0, 50.0, 100.0, 200.0]) * 1e-9
default_activity_colors = np.array([[0.2, 1.0, 0.2],  # green  
                                    [1.0, 1.0, 0.0],  # yellow
                                    [1.0, 0.6, 0.0],  # amber
                                    [1.0, 0.0, 0.0]]) # red

    

default_data_types = {
    'MagData': {
        'default': '5s',
        '1s': {
            'channels': sam_channels,
            'path': (base_url + '1s_archive/%Y/%m/{sc}%d%m%y.dgz'),
            'duration': np.timedelta64(24, 'h'),
            'load_converter': load_samnet_data,
            'nominal_cadence': np.timedelta64(1000000, 'us'),
            'units': 'T',
        },
        '5s': {
            'channels': sam_channels,
            'path': (base_url + '5s_archive/%Y/%m/{sc}%d%m%Y.5s.gz'),
            'duration': np.timedelta64(24, 'h'),
            'load_converter': load_samnet_data,
            'nominal_cadence': np.timedelta64(5000000, 'us'),
            'units': 'T',
        },
        'realtime': {
            'channels': sam_channels,
            'path': (base_url +
                     'realtime/{site_lc}/%Y/%m/{site_lc}%Y%m%d.rt'),
            'duration': np.timedelta64(24, 'h'),
            'load_converter': load_rt_data,
            'nominal_cadence': np.timedelta64(1000000, 'us'),
            'units': 'T',
        },
        'realtime_baseline': {
            'channels': sam_channels,
            'path': (base_url +
                     'baseline/realtime/{site_lc}/{site_lc}_%Y.txt'),
            'duration': np.timedelta64(1, 'Y'),
            'load_converter': ap.data._generic_load_converter,
            'save_converter': ap.data._generic_save_converter,
            'nominal_cadence': np.timedelta64(1, 'D'),
            'units': 'T',
            # Information for generic load/save 
            'constructor': ap.magdata.MagData,
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
            'channels': sam_channels,
            'path': base_url + 'qdc/new/{site_lc}/%Y/{site_lc}_qdc_%Y%m.txt',
            'duration': np.timedelta64(24, 'h'),
            # Use the standard converter for MagQDC
            'load_converter': ap.magdata.load_qdc_data,
            'nominal_cadence': np.timedelta64(5000000, 'us'),
            'units': 'T',
            },
        },
    }

for s in sites:
    site_lc = s.lower()
    sc = sites[s]['samnet_code'] # Two-letter lower-case abbreviation

    # Populate the data types
    if 'data_types' not in sites[s]:
        sites[s]['data_types'] = {}

    sdt = sites[s]['data_types']

    for dt in default_data_types:
        if dt not in sdt:
            sdt[dt] = {}
        for an,av in default_data_types[dt].iteritems():
            if an not in sdt[dt]:
                sdt[dt][an] = \
                    copy.deepcopy(av)
                if an == 'default':
                    continue
                if not hasattr(sdt[dt][an]['path'], '__call__'):
                    sdt[dt][an]['path'] = \
                        sdt[dt][an]['path'].format(site_lc=site_lc, sc=sc)
            elif sdt[dt] is None:
                # None used as a placeholder to prevent automatic
                # population, now clean up
                del(sdt[dt])
                    

    if 'activity_thresholds' not in sites[s]:
        sites[s]['activity_thresholds'] = default_activity_thresholds
    if 'activity_colors' not in sites[s]:
        sites[s]['activity_colors'] = default_activity_colors

ap.add_project('SAMNET', sites)

