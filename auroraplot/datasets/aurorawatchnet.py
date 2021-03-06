import copy
from decimal import Decimal
import os
import logging

# Python 2/3 compatibility
import six
from six import iteritems

try:
    from urllib.request import urlopen
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse
    from urllib import urlopen

import numpy as np

import auroraplot as ap
import auroraplot.auroralactivity
import auroraplot.data
import auroraplot.magdata
import auroraplot.tools
from auroraplot.magdata import MagData
from auroraplot.magdata import MagQDC
from auroraplot.temperaturedata import TemperatureData
from auroraplot.voltagedata import VoltageData

logger = logging.getLogger(__name__)

base_url = 'http://aurorawatch.lancs.ac.uk/data/aurorawatchnet/'


def check_mag_data(data):
    data[np.logical_or(data < -0.0001, data > 0.0001)] = np.nan
    return data


def check_temperature(data):
    data[np.logical_or(data < -40, data > 100)] = np.nan
    return data


def check_voltage(data):
    data[np.logical_or(data < 0, data > 50)] = np.nan
    return data


def load_awn_data(file_name, archive_data,
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
        'TemperatureData': {
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
        if os.path.exists(file_name):
            uh = open(file_name)
        else:
            uh = urlopen(file_name)
        try:
            data = np.genfromtxt(uh, unpack=True, invalid_raise=False)
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


def k_index_filter_battery(mag_data):
    '''Filter data for K index (battery-powered magnetometers).
    
    Battery-powered magnetometers have higher noise, filter to reduce.'''
    md_filt = ap.tools.sgolay_filt(mag_data,
                                   np.timedelta64(630, 's'), 3)
    md_filt.set_cadence(np.timedelta64(5, 'm'), inplace=True)
    return md_filt


def load_bad_data(project, site, data_type, start_time, end_time,
                  archive=None, path=None, extension='.bad', **kwargs):
    '''Load data from bad data files, usually those which end in .bad.

    This function is a wrapper which calls aurorawatchnet.load_data()
    after appending an extension to the file names. The path cannot be
    a callable.
    '''
    if path is None:
        ai = ap.get_archive_info(project, site, data_type, archive=archive)
        path = ai[1]['path'] + extension

    return ap.load_data(project, site, data_type, start_time, end_time,
                        archive=ai[0], path=path, **kwargs)


def remove_spikes(md, **kwargs):
    return md.remove_spikes_chauvenet(savgol_window=np.timedelta64(5, 'm'),
                                      chauvenet_window=np.array([89, 79]).astype('timedelta64[s]'))


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
        'end_time': None,  # Still operational
        'k_index_scale': 650e-9,  # Estimated
        'k_index_filter': None,
        'copyright': 'Steve Marple.',
        'license': cc3_by_nc_sa,
        'attribution': 'Data provided by Steve Marple.',
        'line_color': [0, 0.6, 0],
        'data_types': {
            'MagData': {
                'default': 'realtime',
                'realtime': {
                    'channels': np.array(['H', 'E', 'Z']),
                    'path': base_url + 'lan1/%Y/%m/lan1_%Y%m%d.txt',
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'load_converter': load_awn_data,
                    'nominal_cadence': np.timedelta64(5000000, 'us'),
                    'units': 'T',
                    'sort': True,
                },
                'legacy_realtime': {
                    'channels': np.array(['H']),
                    'path': base_url + 'lan1/%Y/%m/lan1_%Y%m%d.txt',
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'load_converter': load_awn_data,
                    'nominal_cadence': np.timedelta64(30000000, 'us'),
                    'units': 'T',
                    'sort': True,
                },
                'realtime_baseline': {
                    'channels': np.array(['H', 'E', 'Z']),
                    'path': (base_url +
                             'baseline/realtime/lan1/lan1_%Y.txt'),
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
                    'data_multiplier': 1000000000,  # Store as nT values
                    # Information for making the data files
                    'qdc_fit_duration': np.timedelta64(10, 'D'),
                    'realtime_qdc': True,
                },
            },
        },
    },  # LAN1

    'LAN3': {
        'location': 'Lancaster, UK',
        'latitude': 54.01,
        'longitude': -2.77,
        'elevation': 93,
        'start_time': np.datetime64('2012-12-18T00:00Z'),
        'end_time': None,  # Still operational
        'k_index_scale': 650e-9,  # Estimated
        'license': cc3_by_nc_sa,
        'copyright': 'Lancaster University.',
        'attribution': 'Space and Plasma Physics group, ' + \
                       'Department of Physics, Lancaster University, UK.',
        'line_color': [1, 0, 0],
    },  # LAN3

    'ORM': {
        'location': 'Ormskirk, UK',
        'latitude': 53.569195,
        'longitude': -2.887264,
        'elevation': None,
        'start_time': np.datetime64('2013-08-01T00:00Z'),
        'end_time': np.datetime64('2017-04-14T00:00Z'),
        'k_index_scale': 650e-9,  # Estimated
        'license': cc3_by_nc_sa,
        'copyright': 'Lancaster University.',
        'attribution': 'Space and Plasma Physics group, ' + \
                       'Department of Physics, Lancaster University, UK.',
        'line_color': [186.0 / 255, 216.0 / 255, 10.0 / 255],
    },  # ORM

    'TEST1': {
        'location': 'Lancaster, UK',
        'latitude': 54.0,
        'longitude': -2.78,
        'elevation': 27,
        'start_time': np.datetime64('2013-11-10T00:00Z'),
        'end_time': np.datetime64('2017-08-03T00:00Z'),
        'k_index_scale': 650e-9,  # Estimated
        'license': cc3_by_nc_sa,
        'copyright': 'Lancaster University.',
        'attribution': 'Space and Plasma Physics group, ' + \
                       'Department of Physics, Lancaster University, UK.',
        'description': (
            'Test magnetometer system. ' +
            '2013: battery-operated AuroraWatchNet magnetometer. ' +
            '2016: Raspberry Pi magnetometer sytem operating outside with sensor buried for temperature stability.'),
        'line_color': [.1, .4, .1],
    },  # TEST1

    'BRA': {
        'location': 'Brae, Shetland, UK',
        'latitude': 60.395869,
        'longitude': -1.351124000000027,
        'elevation': 11,
        'start_time': np.datetime64('2014-01-01T00:00:00+0000'),
        'end_time': None,  # Still operational
        'k_index_scale': 1000e-9,  # Estimated, based on BGS Lerwick site
        'license': cc3_by_nc_sa,
        'copyright': 'Lancaster University.',
        'attribution': 'Space and Plasma Physics group, ' + \
                       'Department of Physics, Lancaster University, UK.',
    },  # BRA

    'SAN': {
        'location': 'Sanday, UK',
        'latitude': 59.25110830191925,
        'longitude': -2.5873320735991,
        'elevation': 16.366,
        'start_time': np.datetime64('2014-01-01T00:00:00+0000'),
        'end_time': None,  # Still operational
        'k_index_scale': 950e-9,  # Estimated
        'license': cc3_by_nc_sa,
        'copyright': 'Lancaster University.',
        'attribution': 'Space and Plasma Physics group, ' + \
                       'Department of Physics, Lancaster University, UK.',
    },  # SAN

    'TOB': {
        'location': 'Tobermory, Mull, UK',
        'latitude': 56.62415194965067,
        'longitude': -6.068624798208475,
        'elevation': 43,
        'start_time': np.datetime64('2014-01-01T00:00:00+0000'),
        'end_time': None,  # Still operational
        'k_index_scale': 850e-9,  # Estimated
        'license': cc3_by_nc_sa,
        'copyright': 'Lancaster University.',
        'attribution': 'Space and Plasma Physics group, ' + \
                       'Department of Physics, Lancaster University, UK.',
    },  # TOB

    'WHI': {
        'location': 'Whitehaven, Cumbria, UK',
        'latitude': 54.543384,
        'longitude': -3.5610000000000355,
        'elevation': 132,
        'start_time': np.datetime64('2014-01-01T00:00:00+0000'),
        'end_time': None,  # Still operational
        'k_index_scale': 680e-9,  # Estimated
        'license': cc3_by_nc_sa,
        'copyright': 'Lancaster University.',
        'attribution': 'Space and Plasma Physics group, ' + \
                       'Department of Physics, Lancaster University, UK.',
        'line_color': [.6, .6, .6],
    },  # WHI

    'ALT': {
        'location': 'Altrincham, UK',
        'latitude': 53.381988,
        'longitude': -2.3597310000000107,
        'elevation': 68,
        'start_time': np.datetime64('2014-01-01T00:00:00+0000'),
        'end_time': None,  # Still operational
        'k_index_scale': 600e-9,  # Estimated
        'license': cc3_by_nc_sa,
        'copyright': 'Lancaster University.',
        'attribution': 'Space and Plasma Physics group, ' + \
                       'Department of Physics, Lancaster University, UK.',
    },  # ALT

    'MAL': {
        'location': 'Malpas, UK',
        'latitude': 53.029658,
        'longitude': -2.760221999999999,
        'elevation': 122,
        'start_time': np.datetime64('2014-01-01T00:00:00+0000'),
        'end_time': None,  # Still operational
        'k_index_scale': 600e-9,  # Estimated
        'license': cc3_by_nc_sa,
        'copyright': 'Lancaster University.',
        'attribution': 'Space and Plasma Physics group, ' + \
                       'Department of Physics, Lancaster University, UK.',
    },  # MAL

    'ASH': {
        'location': 'Ashbourne, UK',
        'latitude': 53.021899,
        'longitude': -1.7287959999999885,
        'elevation': 157,
        'start_time': np.datetime64('2014-01-01T00:00:00+0000'),
        'end_time': None,  # Still operational
        'k_index_scale': 600e-9,  # Estimated
        'license': cc3_by_nc_sa,
        'copyright': 'Lancaster University.',
        'attribution': 'Space and Plasma Physics group, ' + \
                       'Department of Physics, Lancaster University, UK.',
    },  # ASH

    'PEL': {
        'location': 'Pelsall, UK',
        'latitude': 52.623806,
        'longitude': -1.9490200000000186,
        'elevation': 143,
        'start_time': np.datetime64('2014-01-01T00:00:00+0000'),
        'end_time': None,  # Still operational
        'k_index_scale': 580e-9,  # Estimated
        'license': cc3_by_nc_sa,
        'copyright': 'Lancaster University.',
        'attribution': 'Space and Plasma Physics group, ' + \
                       'Department of Physics, Lancaster University, UK.',
    },  # PEL

    'BRE': {
        'location': 'Brecon, UK',
        'latitude': 52.02851,
        'longitude': -3.2026879999999665,
        'elevation': 121,
        'start_time': np.datetime64('2014-01-01T00:00:00+0000'),
        'end_time': None,  # Still operational
        'k_index_scale': 550e-9,  # Estimated
        'license': cc3_by_nc_sa,
        'copyright': 'Lancaster University.',
        'attribution': 'Space and Plasma Physics group, ' + \
                       'Department of Physics, Lancaster University, UK.',
        'line_color': [119.0 / 255, 11.0 / 255, 0],
    },  # BRE

    'CAN': {
        'location': 'Canterbury, UK',
        'latitude': 51.260914,
        'longitude': 1.084820000000036,
        'elevation': 48,
        'start_time': np.datetime64('2014-01-01T00:00:00+0000'),
        'end_time': None,  # Still operational
        'k_index_scale': 500e-9,  # Estimated, based on BGS Hartland site
        'license': cc3_by_nc_sa,
        'copyright': 'Lancaster University.',
        'attribution': 'Space and Plasma Physics group, ' + \
                       'Department of Physics, Lancaster University, UK.',
        'line_color': [64.0 / 255, 224.0 / 255, 208.0 / 255],
    },  # CAN

    'CWX': {
        # Private station operated by Cumbernauld Weather
        'location': 'Cumbernauld, UK',
        'latitude': 55 + (56.1 / 60),
        'longitude': -(4 + (2.2 / 60)),
        'elevation': 82,
        'start_time': np.datetime64('2014-01-01T00:00:00+0000'),
        'end_time': None,  # Still operational
        'url': 'http://www.cumbernauld-weather.co.uk/',  # Provisional
        'k_index_scale': 750e-9,  # Estimated, based on BGS Eskdalemuir site
        'license': cc3_by_nc_sa,
        'copyright': 'Cumbernauld Weather.',
        'attribution': 'Cumbernauld Weather, ' + \
                       'http://www.cumbernauld-weather.co.uk/',
        'line_color': [0.3, 0.3, 1],
        'data_types': {
            'MagData': {
                'default': 'realtime',
                'filtered': {
                    'channels': np.array(['H']),
                    'path': base_url + 'cwx/%Y/%m/cwx_%Y%m%d.txt',
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'aurorawatchnet',
                    'load_converter': load_awn_data,
                    'nominal_cadence': np.timedelta64(30000000, 'us'),
                    'units': 'T',
                    'sort': True,
                    'filter_function': remove_spikes,
                },
            },
        }
    },  # CWX

    'EXE': {
        # Met Office station
        'location': 'Exeter, UK',
        'latitude': 50.718414,
        'longitude': -3.537151,
        'elevation': np.nan,
        'start_time': np.datetime64('2014-11-13T00:00:00+0000'),
        'end_time': None,  # Still operational
        'url': 'http://www.metoffice.gov.uk/',  # Provisional
        'k_index_scale': 500e-9,  # Estimated, based on BGS Hartland site
        'k_index_filter': None,
        'license': 'This work is licensed under the Open Government Licence (OGL). To view a copy of this licence, visit http://www.nationalarchives.gov.uk/doc/open-government-licence',
        'copyright': u'\xa9 Crown Copyright, Met Office',
        'attribution': u'Please use the following attribution statements on any copies/reproductions etc: "\xa9 Crown Copyright, Met Office" (or "\xa9 British Crown Copyright, Met Office" where the reproduction is published outside of the UK)',
        'line_color': [186.0 / 255, 216.0 / 255, 10.0 / 255],
    },  # EXE

    'SID': {
        # University of Exeter mag at Norman Lockyer observatory
        'location': 'Sidmouth, UK',
        'latitude': Decimal('50.687911'),
        'longitude': Decimal('-3.219600'),
        'elevation': np.nan,
        'start_time': np.datetime64('2016-07-15T00:00:00+0000'),
        'end_time': None,  # Still operational
        'url': 'http://www.exeter.ac.uk/',  # Provisional
        'k_index_scale': 500e-9,  # Estimated, based on BGS Hartland site
        'k_index_filter': None,
        'license': cc3_by_nc_sa,
        'copyright': 'University of Exeter.',
        'attribution': 'University of Exeter and Norman Lockyer Observatory',
        'line_color': [0x00 / 255.0, 0x5d / 255.0, 0xab / 255.0],
    },  # SID

    'SUM': {
        'location': 'Sumburgh Head, UK',
        'latitude': Decimal('59.853'),
        'longitude': Decimal('-1.276'),
        'elevation': 248,
        'start_time': np.datetime64('2017-08-01T00:00Z'),
        'end_time': None,  # Still operational
        'k_index_scale': 1000e-9,  # From BGS Monthly Magnetic Bulletin value for Lerwick
        'license': cc3_by_nc_sa,
        'copyright': 'Lancaster University.',
        'attribution': 'Space and Plasma Physics group, ' + \
                       'Department of Physics, Lancaster University, UK.',
        'description': 'Raspberry Pi magnetometer system, hosted by the Shetland Amenity Trust. Made possible through funding from Gradconsult.',
        'line_color': [0, 0x65/255., 0xCC/255.], # Blue from Shetland flag
    },  # SUM
}

# Set activity color/thresholds unless already set.
default_activity_thresholds = np.array([0.0, 50.0, 100.0, 200.0]) * 1e-9
default_activity_colors = np.array([[0.2, 1.0, 0.2],  # green  
                                    [1.0, 1.0, 0.0],  # yellow
                                    [1.0, 0.6, 0.0],  # amber
                                    [1.0, 0.0, 0.0]])  # red

default_data_types = {
    'MagData': {
        'default': 'realtime',
        'realtime': {
            'channels': np.array(['H']),
            'path': base_url + '{site_lc}/%Y/%m/{site_lc}_%Y%m%d.txt',
            'duration': np.timedelta64(24, 'h'),
            'format': 'aurorawatchnet',
            'load_converter': load_awn_data,
            'nominal_cadence': np.timedelta64(30000000, 'us'),
            'units': 'T',
            'sort': True,
        },
        'realtime_baseline': {
            'channels': np.array(['H']),
            'path': (base_url +
                     'baseline/realtime/{site_lc}/{site_lc}_%Y.txt'),
            'duration': np.timedelta64(1, 'Y'),
            'load_converter': ap.data._generic_load_converter,
            'save_converter': ap.data._generic_save_converter,
            'nominal_cadence': np.timedelta64(1, 'D'),
            'units': 'T',
            # Information for generic load/save 
            'constructor': ap.magdata.MagData,
            'sort': False,
            'timestamp_method': 'YMD',
            'fmt': ['%04d', '%02d', '%02d', '%.2f'],
            'data_multiplier': 1000000000,  # Store as nT values
            # Information for making the data files
            'qdc_fit_duration': np.timedelta64(10, 'D'),
            'realtime_qdc': True,
        },
    },
    'MagQDC': {
        'qdc': {
            'channels': np.array(['H']),
            'path': base_url + 'qdc/{site_lc}/%Y/{site_lc}_qdc_%Y%m.txt',
            'duration': np.timedelta64(24, 'h'),
            'format': 'aurorawatchnet_qdc',
            'load_converter': ap.magdata.load_qdc_data,
            'save_converter': ap.data._generic_save_converter,
            'nominal_cadence': np.timedelta64(5, 's'),
            'units': 'T',
            'sort': False,
            # Information for generic load/save
            'constructor': ap.magdata.MagQDC,
            'timestamp_method': 's',
            'fmt': ['%d', '%.3f'],
            'delimiter': ' ',
            'data_multiplier': 1000000000,  # Store as nT values
            # Information for making the data files
            'qdc_fit_duration': np.timedelta64(10, 'D'),
            'realtime_qdc': True,
        },
    },
    'TemperatureData': {
        'realtime': {
            'channels': np.array(['Sensor temperature',
                                  'System temperature']),
            'path': base_url + '{site_lc}/%Y/%m/{site_lc}_%Y%m%d.txt',
            'duration': np.timedelta64(24, 'h'),
            'format': 'aurorawatchnet',
            'load_converter': load_awn_data,
            'nominal_cadence': np.timedelta64(30000000, 'us'),
            'units': six.u('\N{DEGREE SIGN}C'),
            'sort': True,
        },
    },
    'VoltageData': {
        'realtime': {
            'channels': np.array(['Supply voltage']),
            'path': base_url + '{site_lc}/%Y/%m/{site_lc}_%Y%m%d.txt',
            'duration': np.timedelta64(24, 'h'),
            'format': 'aurorawatchnet',
            'load_converter': load_awn_data,
            'nominal_cadence': np.timedelta64(30000000, 'us'),
            'units': 'V',
            'sort': True,
        },
    },
    'AuroraWatchActivity': {
        'default': 'realtime',
        'realtime': {
            'channels': np.array(['Activity']),
            'path': base_url + 'activity/aurorawatch/{site_lc}/{site_lc}_%Y.txt',
            'duration': np.timedelta64(1, 'Y'),
            'load_converter': ap.data._generic_load_converter,
            'save_converter': ap.data._generic_save_converter,
            'nominal_cadence': np.timedelta64(60, 'm'),
            'units': 'T',
            # Information for generic load/save
            'constructor': ap.auroralactivity.AuroraWatchActivity,
            'timestamp_method': 'YMDh',
            'fmt': ['%04d', '%02d', '%02d', '%02d', '%.2f'],
            'data_multiplier': 1000000000,  # Store as nT values
        }
    },
}

for s in sites:
    site_lc = s.lower()

    # Populate the data types
    if 'data_types' not in sites[s]:
        sites[s]['data_types'] = {}
    sdt = sites[s]['data_types']

    for dt in default_data_types:
        if dt not in sdt:
            sdt[dt] = {}
        for an, av in iteritems(default_data_types[dt]):
            if an not in sdt[dt]:
                sdt[dt][an] = \
                    copy.deepcopy(av)
                if an == 'default':
                    continue
                if not hasattr(sdt[dt][an]['path'], '__call__'):
                    sdt[dt][an]['path'] = \
                        sdt[dt][an]['path'].format(site_lc=site_lc)
            elif sdt[dt] is None:
                # None used as a placeholder to prevent automatic
                # population, now clean up
                del (sdt[dt])

    if 'activity_thresholds' not in sites[s]:
        sites[s]['activity_thresholds'] = default_activity_thresholds
    if 'activity_colors' not in sites[s]:
        sites[s]['activity_colors'] = default_activity_colors

    if 'k_index_filter' not in sites[s]:
        sites[s]['k_index_filter'] = k_index_filter_battery

for s in ('LAN1', 'EXE', 'SID', 'SUM', 'TEST1'):
    for dt in ('MagData', 'MagQDC'):
        for an in sites[s]['data_types'][dt]:
            ai = sites[s]['data_types'][dt][an]
            if isinstance(ai, six.string_types):
                continue
            ai['channels'] = np.array(['H', 'E', 'Z'])
            if dt == 'MagQDC':
                ai['fmt'] = ['%d'] + ['%.3f'] * len(ai['channels'])

    sites[s]['data_types']['MagData']['realtime_baseline']['fmt'] = \
        ['%04d', '%02d', '%02d', '%.2f', '%.2f', '%.2f']

project = {
    'name': 'AuroraWatch Magnetometer Network',
    'abbreviation': 'AWN',
    'url': 'http://aurorawatch.lancs.ac.uk/project-info/aurorawatchnet/',
    'sites': sites,
}

ap.add_project('AWN', project)
