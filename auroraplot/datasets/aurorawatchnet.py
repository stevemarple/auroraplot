import copy
import os
import logging

# Python 2/3 compatibility
import six
try:
    from urllib.request import urlopen
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse
    from urllib import urlopen

import numpy as np

import auroraplot as ap
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
    data[np.logical_or(data < 0, data > 10)] = np.nan
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
            data = np.loadtxt(uh, unpack=True)
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
                                   np.timedelta64(630,'s'), 3)
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


def abbreviate_aurorawatchnet(ax, title=True, labels=True):
    '''Abbreviate AURORAWATCHNET to AWN on plots.'''
    if title:
        ax.set_title(ax.get_title().replace('AURORAWATCHNET', 'AWN'))
    if labels:
        tll = ax.yaxis.get_ticklabels() # tick label list
        labels = [ tl.get_text() for tl in tll]
        # Only update the labels if one or more needs chaning as this
        # breaks normal numerical axis laelling otherwise.
        if True in map(lambda x: x.startswith('AURORAWATCHNET'), labels):
            labels = map(lambda x: x.replace('AURORAWATCHNET', 'AWN'), labels)
            ax.yaxis.set_ticklabels(labels)


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
        'k_index_scale': 650e-9, # Estimated
        'k_index_filter': None,
        'copyright': 'Steve Marple.',
        'license': cc3_by_nc_sa,
        'attribution': 'Data provided by Steve Marple.', 
        'line_color': [0, 0.6, 0],
        }, # LAN1

    'LAN3': {
        'location': 'Lancaster, UK',
        'latitude': 54.01,
        'longitude': -2.77,
        'elevation': 93,
        'start_time': np.datetime64('2012-12-18T00:00Z'),
        'end_time': None, # Still operational
        'k_index_scale': 650e-9, # Estimated
        'license': cc3_by_nc_sa,
        'copyright': 'Lancaster University.',
        'attribution': 'Space and Plasma Physics group, ' + \
            'Department of Physics, Lancaster University, UK.',
        'line_color': [1, 0, 0],
        }, # LAN3

    'ORM': {
        'location': 'Ormskirk, UK',
        'latitude': 53.569195, 
        'longitude': -2.887264,
        'elevation': None,
        'start_time': np.datetime64('2013-08-01T00:00Z'),
        'end_time': None, # Still operational
        'k_index_scale': 650e-9, # Estimated
        'license': cc3_by_nc_sa,
        'copyright': 'Lancaster University.',
        'attribution': 'Space and Plasma Physics group, ' + \
            'Department of Physics, Lancaster University, UK.',
        'line_color': [186.0/255, 216.0/255, 10.0/255],
        }, # ORM
    
    'TEST1': {
        'location': 'Lancaster, UK',
        'latitude': 54.0,
        'longitude': -2.78,
        'elevation': 27,
        'start_time': np.datetime64('2013-11-10T00:00Z'),
        'end_time': None, # Still operational
        'k_index_scale': 650e-9, # Estimated
        'license': cc3_by_nc_sa,
        'copyright': 'Lancaster University.',
        'attribution': 'Space and Plasma Physics group, ' + \
            'Department of Physics, Lancaster University, UK.',
        }, # TEST1

    'BRA': {
        'location': 'Brae, Shetland, UK',
        'latitude': 60.395869,
        'longitude': -1.351124000000027,
        'elevation': 11,
        'start_time': np.datetime64('2014-01-01T00:00:00+0000'),
        'end_time': None, # Still operational
        'k_index_scale': 1000e-9, # Estimated, based on BGS Lerwick site
        'license': cc3_by_nc_sa,
        'copyright': 'Lancaster University.',
        'attribution': 'Space and Plasma Physics group, ' + \
            'Department of Physics, Lancaster University, UK.',
        }, # BRA

    'SAN': {
        'location': 'Sanday, UK',
        'latitude': 59.25110830191925,
        'longitude': -2.5873320735991,
        'elevation': 16.366,
        'start_time': np.datetime64('2014-01-01T00:00:00+0000'),
        'end_time': None, # Still operational
        'k_index_scale': 950e-9, # Estimated
        'license': cc3_by_nc_sa,
        'copyright': 'Lancaster University.',
        'attribution': 'Space and Plasma Physics group, ' + \
            'Department of Physics, Lancaster University, UK.',
        }, # SAN

    'TOB': {
        'location': 'Tobermory, Mull, UK',
        'latitude': 56.62415194965067,
        'longitude': -6.068624798208475,
        'elevation': 43,
        'start_time': np.datetime64('2014-01-01T00:00:00+0000'),
        'end_time': None, # Still operational
        'k_index_scale': 850e-9, # Estimated
        'license': cc3_by_nc_sa,
        'copyright': 'Lancaster University.',
        'attribution': 'Space and Plasma Physics group, ' + \
            'Department of Physics, Lancaster University, UK.',
        }, # TOB

     'WHI': {
        'location': 'Whitehaven, Cumbria, UK',
        'latitude': 54.543384,
        'longitude': -3.5610000000000355,
        'elevation': 132,
        'start_time': np.datetime64('2014-01-01T00:00:00+0000'),
        'end_time': None, # Still operational
        'k_index_scale': 680e-9, # Estimated
        'license': cc3_by_nc_sa,
        'copyright': 'Lancaster University.',
        'attribution': 'Space and Plasma Physics group, ' + \
            'Department of Physics, Lancaster University, UK.',
        'line_color': [.6, .6, .6],
        }, # WHI

    'ALT': {
        'location': 'Altrincham, UK',
        'latitude': 53.381988,
        'longitude': -2.3597310000000107,
        'elevation': 68,
        'start_time': np.datetime64('2014-01-01T00:00:00+0000'),
        'end_time': None, # Still operational
        'k_index_scale': 600e-9, # Estimated
        'license': cc3_by_nc_sa,
        'copyright': 'Lancaster University.',
        'attribution': 'Space and Plasma Physics group, ' + \
            'Department of Physics, Lancaster University, UK.',
        }, # ALT

    'MAL': {
        'location': 'Malpas, UK',
        'latitude': 53.029658,
        'longitude': -2.760221999999999,
        'elevation': 122,
        'start_time': np.datetime64('2014-01-01T00:00:00+0000'),
        'end_time': None, # Still operational
        'k_index_scale': 600e-9, # Estimated
        'license': cc3_by_nc_sa,
        'copyright': 'Lancaster University.',
        'attribution': 'Space and Plasma Physics group, ' + \
            'Department of Physics, Lancaster University, UK.',
        }, # MAL

    'ASH': {
        'location': 'Ashbourne, UK',
        'latitude': 53.021899,
        'longitude': -1.7287959999999885,
        'elevation': 157,
        'start_time': np.datetime64('2014-01-01T00:00:00+0000'),
        'end_time': None, # Still operational
        'k_index_scale': 600e-9, # Estimated
        'license': cc3_by_nc_sa,
        'copyright': 'Lancaster University.',
        'attribution': 'Space and Plasma Physics group, ' + \
            'Department of Physics, Lancaster University, UK.',
        }, # ASH

        'PEL': {
        'location': 'Pelsall, UK',
        'latitude': 52.623806,
        'longitude': -1.9490200000000186,
        'elevation': 143,
        'start_time': np.datetime64('2014-01-01T00:00:00+0000'),
        'end_time': None, # Still operational
        'k_index_scale': 580e-9, # Estimated
        'license': cc3_by_nc_sa,
        'copyright': 'Lancaster University.',
        'attribution': 'Space and Plasma Physics group, ' + \
            'Department of Physics, Lancaster University, UK.',
        }, # PEL

    'BRE': {
        'location': 'Brecon, UK',
        'latitude': 52.02851,
        'longitude': -3.2026879999999665,
        'elevation': 121,
        'start_time': np.datetime64('2014-01-01T00:00:00+0000'),
        'end_time': None, # Still operational
        'k_index_scale': 550e-9, # Estimated
        'license': cc3_by_nc_sa,
        'copyright': 'Lancaster University.',
        'attribution': 'Space and Plasma Physics group, ' + \
            'Department of Physics, Lancaster University, UK.',
        'line_color': [119.0/255, 11.0/255, 0],
        }, # BRE

    'CAN': {
        'location': 'Canterbury, UK',
        'latitude': 51.260914,
        'longitude': 1.084820000000036,
        'elevation': 48,
        'start_time': np.datetime64('2014-01-01T00:00:00+0000'),
        'end_time': None, # Still operational
        'k_index_scale': 500e-9, # Estimated, based on BGS Hartland site
        'license': cc3_by_nc_sa,
        'copyright': 'Lancaster University.',
        'attribution': 'Space and Plasma Physics group, ' + \
            'Department of Physics, Lancaster University, UK.',
        'line_color': [64.0/255, 224.0/255, 208.0/255],
        }, # CAN

    'CWX': {
        # Private station operated by Cumbernauld Weather
        'location': 'Cumbernauld, UK',
        'latitude': 55 + (56.1/60),
        'longitude': -(4 + (2.2/60)),
        'elevation': 82,
        'start_time': np.datetime64('2014-01-01T00:00:00+0000'),
        'end_time': None, # Still operational
        'url': 'http://www.cumbernauld-weather.co.uk/', # Provisional
        'k_index_scale': 750e-9, # Estimated, based on BGS Eskdalemuir site
        'license': cc3_by_nc_sa,
        'copyright': 'Cumbernauld Weather.',
        'attribution': 'Cumbernauld Weather, ' + \
            'http://www.cumbernauld-weather.co.uk/',
        'line_color': [0.3, 0.3, 1],
        }, # CWX

    'EXE': {
        # Met Office station
        'location': 'Exeter, UK',
        'latitude': 50.718414, 
        'longitude': -3.537151,
        'elevation': np.nan,
        'start_time': np.datetime64('2014-11-13T00:00:00+0000'),
        'end_time': None, # Still operational
        'url': 'http://www.metoffice.gov.uk/', # Provisional
        'k_index_scale': 500e-9, # Estimated, based on BGS Hartland site
        'k_index_filter': None,
        'license': cc3_by_nc_sa,
        'copyright': 'Met Office.',
        'attribution': 'Met Office, ' + \
            'http://www.metoffice.gov.uk/',
        'line_color': [186.0/255, 216.0/255, 10.0/255],
        }, # EXE

    }

# Set activity color/thresholds unless already set.
default_activity_thresholds = np.array([0.0, 50.0, 100.0, 200.0]) * 1e-9
default_activity_colors = np.array([[0.2, 1.0, 0.2],  # green  
                                    [1.0, 1.0, 0.0],  # yellow
                                    [1.0, 0.6, 0.0],  # amber
                                    [1.0, 0.0, 0.0]]) # red

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
            'data_multiplier': 1000000000, # Store as nT values
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
            'nominal_cadence': np.timedelta64(5, 's'),
            'units': 'T',
            'sort': False,
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
            'channels': np.array(['Battery voltage']),
            'path': base_url + '{site_lc}/%Y/%m/{site_lc}_%Y%m%d.txt',
            'duration': np.timedelta64(24, 'h'),
            'format': 'aurorawatchnet',
            'load_converter': load_awn_data,
            'nominal_cadence': np.timedelta64(30000000, 'us'),
            'units': 'V',
            'sort': True,
            },
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
        for an,av in default_data_types[dt].iteritems():
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
                del(sdt[dt])

    if 'activity_thresholds' not in sites[s]:
        sites[s]['activity_thresholds'] = default_activity_thresholds
    if 'activity_colors' not in sites[s]:
        sites[s]['activity_colors'] = default_activity_colors

    if 'k_index_filter' not in sites[s]:
         sites[s]['k_index_filter'] = k_index_filter_battery

ap.add_project('AURORAWATCHNET', sites)



