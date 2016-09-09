import copy
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
import auroraplot.data
import auroraplot.riodata
import auroraplot.tools
from auroraplot.riodata import RioData
from auroraplot.riodata import RioQDC

logger = logging.getLogger(__name__)

remote_base_url = 'http://www.riometer.net/data/'
local_base_url = '/data/'

def check_rio_data(data):
    #data[np.logical_or(data < -0.0001, data > 0.0001)] = np.nan
    return data


def load_rn_data(file_name, archive_data, 
                  project, site, data_type, channels, start_time, 
                  end_time, **kwargs):
    '''Convert RiometerNet data to match standard data type

    data: RioData or other similar format data object
    archive: name of archive from which data was loaded
    archive_info: archive metadata
    '''

    data_type_info = {
        'RioData': {
            'class': RioData,
            'col_offset': 1,
            'scaling': 1,
            'data_check': check_rio_data,
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

            data = ap.loadtxt(uh)

            sample_start_time = ap.epoch64_us + \
                (np.timedelta64(1000000, 'us') * data[0])
            # end time and integration interval are guesstimates
            sample_end_time = sample_start_time + np.timedelta64(1000000, 'us')
            integration_interval = 1E6*np.ones([len(channels), 
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
                sort=True,
                processing=[])
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



cc3_by_nc_sa = 'This work is licensed under the Creative Commons ' + \
    'Attribution-NonCommercial-ShareAlike 3.0 Unported License. ' + \
    'To view a copy of this license, visit ' + \
    'http://creativecommons.org/licenses/by-nc-sa/3.0/deed.en_GB.'

unknown_license = 'Please ask permission from the PI of the instrument ' + \
                  'before using this data in a publication.'

sites = {
    'TEST6': {
        'location': 'Lancaster, UK',
        'latitude': 54.0,
        'longitude': -2.78,
        'elevation': 27,
        'start_time': np.datetime64('2016-08-11T00:00Z'),
        'end_time': None, # Still operational
        'copyright': 'Mat Beharrell.',
        'license': unknown_license,
        'attribution': 'Unknown.', 
        'line_color': [0, 0.6, 0],
        }, # TEST6

    }

###### Needed?
# Set activity color/thresholds unless already set.
default_activity_thresholds = np.array([0.0, 50.0, 100.0, 200.0]) * 1e-9
default_activity_colors = np.array([[0.2, 1.0, 0.2],  # green  
                                    [1.0, 1.0, 0.0],  # yellow
                                    [1.0, 0.6, 0.0],  # amber
                                    [1.0, 0.0, 0.0]]) # red
#################################################################
channels = np.arange(1,50).astype('str')
default_data_types = {
    'RioPower': {
        'default': 'remote archive',
        'local capture': {
            'channels': channels,
            'path': local_base_url + 'capture/{site_lc}/%Y/%m/{site_lc}_%Y%m%d.txt',
            'duration': np.timedelta64(24, 'h'),
            'format': 'aurorawatchnet',
            'load_converter': load_rn_data,
            'nominal_cadence': np.timedelta64(1000000, 'us'),
            'units': 'dBm',
            'sort': True,
            },
        'local archive': {
            'channels': channels,
            'path': local_base_url + '%Y/%m/{site_lc}_%Y%m%d.txt',
            'duration': np.timedelta64(24, 'h'),
            'format': 'aurorawatchnet',
            'load_converter': load_rn_data,
            'nominal_cadence': np.timedelta64(1000000, 'us'),
            'units': 'dBm',
            'sort': True,
            },
        'remote archive': {
            'channels': channels,
            'path': remote_base_url + '%Y/%m/{site_lc}_%Y%m%d.txt',
            'duration': np.timedelta64(24, 'h'),
            'format': 'aurorawatchnet',
            'load_converter': load_rn_data,
            'nominal_cadence': np.timedelta64(1000000, 'us'),
            'units': 'dBm',
            'sort': True,
            },
        },
    'RioQDC': {
        'local archive': {
            'channels': channels,
            'path': local_base_url+'qdc/{site_lc}/%Y/{site_lc}_qdc_%Y%m%d.txt',
            'duration': np.timedelta64(14, 'D'),
            'format': 'riometernet_qdc',
            'load_converter': ap.riodata.load_qdc_data,
            'save_converter': ap.riodata._save_baseline_data,
            'nominal_cadence': np.timedelta64(5, 's'),
            'units': 'dBm',
            'sort': False,
            },
        'remote qdc': {
            'channels': channels,
            'path': remote_base_url+'qdc/{site_lc}/%Y/{site_lc}_qdc_%Y%m%d.txt',
            'duration': np.timedelta64(14, 'D'),
            'format': 'riometernet_qdc',
            'load_converter': ap.riodata.load_qdc_data,
            'nominal_cadence': np.timedelta64(5, 's'),
            'units': 'dBm',
            'sort': False,
            },
        },
#    'TemperatureData': {
#        'realtime': {
#            'channels': np.array(['Sensor temperature', 
#                                  'System temperature']),
#            'path': base_url + '{site_lc}/%Y/%m/{site_lc}_%Y%m%d.txt',
#            'duration': np.timedelta64(24, 'h'),
#            'format': 'aurorawatchnet',
#            'load_converter': load_awn_data,
#            'nominal_cadence': np.timedelta64(30000000, 'us'),
#            'units': six.u('\N{DEGREE SIGN}C'),
#            'sort': True,
#            },
#        },
#    'VoltageData': {
#        'realtime': {
#            'channels': np.array(['Battery voltage']),
#            'path': base_url + '{site_lc}/%Y/%m/{site_lc}_%Y%m%d.txt',
#            'duration': np.timedelta64(24, 'h'),
#            'format': 'aurorawatchnet',
#            'load_converter': load_awn_data,
#            'nominal_cadence': np.timedelta64(30000000, 'us'),
#            'units': 'V',
#            'sort': True,
#            },
#        },
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
        for an,av in iteritems(default_data_types[dt]):
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

#    if 'activity_thresholds' not in sites[s]:
#        sites[s]['activity_thresholds'] = default_activity_thresholds
#    if 'activity_colors' not in sites[s]:
#        sites[s]['activity_colors'] = default_activity_colors

#    if 'k_index_filter' not in sites[s]:
#         sites[s]['k_index_filter'] = k_index_filter_battery

project = {
    'name': 'Riometer Network',
    'abbreviation': 'RN',
    'url': 'http://',
    'sites': sites,
}

ap.add_project('RN', project)



