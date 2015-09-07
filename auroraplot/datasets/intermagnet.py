import logging
import os
import traceback

# Python 2/3 compatibility
import six
import numpy as np

import auroraplot as ap
from auroraplot.magdata import MagData
from auroraplot.magdata import load_iaga_2000

logger = logging.getLogger(__name__)

ftp_hostname = 'ftp.intermagnet.org'

def convert_iaga_2002(file_name, archive_data, 
                      project, site, data_type, channels, start_time, 
                      end_time, **kwargs):
    assert data_type == 'MagData', 'Illegal data_type'
    iaga = load_iaga_2000(file_name)
    data = []
    for c in channels:
        iaga_col_name = site.upper() + c.upper()
        n = iaga['column_number'][iaga_col_name]
        data.append([float('nan') if x == '99999.00' else float(x) 
                     for x in iaga['data'][n]])
    
    data = np.array(data) * 1e-9
    r = MagData(project=project,
                site=site,
                channels=channels,
                start_time=start_time,
                end_time=end_time,
                sample_start_time=iaga['sample_time'], 
                sample_end_time=iaga['sample_time'] + \
                    archive_data['nominal_cadence'],
                integration_interval=None,
                nominal_cadence=archive_data['nominal_cadence'],
                data=data,
                units=archive_data['units'],
                sort=False)
    return r



sites = {
    'AAE': {
        'location': 'Addis Addba, Ethopia',
        'latitude': 9.035,
        'longitude': 38.766,
        'elevation': 2441.0,
        'reported': 'XYZF',
        'sensor_orientation': 'HDZF',
        'copyright': 'AAU (Ethiopia) / IPGP (France)',
        },

    'ABK': {
        'location': 'Abisko, Sweden',
        'latitude':  68.400,
        'longitude': 18.800,
        'reported': 'XYZF',
        'copyright': 'Sveriges geologiska undersokning',
        },

    'AIA': {
        'location': 'Vernadsky,  Argentine Islands, Antarctica',
        'latitude':  -65.300,
        'longitude': 295.700,
        'reported': 'XYZF',
        'copyright': 'Lviv Centre of Institute of Space Research',
        },

    'ALE': {
        'location': 'Alert, Canada',
        'latitude':  82.497,
        'longitude': 297.647,
        'elevation': 60.000,
        'reported': 'XYZF',
        'copyright': 'Geological Survey of Canada (GSC)',
        },

    'AMS': {
        'location': 'Martin de Vivies, Amsterdam Island',
        'latitude':  -37.796,
        'longitude': 77.574,
        'elevation': 50,
        'reported': 'XYZF',
        'sensor_orientation': 'HDZF',
        'copyright': 'EOST',
        },

    'ARS': {
        'location': 'Arti, Russia',
        'latitude':  59.400,
        'longitude': 58.600 ,
        'reported': 'HDZF',
        'copyright': 'Institute of Geophysics, Russia',
        },

    'BDV': {
        'location': 'Budkov, Czech Republic',
        'latitude':  49.100,
        'longitude': 14.000,
        'reported': 'XYZF',
        'copyright': 'Geofyzikalni ustav AV',
        },

    'BLC': {
        'location': 'Baker Lake, Canada',
        'latitude':  64.318,
        'longitude': 263.988,
        'elevation': 30.000,
        'reported': 'XYZF',
        'copyright': 'Geological Survey of Canada (GSC)',
        },

    # '': {
    #     'location': ', ',
    #     'latitude':  ,
    #     'longitude': ,
    #     'elevation': ,
    #     'copyright': '',
    #     },

    'LYC': {
        'location': ', ',
        'latitude':  64.600,
        'longitude': 18.700,
        'reported': 'XYZF',
        'copyright': 'Sveriges geologiska undersokning',
        },
    }


default_site_values = {
    'elevation': np.nan,
    'start_time': None,
    'end_time': None,
    }

for sa in sites:
    s = sites[sa]
    for k in default_site_values:
        if k not in s:
            s[k] = default_site_values[k]
    
    if 'data_types' not in s:
        s['data_types'] = {}

    if 'MagData' not in s['data_types']:
        s['data_types']['MagData'] = {}
    
    if 'preliminary' not in s['data_types']['MagData']:
        s['data_types']['MagData']['preliminary'] = {
            'channels': list(s['reported']),
            'path': 'ftp://' + ftp_hostname \
                + '/preliminary/%Y/%m/IAGA2002/' \
                + sa.lower() + '%Y%m%dvmin.min',
            'duration': np.timedelta64(24, 'h'),
            'converter': convert_iaga_2002,
            'nominal_cadence': np.timedelta64(60000000, 'us'),
            'units': 'T',
           }
        
ap.add_project('INTERMAGNET', sites)
