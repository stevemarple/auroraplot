# -*- coding: utf-8 -*-

import copy
import logging
import numpy as np

import auroraplot as ap
import auroraplot.dt64tools as dt64
import auroraplot.magdata
from auroraplot.magdata import MagData
from auroraplot.magdata import load_iaga_2000

logger = logging.getLogger(__name__)
uit_password = None

path_fstr = 'http://flux.phys.uit.no/cgi-bin/mkascii.cgi?site=%(uit_site)s&year=%%Y&month=%%m&day=%%d&res=%(uit_res)s&pwd=%(uit_password)s&format=iagaUnix&comps=%(uit_comp)s&getdata=+Get+Data+'


def convert_iaga_2000(file_name, archive_data, 
                      network, site, data_type, channels, start_time, 
                      end_time, **kwargs):
    assert data_type == 'MagData', 'Illegal data_type'
    iaga = load_iaga_2000(file_name)
    data = []
    for c in channels:
        if archive_data.has_key('uit_channels'):
            iaga_col_name = archive_data['uit_iaga_column'][c]
        else:
            iaga_col_name = site.upper() + c.upper()

        n = iaga['column_number'][iaga_col_name]
        data.append(map(lambda x: 
                        float('nan') if x == '99999.00' else float(x),   
                        iaga['data'][n]))
    
    data = np.array(data) * 1e-9
    r = MagData(network=network,
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


def uit_path(t, network, site, data_type, archive, channels):
    if uit_password is None:
        raise Exception(__name__ + '.uit_password must be set, ' + 
                        'to obtain a password see ' + 
                        'http://flux.phys.uit.no/div/DataAccess.html')

    # Expand the path format string with the specific UIT variables,
    # including password.
    a, d = copy.deepcopy(ap.get_archive_info(network, site, data_type, 
                                             archive=archive))
    d['uit_password'] = uit_password
    fstr = path_fstr % d
    return dt64.strftime(t, fstr)

sites = {
    'DOB': {
        'location': 'Dombås, Norway',
        'latitude': 62.07,
        'longitude': 9.11,
        'data_types': {
            'MagData': {
                'xyz_10s': {
                    'channels': np.array(['X', 'Y', 'Z']),
                    'uit_site': 'dob1a',
                    'uit_res': '10sec',
                    'uit_comp': 'XYZ',
                    'path': uit_path,
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'iaga2000',
                    'converter': convert_iaga_2000,
                    'nominal_cadence': np.timedelta64(10, 's'),
                    'units': 'T',
                    },
                'hz_10s': {
                    'channels': np.array(['H', 'Z']),
                    'uit_site': 'dob1a',
                    'uit_res': '10sec',
                    'uit_comp': 'DHZ',
                    'path': uit_path,
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'iaga2000',
                    'converter': convert_iaga_2000,
                    'nominal_cadence': np.timedelta64(10, 's'),
                    'units': 'T',
                    },
                }
            },
        }, # DOB
    'KAR': {
        'location': 'Karmøy, Norway',
        'latitude': 59.21,
        'longitude': 5.24,
        'data_types': {
            'MagData': {
                'xyz_10s': {
                    'channels': np.array(['X', 'Y', 'Z']),
                    'uit_site': 'kar1a',
                    'uit_res': '10sec',
                    'uit_comp': 'XYZ',
                    'path': uit_path,
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'iaga2000',
                    'converter': convert_iaga_2000,
                    'nominal_cadence': np.timedelta64(10, 's'),
                    'units': 'T',
                    },
                'hz_10s': {
                    'channels': np.array(['H', 'Z']),
                    'uit_site': 'kar1a',
                    'uit_res': '10sec',
                    'uit_comp': 'DHZ',
                    'path': uit_path,
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'iaga2000',
                    'converter': convert_iaga_2000,
                    'nominal_cadence': np.timedelta64(10, 's'),
                    'units': 'T',
                    },
                }
            },
        }, # KAR
    'NOR': {
        'location': 'Nordkapp, Norway',
        'latitude': 71.09,
        'longitude': 25.79,
        'data_types': {
            'MagData': {
                'xyz_10s': {
                    'channels': np.array(['X', 'Y', 'Z']),
                    'uit_site': 'nor1a',
                    'uit_res': '10sec',
                    'uit_comp': 'XYZ',
                    'uit_iaga_column': {'X': '---X',
                                        'Y': '---Y',
                                        'Z': '---Z'},
                    'path': uit_path,
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'iaga2000',
                    'converter': convert_iaga_2000,
                    'nominal_cadence': np.timedelta64(10, 's'),
                    'units': 'T',
                    },
                'hz_10s': {
                    'channels': np.array(['H', 'Z']),
                    'uit_site': 'nor1a',
                    'uit_res': '10sec',
                    'uit_comp': 'DHZ',
                    'uit_iaga_column': {'D': '---D',
                                        'H': '---H',
                                        'Z': '---Z'},
                    'path': uit_path,
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'iaga2000',
                    'converter': convert_iaga_2000,
                    'nominal_cadence': np.timedelta64(10, 's'),
                    'units': 'T',
                    },
                }
            },
        }, # NOR
    }



ap.add_network('UIT', sites)



