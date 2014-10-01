# -*- coding: utf-8 -*-

import copy
import logging
import numpy as np

import auroraplot as ap
import auroraplot.dt64tools as dt64
import auroraplot.magdata
from auroraplot.magdata import MagData
from auroraplot.magdata import convert_iaga_2000

logger = logging.getLogger(__name__)
uit_password = None

path_fstr = 'http://flux.phys.uit.no/cgi-bin/mkascii.cgi?site=%(uit_site)s&year=%%Y&month=%%m&day=%%d&res=%(uit_res)s&pwd=%(uit_password)s&format=iagaUnix&comps=%(uit_comp)s&getdata=+Get+Data+'


def uit_path(t, network, site, data_type, archive, channels):
    if uit_password is None:
        raise Exception(__name__ + '.uit_password must be set')

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
    }



ap.add_network('UIT', sites)



