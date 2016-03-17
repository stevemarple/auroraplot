# -*- coding: utf-8 -*-

import copy
import logging
import numpy as np

import auroraplot as ap
import auroraplot.dt64tools as dt64
import auroraplot.magdata

from auroraplot.magdata import MagData
from auroraplot.magdata import load_iaga_2000
from auroraplot.datasets.uit import load_iaga_2000

import auroraplot.datasets.uit
from auroraplot.datasets.uit import uit_path

logger = logging.getLogger(__name__)
uit_password = None

sites = {
    'BFE': {
        'location': 'Brorfelde, Denmark',
        'latitude': 62.07,
        'longitude': 9.11,
        'data_types': {
            'MagData': {
                'xyz_10s': {
                    'channels': np.array(['X', 'Y', 'Z']),
                    'uit_site': 'bfe6d',
                    'uit_res': '10sec',
                    'uit_comp': 'XYZ',
                    'path': uit_path,
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'iaga2000',
                    'load_converter': load_iaga_2000,
                    'nominal_cadence': np.timedelta64(10, 's'),
                    'units': 'T',
                    },
                'hz_10s': {
                    'channels': np.array(['H', 'Z']),
                    'uit_site': 'bfe6d',
                    'uit_res': '10sec',
                    'uit_comp': 'DHZ',
                    'path': uit_path,
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'iaga2000',
                    'load_converter': load_iaga_2000,
                    'nominal_cadence': np.timedelta64(10, 's'),
                    'units': 'T',
                    },
                }
            },
        }, # BFE
    'ROE': {
        'location': 'Rømø, Denmark',
        'latitude': 62.07,
        'longitude': 9.11,
        'data_types': {
            'MagData': {
                'xyz_10s': {
                    'channels': np.array(['X', 'Y', 'Z']),
                    'uit_site': 'roe1d',
                    'uit_res': '10sec',
                    'uit_comp': 'XYZ',
                    'path': uit_path,
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'iaga2000',
                    'load_converter': load_iaga_2000,
                    'nominal_cadence': np.timedelta64(10, 's'),
                    'units': 'T',
                    },
                'hz_10s': {
                    'channels': np.array(['H', 'Z']),
                    'uit_site': 'roe1d',
                    'uit_res': '10sec',
                    'uit_comp': 'DHZ',
                    'path': uit_path,
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'iaga2000',
                    'load_converter': load_iaga_2000,
                    'nominal_cadence': np.timedelta64(10, 's'),
                    'units': 'T',
                    },
                }
            },
        }, # ROE    
    'TDC': {
        'location': 'Tristan da Cunha',
        'latitude': -37.07,
        'longitude': -12.38,
        'data_types': {
            'MagData': {
                'xyz_10s': {
                    'channels': np.array(['X', 'Y', 'Z']),
                    'uit_site': 'tdc4d',
                    'uit_res': '10sec',
                    'uit_comp': 'XYZ',
                    'path': uit_path,
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'iaga2000',
                    'load_converter': load_iaga_2000,
                    'nominal_cadence': np.timedelta64(10, 's'),
                    'units': 'T',
                    },
                'hz_10s': {
                    'channels': np.array(['H', 'Z']),
                    'uit_site': 'tdc4d',
                    'uit_res': '10sec',
                    'uit_comp': 'DHZ',
                    'path': uit_path,
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'iaga2000',
                    'load_converter': load_iaga_2000,
                    'nominal_cadence': np.timedelta64(10, 's'),
                    'units': 'T',
                    },
                }
            },
        }, # TDC
    }


for s in sites:
    sites[s]['data_types']['MagData']['default'] = 'xyz_10s'

ap.add_project('DTU', sites)



