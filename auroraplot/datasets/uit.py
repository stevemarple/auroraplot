# -*- coding: utf-8 -*-

import copy
import logging
import numpy as np
import os

import auroraplot as ap
import auroraplot.dt64tools as dt64
import auroraplot.magdata
from auroraplot.magdata import MagData


logger = logging.getLogger(__name__)

# Set the UIT password if possible
uit_password = None
for f in ('.uit_password', 'uit_password.txt'):
    uit_password_file = os.path.join(os.path.expanduser('~'), f)
    if os.path.exists(uit_password_file):
        try:
            fh = open(uit_password_file, 'r')
            try:
                uit_password = fh.readline().strip()
            finally:
                fh.close()
        except:
            logger.warn('Could not read UIT data access password')
            raise

path_fstr = 'http://flux.phys.uit.no/cgi-bin/mkascii.cgi?site=%(uit_site)s&year=%%Y&month=%%m&day=%%d&res=%(uit_res)s&pwd=%(uit_password)s&format=iagaUnix&comps=%(uit_comp)s&getdata=+Get+Data+'


def load_iaga_2000(file_name, archive_data, 
                      project, site, data_type, channels, start_time, 
                      end_time, **kwargs):
    assert data_type == 'MagData', 'Illegal data_type'
    iaga = ap.magdata.load_iaga_2000(file_name)
    data = []
    for c in channels:
        iaga_col_name = None
        if 'uit_channels' in archive_data:
            iaga_col_name = archive_data['uit_iaga_column'][c]
        else:
            for poss_name in [site.upper() + c.upper(),
                              '---' + c.upper()]:
                if poss_name in iaga['column_number']:
                    iaga_col_name = poss_name
                    break
        if iaga_col_name is None:
            raise Exception('Cannot find column for ' + c)

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


def uit_path(t, project, site, data_type, archive, channels):
    if uit_password is None:
        raise Exception(__name__ + '.uit_password must be set, ' + 
                        'to obtain a password see ' + 
                        'http://flux.phys.uit.no/div/DataAccess.html')

    # Expand the path format string with the specific UIT variables,
    # including password.
    a, d = copy.deepcopy(ap.get_archive_info(project, site, data_type, 
                                             archive=archive))
    d['uit_password'] = uit_password
    fstr = path_fstr % d
    return dt64.strftime(t, fstr)

sites = {
    'AMK': {
        'location': 'Ammassalik, Greenland',
        'latitude': 65.60,
        'longitude': -37.63,
        'data_types': {
            'MagData': {
                'xyz_10s': {
                    'channels': np.array(['X', 'Y', 'Z']),
                    'uit_site': 'amk1f',
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
                    'uit_site': 'amk1f',
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
        }, # AMK
    'AND': {
        'location': 'Andenes, Norway',
        'latitude': 69.30,
        'longitude': 16.03,
        'data_types': {
            'MagData': {
                'xyz_10s': {
                    'channels': np.array(['X', 'Y', 'Z']),
                    'uit_site': 'and1a',
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
                    'uit_site': 'and1a',
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
        }, # AND
    'BJN': {
        'location': 'Bjørnøya, Svalbard',
        'latitude': 74.50,
        'longitude': 19.00,
        'data_types': {
            'MagData': {
                'xyz_10s': {
                    'channels': np.array(['X', 'Y', 'Z']),
                    'uit_site': 'bjn1a',
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
                    'uit_site': 'bjn1a',
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
        }, # BJN
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
                    'load_converter': load_iaga_2000,
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
                    'load_converter': load_iaga_2000,
                    'nominal_cadence': np.timedelta64(10, 's'),
                    'units': 'T',
                    },
                }
            },
        }, # DOB
    'DON': {
        'location': 'Dønna, Norway',
        'latitude': 66.11,
        'longitude': 12.50,
        'data_types': {
            'MagData': {
                'xyz_10s': {
                    'channels': np.array(['X', 'Y', 'Z']),
                    'uit_site': 'don1a',
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
                    'uit_site': 'don1a',
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
        }, # DON
    'HOP': {
        'location': 'Hopen, Svalbard',
        'latitude': 76.51,
        'longitude': 25.01,
        'data_types': {
            'MagData': {
                'xyz_10s': {
                    'channels': np.array(['X', 'Y', 'Z']),
                    'uit_site': 'hop1a',
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
                    'uit_site': 'hop1a',
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
        }, # HOP
    'JCK': {
        'location': 'Jäckvik, Sweden',
        'latitude': 66.40,
        'longitude': 16.98,
        'data_types': {
            'MagData': {
                'xyz_10s': {
                    'channels': np.array(['X', 'Y', 'Z']),
                    'uit_site': 'jck1a',
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
                    'uit_site': 'jck1a',
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
        }, # JCK
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
                    'load_converter': load_iaga_2000,
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
                    'load_converter': load_iaga_2000,
                    'nominal_cadence': np.timedelta64(10, 's'),
                    'units': 'T',
                    },
                }
            },
        }, # KAR
    'LYR': {
        'location': 'Longyearbyen, Svalbard',
        'latitude': 78.2,
        'longitude': 15.83,
        'data_types': {
            'MagData': {
                'xyz_10s': {
                    'channels': np.array(['X', 'Y', 'Z']),
                    'uit_site': 'lyr2a',
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
                    'uit_site': 'lyr2a',
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
        }, # LYR
    'NAL': {
        'location': 'Ny Ålesund, Svalbard',
        'latitude': 78.92,
        'longitude': 11.93,
        'data_types': {
            'MagData': {
                'xyz_10s': {
                    'channels': np.array(['X', 'Y', 'Z']),
                    'uit_site': 'nal1a',
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
                    'uit_site': 'nal1a',
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
        }, # NAL
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
                    'load_converter': load_iaga_2000,
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
                    'load_converter': load_iaga_2000,
                    'nominal_cadence': np.timedelta64(10, 's'),
                    'units': 'T',
                    },
                }
            },
        }, # NOR
    'RVK': {
        'location': 'Rørvik, Norway',
        'latitude': 64.95,
        'longitude': 10.99,
        'data_types': {
            'MagData': {
                'xyz_10s': {
                    'channels': np.array(['X', 'Y', 'Z']),
                    'uit_site': 'rvk1a',
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
                    'uit_site': 'rvk1a',
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
        }, # RVK
    'SCO': {
        'location': 'Scoresbysund, Greenland',
        'latitude': 70.48,
        'longitude': -21.97,
        'data_types': {
            'MagData': {
                'xyz_10s': {
                    'channels': np.array(['X', 'Y', 'Z']),
                    'uit_site': 'sco8f',
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
                    'uit_site': 'sco8f',
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
        }, # SCO
    'SOL': {
        'location': 'Solund, Norway',
        'latitude': 61.08,
        'longitude': 4.84,
        'data_types': {
            'MagData': {
                'xyz_10s': {
                    'channels': np.array(['X', 'Y', 'Z']),
                    'uit_site': 'sol1a',
                    'uit_res': '10sec',
                    'uit_comp': 'XYZ',
                    'uit_iaga_column': {'X': '---X',
                                        'Y': '---Y',
                                        'Z': '---Z'},
                    'path': uit_path,
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'iaga2000',
                    'load_converter': load_iaga_2000,
                    'nominal_cadence': np.timedelta64(10, 's'),
                    'units': 'T',
                    },
                'hz_10s': {
                    'channels': np.array(['H', 'Z']),
                    'uit_site': 'sol1a',
                    'uit_res': '10sec',
                    'uit_comp': 'DHZ',
                    'uit_iaga_column': {'X': '---X',
                                        'Y': '---Y',
                                        'Z': '---Z'},
                    'path': uit_path,
                    'duration': np.timedelta64(24, 'h'),
                    'format': 'iaga2000',
                    'load_converter': load_iaga_2000,
                    'nominal_cadence': np.timedelta64(10, 's'),
                    'units': 'T',
                    },
                }
            },
        }, # SOL
    'SOR': {
        'location': 'Sørøya, Norway',
        'latitude': 70.54,
        'longitude': 22.22,
        'data_types': {
            'MagData': {
                'xyz_10s': {
                    'channels': np.array(['X', 'Y', 'Z']),
                    'uit_site': 'sor1a',
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
                    'uit_site': 'sor1a',
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
        }, # SOR
    'TDC': {
        'location': 'Tristan da Cunha',
        'latitude': -37.07,
        'longitude': -12.38,
        'data_types': {
            'MagData': {
                'xyz_10s': {
                    'channels': np.array(['X', 'Y', 'Z']),
                    'uit_site': 'tdc4f',
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
                    'uit_site': 'tdc4f',
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
    'TRO': {
        'location': 'Tromsø, Norway',
        'latitude': 69.66,
        'longitude': 18.94,
        'data_types': {
            'MagData': {
                'xyz_10s': {
                    'channels': np.array(['X', 'Y', 'Z']),
                    'uit_site': 'tro2a',
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
                    'uit_site': 'tro2a',
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
        }, # TRO
    }


for s in sites:
    sites[s]['data_types']['MagData']['default'] = 'xyz_10s'

ap.add_project('UIT', sites)



