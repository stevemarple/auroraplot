import os
import numpy as np
import urllib2

import auroraplot as ap
from auroraplot.magdata import MagData as MagData

data_dir = '/data/samnet'

def convert_samnet_data(file_name, archive_data, 
                        network, site, data_type, channels, start_time, 
                        end_time, **kwargs):

    chan_tup = tuple(archive_data['channels'])
    col_idx = []
    for c in channels:
        col_idx.append(chan_tup.index(c))
    nominal_cadence_s = (archive_data['nominal_cadence'] / 
                         np.timedelta64(1, 's'))
    try:
        # if file_name.startswith('/'):
        #     #uh = urllib2.urlopen('file:' + file_name)
        # else:
        #     uh = urllib2.urlopen(file_name)
        try:
            conv = lambda s: (s.strip().startswith('9999.9') and np.nan) \
                or float(s.strip())
            # data = np.loadtxt(uh,
            comments = ap.get_site_info(network, site, 'samnet_code')
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
                network=network,
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
            if kwargs.get('verbose'):
                print('Could not read ' + file_name)
                print(str(e))

        finally:
            # uh.close()
            pass
    except Exception as e:
        if kwargs.get('verbose'):
            print('Could not open ' + file_name)
            # print(str(e))
    return None



def convert_rt_data(file_name, archive_data,
                    network, site, data_type, channels, start_time, 
                    end_time, **kwargs):
    assert(data_type == 'MagData')
    try:
        chan_tup = tuple(archive_data['channels'])
        col_idx = []
        for c in channels:
            col_idx.append(1 + chan_tup.index(c))

        if file_name.startswith('/'):
            uh = urllib2.urlopen('file:' + file_name)
        else:
            uh = urllib2.urlopen(file_name)

        try:
            data = np.loadtxt(uh, unpack=True)
            sample_start_time = start_time + (np.timedelta64(1, 's') * data[0])
            sample_end_time = sample_start_time + np.timedelta64(1, 's')
            integration_interval = np.ones([len(channels), 
                                            len(sample_start_time)],
                                            dtype='m8[s]')
            r = MagData(network=network,
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
            if kwargs.get('verbose'):
                print('Could not read ' + file_name)
                print(str(e))
        finally:
            uh.close()
    except Exception as e:
        if kwargs.get('verbose'):
            print('Could not open ' + file_name + ': ' + str(e))
    return None

sites = {
    'BOR1': {
        'location': 'Borok, CIS',
        'latitude': 58.03,
        'longitude': 38.33,
        'elevation': np.nan,
        'start_time': np.datetime64('1998-12-03T00:00:00+0000'),
        'end_time': None,
        'copyright': 'Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space Plasma Environment and Radio Science (SPEARS) group, Department of Physics, Lancaster University.',
        'samnet_code': 'bo',
        },
    'CRK1': {
        'location': 'Crooktree, UK',
        'latitude': 57.09,
        'longitude': -2.64,
        'elevation': np.nan,
        'start_time': np.datetime64('2002-05-17T00:00:00+0000'),
        'end_time': None,
        'copyright': 'Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space Plasma Environment and Radio Science (SPEARS) group, Department of Physics, Lancaster University.',
        'samnet_code': 'cr',
        },
    'ESK1': {
        'location': 'Eskdalemuir, UK',
        'latitude': 55.32,
        'longitude': -3.2,
        'elevation': np.nan,
        'start_time': np.datetime64('2001-01-01T00:00:00+0000'),
        'end_time': None,
        'copyright': 'Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space Plasma Environment and Radio Science (SPEARS) group, Department of Physics, Lancaster University.',
        'samnet_code': 'es',
        },
    'FAR1': {
        'location': ', Faroes',
        'latitude': 62.05,
        'longitude': -7.02,
        'elevation': np.nan,
        'start_time': np.datetime64('1987-10-01T00:00:00+0000'),
        'end_time': None,
        'copyright': 'Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space Plasma Environment and Radio Science (SPEARS) group, Department of Physics, Lancaster University.',
        'samnet_code': 'fa',
        },
    'GML1': {
        'location': 'Glenmore Lodge, UK',
        'latitude': 57.16,
        'longitude': -3.68,
        'elevation': np.nan,
        'start_time': np.datetime64('1987-10-01T00:00:00+0000'),
        'end_time': np.datetime64('2002-05-15T00:00:00+0000'),
        'copyright': 'Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space Plasma Environment and Radio Science (SPEARS) group, Department of Physics, Lancaster University.',
        'samnet_code': 'gm',
        },
    'HAD1': {
        'location': 'Hartland, UK',
        'latitude': 50.99,
        'longitude': -4.48,
        'elevation': np.nan,
        'start_time': np.datetime64('2001-01-01T00:00:00+0000'),
        'end_time': None,
        'copyright': 'Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space Plasma Environment and Radio Science (SPEARS) group, Department of Physics, Lancaster University.',
        'samnet_code': 'hd',
        },
    'HAN1': {
        'location': 'Hankasalmi, Finland',
        'latitude': 62.3,
        'longitude': 26.65,
        'elevation': np.nan,
        'start_time': np.datetime64('1997-01-03T00:00:00+0000'),
        'end_time': np.datetime64('2005-05-16T00:00:00+0000'),
        'copyright': 'Finnish Meteorological Institute / Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space Plasma Environment and Radio Science (SPEARS) group, Department of Physics, Lancaster University. Original data provided by the Finnish Meteorological Institute.',
        'samnet_code': 'ha',
        },
    'HAN3': {
        'location': 'Hankasalmi, Finland',
        'latitude': 62.2539,
        'longitude': 26.5967,
        'elevation': np.nan,
        'start_time': np.datetime64('2005-08-18T00:00:00+0000'),
        'end_time': None,
        'copyright': 'Finnish Meteorological Institute / Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space Plasma Environment and Radio Science (SPEARS) group, Department of Physics, Lancaster University. Original data provided by the Finnish Meteorological Institute.',
        'samnet_code': 'ha',
        },
    'HLL1': {
        'location': 'Hella, Iceland',
        'latitude': 63.77,
        'longitude': -20.56,
        'elevation': np.nan,
        'start_time': np.datetime64('1998-10-06T00:00:00+0000'),
        'end_time': None,
        'copyright': 'Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space Plasma Environment and Radio Science (SPEARS) group, Department of Physics, Lancaster University.',
        'samnet_code': 'hl',
        },
    'KIL1': {
        'location': 'Kilpisjarvi, Finland',
        'latitude': 69.02,
        'longitude': 20.79,
        'elevation': np.nan,
        'start_time': np.datetime64('1997-12-01T00:00:00+0000'),
        'end_time': None,
        'copyright': 'Finnish Meteorological Institute / Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space Plasma Environment and Radio Science (SPEARS) group, Department of Physics, Lancaster University. Original data provided by the Finnish Meteorological Institute.',
        'samnet_code': 'ki',
        },
    'KVI1': {
        'location': 'Kvistaberg, Sweden',
        'latitude': 59.5,
        'longitude': 17.63,
        'elevation': np.nan,
        'start_time': np.datetime64('1987-10-13T00:00:00+0000'),
        'end_time': np.datetime64('2003-01-01T00:00:00+0000'),
        'copyright': 'Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space Plasma Environment and Radio Science (SPEARS) group, Department of Physics, Lancaster University.',
        'samnet_code': 'kv',
        },
    'LAN1': {
        'location': 'Lancaster, UK',
        'latitude': 54.01,
        'longitude': -2.77,
        'elevation': np.nan,
        'start_time': np.datetime64('2003-02-20T17:00:00+0000'),
        'end_time': None,
        'copyright': 'Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space Plasma Environment and Radio Science (SPEARS) group, Department of Physics, Lancaster University.',
        'samnet_code': 'la',
        },
    'LER1': {
        'location': 'Lerwick, UK',
        'latitude': 60.13,
        'longitude': -1.18,
        'elevation': np.nan,
        'start_time': np.datetime64('2001-01-01T00:00:00+0000'),
        'end_time': None,
        'copyright': 'Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space Plasma Environment and Radio Science (SPEARS) group, Department of Physics, Lancaster University.',
        'samnet_code': 'le',
        },
    'LNC2': {
        'location': 'LNCcaster, UK',
        'latitude': 54.01,
        'longitude': -2.77,
        'elevation': np.nan,
        'start_time': np.datetime64('2006-12-24T00:00:00+0000'),
        'end_time': np.datetime64('2008-06-27T00:00:00+0000'),
        'copyright': 'Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space Plasma Environment and Radio Science (SPEARS) group, Department of Physics, Lancaster University.',
        'samnet_code': 'ln',
        },
    'NOR1': {
        'location': 'Nordli, Norway',
        'latitude': 64.37,
        'longitude': 13.36,
        'elevation': np.nan,
        'start_time': np.datetime64('1987-10-01T00:00:00+0000'),
        'end_time': np.datetime64('2003-03-01T00:00:00+0000'),
        'copyright': 'Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space Plasma Environment and Radio Science (SPEARS) group, Department of Physics, Lancaster University.',
        'samnet_code': 'no',
        },
    'NUR1': {
        'location': 'Nurmijarvi, Finland',
        'latitude': 60.51,
        'longitude': 24.66,
        'elevation': np.nan,
        'start_time': np.datetime64('1987-10-01T00:00:00+0000'),
        'end_time': np.datetime64('2003-01-01T00:00:00+0000'),
        'copyright': 'Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space Plasma Environment and Radio Science (SPEARS) group, Department of Physics, Lancaster University.',
        'samnet_code': 'nu',
        },
    'NUR3': {
        'location': 'Nurmijarvi, Finland',
        'latitude': 60.5,
        'longitude': 24.65,
        'elevation': np.nan,
        'start_time': np.datetime64('2003-01-01T00:00:00+0000'),
        'end_time': None,
        'copyright': 'Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space Plasma Environment and Radio Science (SPEARS) group, Department of Physics, Lancaster University.',
        'samnet_code': 'nu',
        },
    'OUJ2': {
        'location': 'Oulujarvi, Finland',
        'latitude': 64.52,
        'longitude': 27.23,
        'elevation': np.nan,
        'start_time': np.datetime64('2003-01-01T00:00:00+0000'),
        'end_time': None,
        'copyright': 'Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space Plasma Environment and Radio Science (SPEARS) group, Department of Physics, Lancaster University.',
        'samnet_code': 'oj',
        },
    'OUL1': {
        'location': 'Oulu, Finland',
        'latitude': 65.1,
        'longitude': 25.85,
        'elevation': np.nan,
        'start_time': np.datetime64('1987-10-01T00:00:00+0000'),
        'end_time': np.datetime64('2003-01-01T00:00:00+0000'),
        'copyright': 'Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space Plasma Environment and Radio Science (SPEARS) group, Department of Physics, Lancaster University.',
        'samnet_code': 'ou',
        },
    'UPS2': {
        'location': 'Uppsala, Sweden',
        'latitude': 59.9,
        'longitude': 17.35,
        'elevation': np.nan,
        'start_time': np.datetime64('2003-01-01T00:00:00+0000'),
        'end_time': None,
        'copyright': 'Geological Survey of Sweden / Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space Plasma Environment and Radio Science (SPEARS) group, Department of Physics, Lancaster University. Original data provided by Geological Survey of Sweden.',
        'samnet_code': 'up',
        },
    'YOR1': {
        'location': 'York, UK',
        'latitude': 53.95,
        'longitude': -1.05,
        'elevation': np.nan,
        'start_time': np.datetime64('1987-10-01T00:00:00+0000'),
        'end_time': np.datetime64('2006-03-30T12:00:00+0000'),
        'copyright': 'Lancaster University',
        'license': 'Data users are not entitled to distribute data to third parties outside their own research teams without requesting permission from Prof. F. Honary. Similarly, SAMNET data should not become part of a distributed database without permission first being sought. Commerical use prohibited.',
        'attribution': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space Plasma Environment and Radio Science (SPEARS) group, Department of Physics, Lancaster University.',
        'samnet_code': 'yo',
        },
    
    }



# Set activity color/thresholds unless already set.
default_activity_thresholds = np.array([0.0, 50.0, 100.0, 200.0]) * 1e-9
default_activity_colors = np.array([[0.2, 1.0, 0.2],  # green  
                                    [1.0, 1.0, 0.0],  # yellow
                                    [1.0, 0.6, 0.0],  # amber
                                    [1.0, 0.0, 0.0]]) # red

# Set default archive details
#for k in sites:
#    sites[k]['data_types']['MagData']['default'] = \
#        sites[k]['data_types']['MagData']['1s']
    

for s in sites:
    s_lc = s.lower()
    sc = sites[s]['samnet_code'] # Two-letter lower-case abbreviation
    sites[s]['data_types'] = {
        'MagData': {
            '1s': {
                'channels': ['H', 'D', 'Z'],
                'path': os.path.join(data_dir, '1s_archive/%Y/%m/' +
                                     sc + '%d%m%y.dgz'),
                'duration': np.timedelta64(24, 'h'),
                'converter': convert_samnet_data,
                'nominal_cadence': np.timedelta64(1, 's'),
                'units': 'T',
                },
            '5s': {
                'channels': ['H', 'D', 'Z'],
                'path': os.path.join(data_dir, '5s_archive/%Y/%m/' + 
                                     sc + '%d%m%Y.5s.gz'),
                'duration': np.timedelta64(24, 'h'),
                'converter': convert_samnet_data,
                'nominal_cadence': np.timedelta64(5, 's'),
                'units': 'T',
                },
                'realtime': {
                    'channels': ['H', 'D', 'Z'],
                    'path': '/data/samnet/realtime/' + s_lc + \
                        '/%Y/%m/' + s_lc + '%Y%m%d.rt',
                    'duration': np.timedelta64(24, 'h'),
                    'converter': convert_rt_data,
                    'nominal_cadence': np.timedelta64(1, 's'),
                    'units': 'T',
                    },
                },
            'MagQDC': {
                'qdc': {
                    'channels': ['H', 'D', 'Z'],
                    'path': os.path.join(data_dir, 'activity/quiet/%Y/' +
                                         sc + '00%m%Y.5s'),
                    'duration': np.timedelta64(24, 'h'),
                    'converter': convert_samnet_data,
                    'nominal_cadence': np.timedelta64(5, 's'),
                    'units': 'T',
                    },
                },
        }

    if not sites[s].has_key('activity_thresholds'):
        sites[s]['activity_thresholds'] = default_activity_thresholds
    if not sites[s].has_key('activity_colors'):
        sites[s]['activity_colors'] = default_activity_colors

ap.add_network('SAMNET', sites)

