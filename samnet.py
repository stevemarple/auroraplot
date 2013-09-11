import os
import numpy as np
import urllib2

import awplot

data_dir = '/data/samnet'

def load_samnet_data():
    pass

def convert_rt_data(file_name, archive_data,
                    network, site, data_type, channels, start_time, 
                    end_time, **kwargs):
    assert(data_type == 'MagData')

    try:
        chan_tup = tuple(archive_data['channels'])
        col_idx = []
        for c in channels:
            col_idx.append(1 + chan_tup.index(c))
        
        uh = urllib2.urlopen(file_name)
        try:
            data = np.loadtxt(uh, unpack=True)
            sample_start_time = start_time + (np.timedelta64(1, 's') * data[0])
            sample_end_time = sample_start_time + np.timedelta64(1, 's')
            integration_interval = np.timedelta64(1, 's').repeat(len(data[0]))
            r = awplot.MagData(network=network,
                               site=site,
                               channels=channels,
                               start_time=start_time,
                               end_time=end_time,
                               sample_start_time=sample_start_time, 
                               sample_end_time=sample_end_time,
                               integration_interval=integration_interval,
                               nominal_cadence=archive_data['nominal_cadence'],
                               data=data[col_idx]*1e9,
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
            print('Could not open ' + file_name)
    return None

sites = {
    'LAN': {    
        'location': 'Lancaster, UK',
        'latitude': 54.01,
        'longitude': -2.77,
        'elevation': 93,
        'data_types': {
            'MagData': {
                '1s': {
                    'channels': ['H', 'D', 'Z'],
                    #'path': os.path.join(data_dir, 
                    #                     'samnet/1s_archive/%Y/%m/la%d%m%y.dgz'),
                    'path': 'http://www.dcs.lancs.ac.uk/iono/miadata/magnetometer/samnet/5s_archive/%Y/%m/la%d%m%Y.5s.gz',
                    'duration': np.timedelta64(1, 'D'),
                    'format': 'samnet_1s',
                    'converter': load_samnet_data,
                    'nominal_cadence': np.timedelta64(1, 's'),
                    'units': 'T',
                    },
                '5s': {
                    'channels': ['H', 'D', 'Z'],
                    'path': os.path.join(data_dir,
                                         'samnet/1s_archive/%Y/%m/la%d%m%Y.5s.gz'),
                    'duration': np.timedelta64(1, 'D'),
                    'format': 'samnet_5s',
                    'converter': load_samnet_data,
                    'nominal_cadence': np.timedelta64(5, 's'),
                    'units': 'T',
                    },
                'realtime': {
                    'channels': ['H', 'D', 'Z'],
                    #'path': os.path.join(data_dir,
                    #                     'realtime/lan/%Y/%m/lan%Y%m%d.rt'),
                    'path': 'http://spears.lancs.ac.uk/~marple/realtime/lan/%Y/%m/lan%Y%m%d.rt',
                    'duration': np.timedelta64(1, 'D'),
                    'format': 'samnetrt',
                    'converter': convert_rt_data,
                    'nominal_cadence': np.timedelta64(1, 's'),
                    'units': 'T',
                    },
                },
            'MagQDC': {
                'qdc': {
                    'channels': ['H', 'D', 'Z'],
                    'path': os.path.join(data_dir, 
                                         'samnet/activity/quiet/%Y/la00%m%Y.5s'),
                    'duration': np.timedelta64(1, 'D'),
                    'format': 'samnet_qdc',
                    'converter': load_samnet_data,
                    'nominal_cadence': np.timedelta64(5, 's'),
                    'units': 'T',
                    },
                },
            },
        'start_time': np.datetime64('2003-02-20T17:00Z'),
        'end_time': None, # Still operational
        'acknowledgement': {'short': 'Lancaster University, UK.',
                            'long': 'The Sub-Auroral Magnetometer Network data (SAMNET) is operated by the Space Plasma Environment and Radio Science (SPEARS) group, Department of Physics, Lancaster University.'},
        }
    }

for k in sites:
    sites[k]['data_types']['MagData']['default'] = \
        sites[k]['data_types']['MagData']['1s']
    
    
awplot.add_network('SAMNET', sites)

