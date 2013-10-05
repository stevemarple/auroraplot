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
            data = np.loadtxt(file_name, 
                              unpack=True, 
                              converters={0: conv, 1: conv, 2: conv},
                              comments=site.lower()[:2])
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
                    #                     '1s_archive/%Y/%m/la%d%m%y.dgz'),
                    'path': 'http://www.dcs.lancs.ac.uk/iono/miadata/magnetometer/samnet/1s_archive/%Y/%m/la%d%m%y.dgz',
                    'duration': np.timedelta64(24, 'h'),
                    'converter': convert_samnet_data,
                    'nominal_cadence': np.timedelta64(1, 's'),
                    'units': 'T',
                    },
                '5s': {
                    'channels': ['H', 'D', 'Z'],
                    'path': os.path.join(data_dir,
                                         '5s_archive/%Y/%m/la%d%m%Y.5s.gz'),
                    'duration': np.timedelta64(24, 'h'),
                    'converter': convert_samnet_data,
                    'nominal_cadence': np.timedelta64(5, 's'),
                    'units': 'T',
                    },
                'realtime': {
                    'channels': ['H', 'D', 'Z'],
                    #'path': os.path.join(data_dir,
                    #                     'realtime/lan/%Y/%m/lan%Y%m%d.rt'),
                    'path': '/data/samnet/realtime/lan/%Y/%m/lan%Y%m%d.rt',
                    'duration': np.timedelta64(24, 'h'),
                    'converter': convert_rt_data,
                    'nominal_cadence': np.timedelta64(1, 's'),
                    'units': 'T',
                    },
                },
            'MagQDC': {
                'qdc': {
                    'channels': ['H', 'D', 'Z'],
                    'path': os.path.join(data_dir, 
                                         'activity/quiet/%Y/la00%m%Y.5s'),
                    'duration': np.timedelta64(24, 'h'),
                    'converter': convert_samnet_data,
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

# Set default archive details
for k in sites:
    sites[k]['data_types']['MagData']['default'] = \
        sites[k]['data_types']['MagData']['1s']
    
# Set activity color/thresholds unless already set.
default_activity_thresholds = np.array([0.0, 50.0, 100.0, 200.0]) * 1e-9
default_activity_colors = np.array([[0.2, 1.0, 0.2],  # green  
                                    [1.0, 1.0, 0.0],  # yellow
                                    [1.0, 0.6, 0.0],  # amber
                                    [1.0, 0.0, 0.0]]) # red
for s in sites:
    if not sites[s].has_key('activity_thresholds'):
        sites[s]['activity_thresholds'] = default_activity_thresholds
    if not sites[s].has_key('activity_colors'):
        sites[s]['activity_colors'] = default_activity_colors

ap.add_network('SAMNET', sites)

