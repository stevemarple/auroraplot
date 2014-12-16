"""Analyse and plot space weather datasets."""


__author__ = 'Steve Marple'
__version__ = '0.2.1'
__license__ = 'PSF'

import logging
import six
import numpy as np
import auroraplot.dt64tools as dt64

logger = logging.getLogger(__name__)


epoch64_us = np.datetime64('1970-01-01T00:00:00Z','us')

networks = { }

NaN = float('nan')
NaT = np.timedelta64('NaT', 'us')

colors = ['b', 'g', 'r']

# def safe_eval(s):
#     '''Like eval but without allowing the user to access builtin
#     functions or locals.'''
#     return eval(s, {'__builtins__': None}, {})


#class Instrument(object):
#

# def copy_dict(d, *keys):
#     '''Make a copy of only the `keys` from dictionary `d`.'''
#     return {key: d[key] for key in keys}


def add_network(network_name, sites):
    '''
    Helper function for datasets to register network and site
    information. To allow local customisation of file location a call
    to the "add_network_hook" function in the auroraplot_custom module
    is loaded. This function, if it exists, should modify the
    registered site information to suit local policy.
    '''

    if network_name in networks:
        networks[network_name].update(sites)
    else:
        networks[network_name] = sites
        
    if hasattr(auroraplot_custom, 'add_network_hook'):
        auroraplot_custom.add_network_hook(network_name=network_name)


def str_units(val, unit, prefix=None, sep=None, degrees_dir=None,
              fmt='%(adj)g%(sep)s%(prefix)s%(unit)s%(dir)s', ascii=True,
              wantstr=True):
    '''Return a string formatted with its units
    val: data value
    unit: SI or other unit
    prefix: standard prefix to use with unit, or None for automatic selection
    sep: separator, ' ' unless units are degrees when it is empty
    degrees_dir: direction indicator for degrees, length 2 iterable eg, ['N', 'S']
    fmt: format specifier
    ascii: if true use u as prefix for micro
    wantstr: if true return th formatted string, else return dict of info
    '''
    
    is_degrees = unit in (six.u('deg'), 
                          six.u('degrees'), 
                          six.u('\N{DEGREE SIGN}'))
    prefixes = {'y': -24, # yocto
                'z': -21, # zepto
                'a': -18, # atto
                'f': -15, # femto 
                'p': -12, # pico
                'n': -9,  # nano
                six.u('\N{MICRO SIGN}'): -6, # micro
                'u': -6,  # micro
                'm': -3,  # milli
                'c': -2,  # centi
                'd': -1,  # deci
                '': 0,
                'da': 1,  # deca
                'h': 2,   # hecto
                'k': 3,   # kilo
                'M': 6,   # mega
                'G': 9,   # giga
                'T': 12,  # tera
                'P': 15,  # peta
                'E': 18,  # exa
                'Z': 21,  # zetta
                'Y': 24,  # yotta
                }

    d = {'sep': sep,
         'prefix': prefix,
         'unit': unit,
         'val': val}
    if is_degrees:
        if sep is None:
            d['sep'] = ''
        if prefix is None:
            d['prefix'] = '' # Do not calculate automatically
            d['mul'] = 1
            # prefix = ''
    # elif unicode(unit) == six.u('\N{DEGREE SIGN}C'):
    elif unit == six.u('\N{DEGREE SIGN}C'):
        # Don't calculate prefixes with degrees C
        if sep is None:
            d['sep'] = ' '
        if prefix is None:
            d['prefix'] = '' # Do not calculate automatically
            d['mul'] = 1   
    elif sep is None:
        d['sep'] = ' '

    if d['prefix'] is None:
        if unit in ('', '%'):
            logmul = 0
        elif np.isfinite(val) and val != 0:
            log10_val = np.log10(val)
            logmul = int(np.floor(np.abs(np.spacing(log10_val)) + 
                                  (log10_val) / 3.0) * 3)
        else:
            logmul = 1

        # Find matching prefix
        for k in prefixes:
            if prefixes[k] == logmul:
                d['prefix'] = k
                break
        assert d['prefix'] is not None, 'prefix should not be None'
        d['mul'] = 10 ** logmul

    else:
        d['mul'] = 10 ** prefixes[d['prefix']]

    if ascii and d['mul'] >= 9e-07 and d['mul'] < 11e-7:
        d['prefix'] = 'u'

    if d['mul'] == 1:
        d['adj'] = val
    else:
        d['adj'] = val / d['mul']

    if is_degrees and degrees_dir is not None:
        # include N/S or other direction indicator
        if val >= 0:
            d['dir'] = degrees_dir[0] 
        else:
            d['dir'] = degrees_dir[1]
            d['adj'] = -d['adj']
    else:
        d['dir'] = ''

    if wantstr:
        return fmt % d
    else:
        d['fmt'] = fmt
        d['str'] = fmt % d
        d['fmtunit'] = d['prefix']
        if unit is not None:
            d['fmtunit'] += unit
        return d


def has_site_info(network, site, info):
    # Sanity checking
    if network not in networks:
        raise Exception('Unknown network')
    elif site not in networks[network]:
        raise Exception('Unknown site')
    return info in networks[network][site]


def get_site_info(network, site, info=None):
    # Sanity checking
    if network not in networks:
        raise Exception('Unknown network')
    elif site not in networks[network]:
        raise Exception('Unknown site')
    if info is None:
        return networks[network][site]
    elif info not in networks[network][site]:
        raise Exception('Unknown info')
    else:
        return networks[network][site][info]

def get_archive_info(network, site, data_type, **kwargs):
    '''
    Get relevant details about a data archive
    
    network: name of the network (upper case)
    
    site: site abbreviation (upper case)
    
    data_type: class name of the data type to be loaded
    
    The following optional parameters are recognised: 
    
    archive: name of the archive. Required if more than one archive is
        present and there is not an archive called "default".


    Returns: 
        A tuple containing the archive name and a dictionary of
        archive details. This includes the following keys:

        channels: numpy array of channel names (or possibly numbers)

        path: strftime format path or URL to load/save data

        converter: function reference for converting a file into a
            single object of type data_type. Used by load_data(). Not
            required if load_function is included.
    
        load_function: function reference used to load data. If not
            None then load_data() hans over the entire data loading
            process to this function.

        nominal_cadence: numpy timedelta64 interval indicating maximum
            normal interval between samples. Used to mark missing data
            when plotting.

        format: name of the data file format (optional).
    '''

    site_info = get_site_info(network, site)

    # Sanity checking
    if data_type not in site_info['data_types']:
        raise Exception('Unknown data_type')
    
    if kwargs.get('archive') is None:
        if len(site_info['data_types'][data_type]) == 1:
            # Only one archive, so default is implicit
            archive = list(site_info['data_types'][data_type].keys())[0]
        elif 'default' in site_info['data_types'][data_type]:
            # Use explicit default
            if isinstance(site_info['data_types'][data_type]['default'],
                          six.string_types):
                archive = site_info['data_types'][data_type]['default']
            else:
                archive = 'default'
        else:
            raise Exception('archive must be specified')
    else:
        archive = kwargs.get('archive')

    if archive not in site_info['data_types'][data_type]:
        raise Exception('Unknown archive')

    # archive data
    return (archive, site_info['data_types'][data_type][archive])


def load_data(network, site, data_type, start_time, end_time, **kwargs):
    '''Load data. 
    network: name of the network (upper case)

    site: site abbreviation (upper case)

    data_type: class name of the data type to be loaded

    start_time: start time (inclusive) of the data set

    end_time: end time (exclusive) of the data set
    
    The following optional parameters are recognised: 
    
    archive: name of the archive. Required if more than one archive is
        present and there is not an archive called "default".

    channels: data channel(s) to load. All are loaded if not specified

    path: URL or file path, specified as a strftime format specifier.
        Alternatively can be a function reference which is passed the
        time and returns the filename. If given this overrides the
        standard load path.

    load_function: Pass responsibility for loading the data to the given
        function reference, after validating the input parameters.
        
    '''
    archive, ad = get_archive_info(network, site, data_type, **kwargs)
    channels = kwargs.get('channels')
    if channels:
        # Could be as single channel name or a list of channels
        if isinstance(channels, six.string_types):
            if channels not in ad['channels']:
                raise Exception('Unknown channel')
        else:
            for c in channels:
                if c not in ad['channels']:
                    raise Exception('Unknown channel')
    else:
        channels = ad['channels']

    path = kwargs.get('path', ad['path'])

    load_function = kwargs.get('load_function', ad.get('load_function'))


    kwargs2 = kwargs.copy()
    kwargs2['archive'] = archive
    kwargs2['channels'] = channels
    kwargs2['load_function'] = load_function
    kwargs2['path'] = path
        
    if load_function:
        # Pass responsibility for loading to some other
        # function. Parameters have already been checked.
        return load_function(network, site, data_type, start_time, 
                             end_time, **kwargs2)

    data = []
    t = dt64.floor(start_time, ad['duration'])
    while t < end_time:
        t2 = t + ad['duration']
        if hasattr(path, '__call__'):
            # Function: call it with relevant information to get the path
            file_name = path(t, network=network, site=site, 
                             data_type=data_type, archive=archive,
                             channels=channels)
        else:
            file_name = dt64.strftime(t, path)

        logger.info('loading ' + file_name)

        try:
            tmp = ad['converter'](file_name, 
                                  ad,
                                  network=network,
                                  site=site, 
                                  data_type=data_type, 
                                  start_time=t, 
                                  end_time=t2, **kwargs2)
            if tmp is not None:
                data.append(tmp)
        except Exception as e:
            logger.info('Could not load ' + file_name)
            logger.debug(str(e))

                        
        
        t = t2
        
    if len(data) == 0:
        return None

    r = concatenate(data).sort(inplace=True)
    r.extract(inplace=True, 
              start_time=start_time, 
              end_time=end_time, 
              channels=channels)
    return r


def concatenate(objs, sort=True):
    obj_type = type(objs[0])
    network = objs[0].network
    site = objs[0].site
    channels = objs[0].channels
    start_time = []
    end_time = []
    sam_st_list = [] # sample start times
    sam_et_list = [] # sample start times
    integration_interval = []
    cadence_list = []
    data_list = []
    units = objs[0].units
    for a in objs:
        assert(type(a) == obj_type)
        assert(a.network == network)
        assert(a.site == site)
        assert(np.all(a.channels == channels))
        assert(a.units == units)
        start_time.append(a.start_time)
        end_time.append(a.end_time)
        sam_st_list.append(a.sample_start_time)
        sam_et_list.append(a.sample_end_time)
        if a.integration_interval is None:
            # All integration intervals must be discarded
            integration_interval = None
        elif integration_interval is not None:
            integration_interval.append(a.integration_interval)
        cadence_list.append(a.nominal_cadence)
        data_list.append(a.data)

    if integration_interval is not None:
        integration_interval=np.concatenate(integration_interval, axis=1)
    return obj_type(network=network,
                    site=site,
                    channels=channels,
                    start_time=np.min(start_time),
                    end_time=np.max(end_time),
                    sample_start_time=np.concatenate(sam_st_list),
                    sample_end_time=np.concatenate(sam_et_list),
                    integration_interval=integration_interval,
                    nominal_cadence=np.max(cadence_list),
                    data=np.concatenate(data_list, axis=1),
                    units=units,
                    sort=sort)

# Initialise

try:
    import auroraplot_custom
except ImportError as e:
    # No custom module
    logger.debug('auroraplot_custom.py not found')
    auroraplot_custom = {}
except Exception as e:
    # Error loading custom module
    logger.error('Could not load custom module:' + str(e))
    auroraplot_custom = {}

    
