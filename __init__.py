import numpy as np
import dt64tools as dt64

epoch64_ns = np.datetime64('1970-01-01T00:00:00Z','ns')

networks = { }

NaN = float('nan')
NaT = np.timedelta64('NaT', 'ns')

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

verbose = True
def add_network(network_name, sites):
    if networks.has_key(network_name):
        networks[network_name].update(sites)
    else:
        networks[network_name] = sites


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
    
    is_degrees = unicode(unit) in (u'deg', u'degrees', u'\N{DEGREE SIGN}')
    prefixes = {'y': -24, # yocto
                'z': -21, # zepto
                'a': -18, # atto
                'f': -15, # femto 
                'p': -12, # pico
                'n': -9,  # nano
                u'\N{MICRO SIGN}': -6, # micro
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
    elif sep is None:
        d['sep'] = ' '

    if d['prefix'] is None:
        if np.isfinite(val) and val != 0:
            logmul = int(np.floor((np.spacing(np.log10(val)) + 
                                           np.log10(val)) / 3.0) * 3)
        else:
            logmul = 1
        # Find matching prefix
        for k in prefixes:
            if prefixes[k] == logmul:
                d['prefix'] = k
                break
        assert d['prefix'] is not None
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
        d['fmtunit'] = d['prefix'] + unit
        return d

### TODO: allow load path to be overriden by function argument
# def load_data(network, site, data_type, 
#               start_time, end_time,
#               archive=None, channels=None, path=None, 
#               resolution=None, verbose=None):
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
        Alternatively can be a function reference which is passed the time 
        and returns the filename.

    load_function: Pass responsibility for loading the data to the given
        function reference, after validating the input parameters.
        
    verbose: flag to indicate if verbose messages should be
        printed. If None then the global verbose parameter is checked.

    '''
    # Sanity checking
    if not networks.has_key(network):
        raise Exception('Unknown network')
    elif not networks[network].has_key(site):
        raise Exception('Unknown site')
    elif not networks[network][site]['data_types'].has_key(data_type):
        raise Exception('Unknown data_type')

    if kwargs.get('archive') is None:
        if len(networks[network][site]['data_types'][data_type]) == 1:
            # Only one archive, so default is implicit
            archive = networks[network][site]['data_types'][data_type].keys()[0]
        elif networks[network][site]['data_types'][data_type].has_key('default'):
            # Use explicit default
            archive = 'default'
        else:
            raise Exception('archive must be specified')
    else:
        archive = kwargs.get('archive')

    if not networks[network][site]['data_types'][data_type].has_key(archive):
        raise Exception('Unknown archive')

    # archive data
    ad = networks[network][site]['data_types'][data_type][archive] 
    channels = kwargs.get('channels')
    if channels:
        # Could be as single channel name or a list of channels
        if isinstance(channels, basestring):
            if channels not in ad['channels']:
                raise Exception('Unknown channel')
        else:
            for c in channels:
                if c not in ad['channels']:
                    raise Exception('Unknown channel')
    else:
        channels = ad['channels']

    verbose = kwargs.get('verbose', globals()['verbose'])
    path = kwargs.get('path', ad['path'])

    load_function = kwargs.get('load_function', ad.get('load_function'))


    kwargs2 = kwargs.copy()
    kwargs2['archive'] = archive
    kwargs2['channels'] = channels
    kwargs2['load_function'] = load_function
    kwargs2['path'] = path
    kwargs2['verbose'] = verbose
        
    if load_function:
        # Pass responsibility for loading to some other
        # function. Parameters have already been checked and verbose
        # is set to the value the user desires.
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
            file_name = dt64.strftime64(t, path)

        if verbose:
            print('loading ' + file_name)

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
            if verbose:
                print('Could not load ' + file_name + ' ' + str(e))

                        
        
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
