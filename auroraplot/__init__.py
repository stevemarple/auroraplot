"""Analyse and plot space weather datasets."""


__author__ = 'Steve Marple'
__version__ = '0.4.6'
__license__ = 'PSF'

import copy
import gzip
import importlib
import logging
import netrc
import re
import shutil
import six
import traceback
import warnings

try:
    # Python 3.x
    from urllib.parse import quote
    from urllib.parse import urlparse
    from urllib.parse import urlunparse
    from urllib.request import urlopen

except ImportError:
    # Python 2.x
    from urllib import quote
    from urllib import urlopen
    from urlparse import urlparse
    from urlparse import urlunparse

import numpy as np
import os
from tempfile import NamedTemporaryFile

import auroraplot.dt64tools as dt64

logger = logging.getLogger(__name__)


epoch64_us = np.datetime64('1970-01-01T00:00:00Z','us')

projects = { }

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


def add_project(project_name, sites):
    '''
    Helper function for datasets to register project and site
    information. To allow local customisation of file location a call
    to the "add_project_hook" function in the auroraplot_custom module
    is loaded. This function, if it exists, should modify the
    registered site information to suit local policy.
    '''

    if project_name in projects:
        projects[project_name].update(sites)
    else:
        projects[project_name] = sites
        
    if hasattr(auroraplot_custom, 'add_project_hook'):
        auroraplot_custom.add_project_hook(project_name=project_name)


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
    wantstr: if true return the formatted string, else return dict of info
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


def format_project_site(project, site):
    return project + ' / ' + site


def has_site_info(project, site, info):
    # Sanity checking
    if project not in projects:
        raise Exception('Unknown project (%s)' % project)
    elif site not in projects[project]:
        raise Exception('Unknown site (%s)' % site)
    return info in projects[project][site]


def get_site_info(project, site, info=None):
    # Sanity checking
    if project not in projects:
        raise Exception('Unknown project (%s)' % project)
    elif site not in projects[project]:
        raise Exception('Unknown site (%s)' % site)
    if info is None:
        return projects[project][site]
    elif info not in projects[project][site]:
        raise Exception('Unknown info (%s)' % info)
    else:
        return projects[project][site][info]


def get_archive_info(project, site, data_type, archive=None):
    '''
    Get relevant details about a data archive
    
    project: name of the project (upper case)
    
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
            None then load_data() hands over the entire data loading
            process to this function.

        nominal_cadence: numpy.timedelta64 interval indicating maximum
            normal interval between samples. Used to mark missing data
            when plotting.

        format: name of the data file format (optional).
    '''

    site_info = get_site_info(project, site)

    # Sanity checking
    if data_type not in site_info['data_types']:
        raise ValueError('Unknown data_type (%s)' % data_type)
    
    if archive is None or archive == 'default':
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
            raise TypeError('archive must be specified (multiple choices and no default)')

    if archive not in site_info['data_types'][data_type]:
        raise ValueError('Unknown archive (%s) for %s' \
                             % (archive, format_project_site(project, site)))

    # archive data
    return (archive, site_info['data_types'][data_type][archive])


def load_data(project, 
              site, 
              data_type, 
              start_time, 
              end_time, 
              archive=None,
              channels=None,
              path=None,
              load_function=None,
              raise_all=False,
              cadence=None,
              aggregate=None):
    '''Load data. 
    project: name of the project (upper case)

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
    archive, ad = get_archive_info(project, site, data_type, 
                                   archive=archive)
    cad_units = dt64.get_units(ad['nominal_cadence'])
    start_time = start_time.astype('datetime64[%s]' % cad_units)
    end_time = end_time.astype('datetime64[%s]' % cad_units)

    if channels is None:
        channels = ad['channels']
    else:
        # Could be as single channel name or a list of channels
        if isinstance(channels, six.string_types):
            if channels not in ad['channels']:
                raise Exception('Unknown channel')
        else:
            for c in channels:
                if c not in ad['channels']:
                    raise Exception('Unknown channel')

    if path is None:
        path = ad['path']

    if load_function is None:
        load_function = ad.get('load_function')
        
    if load_function:
        # Pass responsibility for loading to some other
        # function. Parameters have already been checked.
        return load_function(project, 
                             site, 
                             data_type, 
                             start_time, 
                             end_time,
                             archive=archive,
                             channels=channels,
                             path=path,
                             raise_all=raise_all,
                             cadence=cadence,
                             aggregate=aggregate)


    data = []
    for t in dt64.dt64_range(dt64.floor(start_time, ad['duration']), 
                             end_time, 
                             ad['duration']):
        # A local copy of the file to be loaded, possibly an
        # uncompressed version.
        temp_file_name = None

        t2 = t + ad['duration']
        if hasattr(path, '__call__'):
            # Function: call it with relevant information to get the path
            file_name = path(t, project=project, site=site, 
                             data_type=data_type, archive=archive,
                             channels=channels)
        else:
            file_name = dt64.strftime(t, path)

        url_parts = urlparse(file_name)
        if url_parts.scheme in ('ftp', 'http', 'https'):
            file_name = download_url(file_name)
            if file_name is None:
                continue
            temp_file_name = file_name
        elif url_parts.scheme == 'file':
            file_name = url_parts.path
            
        if not os.path.exists(file_name):
            logger.info('missing file %s', file_name)
            continue

        # Now only need to access local files
        if os.path.splitext(url_parts.path)[1] in ('.gz', '.dgz'):
            # Transparently uncompress
            gunzipped_file = None
            try:
                logger.debug('unzipping %s', file_name)
                gunzipped_file = NamedTemporaryFile(prefix=__name__, 
                                                    delete=False)
                with gzip.open(file_name, 'rb') as gzip_file:
                    shutil.copyfileobj(gzip_file, gunzipped_file)
                gunzipped_file.close()
            except Exception as e:
                if gunzipped_file:
                    os.unlink(gunzipped_file.name)
                continue    
            finally:
                if temp_file_name:
                    logger.debug('deleting temporary file ' + temp_file_name)
                    os.unlink(temp_file_name)

            temp_file_name = gunzipped_file.name
            file_name = temp_file_name
            
        logger.info('loading ' + file_name)

        try:
            tmp = ad['load_converter'](file_name, 
                                       ad,
                                       project=project,
                                       site=site, 
                                       data_type=data_type, 
                                       start_time=t, 
                                       end_time=t2, 
                                       archive=archive,
                                       channels=channels,
                                       path=path,
                                       raise_all=raise_all)
            if tmp is not None:
                if cadence is not None and cadence <= ad['duration']:
                    tmp.set_cadence(cadence, 
                                    aggregate=aggregate,
                                    inplace=True)
                data.append(tmp)
        except Exception as e:
            if raise_all:
                raise
            logger.info('Could not load ' + file_name)
            logger.debug(str(e))
            logger.debug(traceback.format_exc())

        finally:
            if temp_file_name:
                logger.debug('deleting temporary file ' + temp_file_name)
                os.unlink(temp_file_name)

    if len(data) == 0:
        return None

    r = concatenate(data, sort=False)
    r.extract(inplace=True, 
              start_time=start_time, 
              end_time=end_time, 
              channels=channels)

    if cadence is not None and cadence > ad['duration']:
        # cadence too large to apply on results of loading each file, 
        # apply to combined object
        r.set_cadence(cadence, 
                      aggregate=aggregate,
                      inplace=True)

    return r


def concatenate(objs, sort=False):
    obj_type = type(objs[0])
    project = objs[0].project
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
        assert(a.project == project)
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
        integration_interval = \
            np.concatenate(dt64.match_units(integration_interval), axis=1)
    
    sample_start_time = np.concatenate(dt64.match_units(sam_st_list))
    sample_end_time = np.concatenate(dt64.match_units(sam_et_list))
    return obj_type(project=project,
                    site=site,
                    channels=channels,
                    start_time=np.min(start_time),
                    end_time=np.max(end_time),
                    sample_start_time=sample_start_time,
                    sample_end_time=sample_end_time,
                    integration_interval=integration_interval,
                    nominal_cadence=np.max(cadence_list),
                    data=np.concatenate(data_list, axis=1),
                    units=units,
                    sort=sort)


def parse_project_site_list(n_s_list):
    '''Parse an array of strings to unique project/site lists'''
    project_list = []
    site_list = []
    sites_found = {}
    for n_s in n_s_list:
        m = re.match('^([a-z0-9_]+)(/([a-z0-9_]+))?$', n_s, re.IGNORECASE)
        assert m is not None, \
            'Not in form PROJECT or PROJECT/SITE'
        n = m.groups()[0].upper()
        if n not in projects:
            try:
                logger.info('trying to import auroraplot.datasets.' 
                            + n.lower())
                importlib.import_module('auroraplot.datasets.' + n.lower())
            finally:
                if n not in projects:
                    raise Exception('Project %s is not known' % n)

        if m.groups()[2] is None:
            # Given just 'PROJECT'
            sites = projects[n].keys()
        else:
            sites = [m.groups()[2].upper()]
        
        if n not in sites_found:
            sites_found[n] = {}
        for s in sites:
            if s not in sites_found[n]:
                # Not seen this project/site before
                assert s in projects[n], 'Site %s/%s is not known' % (n, s)
                sites_found[n][s] = True
                project_list.append(n)
                site_list.append(s)

    return project_list, site_list


def parse_archive_selection(selection, defaults={}):
    r = copy.deepcopy(defaults)
    for proj_site, arch in selection:
        p_list, s_list = parse_project_site_list([proj_site])
        for n in range(len(p_list)):
            if p_list[n] not in r:
                r[p_list[n]] = { }
            r[p_list[n]][s_list[n]] = arch
    return r


def download_url(url, prefix=__name__, temporary_file=True):
    logger.info('downloading ' + url)
    # For selected schemes attempt to insert authentication
    # data from .netrc
    url_parts = urlparse(url)
    if url_parts.scheme in ('ftp', 'http', 'https') \
            and url_parts.netloc.find('@') == -1:
        # No authentication so attempt to insert details from netrc
        auth = None
        try:
            n = netrc.netrc()
            auth = n.authenticators(url_parts.hostname)
        except IOError as e:
            pass

        if auth:
            logger.debug('inserting authentication details into URL')
            netloc = auth[0] + ':' + auth[2] + '@' + url_parts.hostname

            if url_parts.port:
                netloc += ':' + url_parts.port
            url_parts2 = [url_parts[0], netloc]
            url_parts2.extend(url_parts[2:])
            url = urlunparse(url_parts2)
            # Update parsed values
            # url_parts = urlparse(url)

    url_file = None
    local_file = None
    try:
        url_file = urlopen(url)
        if temporary_file:
            local_file = NamedTemporaryFile(prefix=prefix, 
                                            delete=False)
            logger.debug('saving to ' + local_file.name)
            shutil.copyfileobj(url_file, local_file)
            local_file.close()
            return local_file.name

    except:
        logger.debug(traceback.format_exc())
        if local_file:
            os.unlink(local_file.name)
            raise
    finally:
        if url_file:
            url_file.close()
            
    return None


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

# Warn if timezone not GMT/UTC. Test by comparing two identical times,
# one with a timezone and one without. Repeat the test for a date six
# months later since DST is in operation in July for N hemisphere and
# January for S hemisphere.
if (np.datetime64('2000-01-01T00:00:00') != 
    np.datetime64('2000-01-01T00:00:00Z') or 
    np.datetime64('2000-07-01T00:00:00') != 
    np.datetime64('2000-07-01T00:00:00Z')):
    # If this warning annoys you then set the timezone or use
    # warnings.filterwarnings() to ignore it.
    message = 'Timezone is not UTC or GMT. Times defined without ' + \
        'timezone information will use local timezone'
    warnings.warn(message)
