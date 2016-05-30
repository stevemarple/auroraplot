#!/usr/bin/env python

import argparse
import copy
from importlib import import_module
import logging
import os 
import re
import sys
import time
import traceback

import numpy as np

import matplotlib as mpl

from numpy.f2py.auxfuncs import throw_error
from logging import exception
if os.environ.get('DISPLAY', '') == '':
    mpl.use('Agg')
        

import matplotlib.ticker
import matplotlib.pyplot as plt

try:
    # Try to force all times to be read as UTC
    os.environ['TZ'] = 'UTC'
    time.tzset()
except:
    pass

if sys.version_info[0] >= 3:
    import configparser
    from configparser import RawConfigParser
else:
    import ConfigParser
    from ConfigParser import RawConfigParser


import auroraplot as ap
import auroraplot.dt64tools as dt64
import auroraplot.auroralactivity
import auroraplot.magdata
import auroraplot.datasets.aurorawatchnet
import auroraplot.datasets.samnet
import auroraplot.datasets.uit
import auroraplot.datasets.dtu
import auroraplot.datasets.intermagnet
import auroraplot.datasets.bgs_schools


def read_config_files():
    r = {}
    config_dir = os.path.join(\
        os.path.expanduser('~'),
        '.' + os.path.splitext(os.path.basename(__file__))[0])
    if args.test_mode:
        config_dir = os.path.join(config_dir, 'test')


    filename = os.path.join(config_dir, 'master.ini')
    if not os.path.exists(filename):
        raise Exception('missing config file %s' % filename)
    logger.debug('reading config file %s', filename)
    r['master'] = RawConfigParser()
    r['master'].add_section('stackplot')
    r['master'].set('stackplot', 'offset', '100')

    # Relative to summary plots base directory
    r['master'].set('stackplot', 'path', 
                    os.path.join('stackplots',
                                 '{data:%Y}',
                                 '{data:%m}',
                                 '{data:%Y%m%d}'))
    r['master'].set('stackplot', 'image_types', 'png')

    r['master'].set('stackplot', 'rolling_path', 
                    os.path.join('stackplots',
                                 'rolling'))
    r['master'].set('stackplot', 'rolling_image_types', 'png svg')

    r['master'].add_section('MagData')
    r['master'].set('MagData', 'path', 
                    os.path.join('{data:project_lc}',
                                 '{data:site_lc}',
                                 '{data:%Y}',
                                 '{data:%m}',
                                 '{data:site_lc}_{data:%Y%m%d}'))
    r['master'].set('MagData', 'image_types', 'png')

    r['master'].set('MagData', 'rolling_path', 
                    os.path.join('{data:project_lc}',
                                 '{data:site_lc}',
                                 'rolling'))
    r['master'].set('MagData', 'rolling_image_types', 'png svg')


    r['master'].add_section('AuroraWatchActivity')
    r['master'].set('AuroraWatchActivity', 'path', 
                    os.path.join('{data:project_lc}',
                                 '{data:site_lc}',
                                 '{data:%Y}',
                                 '{data:%m}',
                                 '{data:site_lc}_{data:%Y%m%d}_act'))
    r['master'].set('AuroraWatchActivity', 'image_types', 'png')
    r['master'].set('AuroraWatchActivity', 'rolling_path', 
                    os.path.join('{data:project_lc}',
                                 '{data:site_lc}',
                                 'rolling_act'))
    r['master'].set('AuroraWatchActivity', 'rolling_image_types', 'png svg')

    r['master'].add_section('KIndex')
    r['master'].set('KIndex', 'path', 
                    os.path.join('{data:project_lc}',
                                 '{data:site_lc}',
                                 '{data:%Y}',
                                 '{data:%m}',
                                 '{data:site_lc}_{data:%Y%m%d}_k'))
    r['master'].set('KIndex', 'image_types', 'png')
    r['master'].set('KIndex', 'rolling_path', 
                    os.path.join('{data:project_lc}',
                                 '{data:site_lc}',
                                 'rolling_k'))
    r['master'].set('KIndex', 'rolling_image_types', 'png svg')


    r['master'].add_section('TemperatureData')
    r['master'].set('TemperatureData', 'path', 
                    os.path.join('{data:project_lc}',
                                 '{data:site_lc}',
                                 '{data:%Y}',
                                 '{data:%m}',
                                 '{data:site_lc}_{data:%Y%m%d}_temp'))
    r['master'].set('TemperatureData', 'image_types', 'png')
    r['master'].set('TemperatureData', 'rolling_path', 
                    os.path.join('{data:project_lc}',
                                 '{data:site_lc}',
                                 'rolling_temp'))
    r['master'].set('TemperatureData', 'rolling_image_types', 'png svg')

    r['master'].add_section('VoltageData')
    r['master'].set('VoltageData', 'path', 
                    os.path.join('{data:project_lc}',
                                 '{data:site_lc}',
                                 '{data:%Y}',
                                 '{data:%m}',
                                 '{data:site_lc}_{data:%Y%m%d}_voltage'))
    r['master'].set('VoltageData', 'image_types', 'png')
    r['master'].set('VoltageData', 'rolling_path', 
                    os.path.join('{data:project_lc}',
                                 '{data:site_lc}',
                                 'rolling_voltage'))
    r['master'].set('VoltageData', 'rolling_image_types', 'png svg')


    r['master'].read(filename)



    ps_list = []
    ps_list.extend(get_sites(r['master'], 'stackplot'))
    ps_list.extend(get_sites(r['master'], 'status_plots'))

    project_configs = {}
    for p, s in ps_list:
        
        if p not in r:
            r[p] = {}

        if s not in r[p]:
            site_filename = os.path.join(config_dir, 
                                         '%s_%s.ini' % (p.lower(), s.lower()))
            if os.path.exists(site_filename):
                # Site-specific config file is used for this site
                logger.debug('reading site config file %s', site_filename)
                r[p][s] = RawConfigParser()
                r[p][s].read(site_filename)
            elif p in project_configs:
                # Previously loaded project-specific config file is
                # used for this site
                r[p][s] = project_configs[p]

            else:
                proj_filename = os.path.join(config_dir, p.lower() + '.ini')
                if os.path.exists(proj_filename):
                    # Load and cache project-specific config file
                    logger.debug('reading project config file %s', 
                                 proj_filename)
                    project_configs[p] = RawConfigParser()
                    project_configs[p].read(proj_filename)
                    r[p][s] = project_configs[p]
                else:
                    r[p][s] = None
                    logger.debug(('no project or site config file ' +
                                  'found for %s/%s'), p, s)


    return r


def get_option(config, section, option, default=None, get=None):
    if config is not None and config.has_option(section, option):
        if get is None:
            return config.get(section, option)
        else:
            # For getboolean etc
            return get(config, section, option)
    else:
        return default

def get_sites(config, section):
    def _get_sites(config, section, option):
        if not config.has_option(section, option):
            return [], []
        return ap.parse_project_site_list(config.get(section, option).split())

    
    d = {}
    for p, s in zip(*_get_sites(config, section, 'sites')):
        if p not in d:
            d[p] = {}
        d[p][s] = None
    
    for p, s in zip(*_get_sites(config, section, 'excluded_sites')):
        if p in d and s in d[p]:
            del d[p][s]
            
    r = []
    for p in d:
        for s in d[p]:
            r.append((p,s))
    return r


def get_start_end_times():
    # Parse and process start and end times. If end time not given use
    # start time plus 1 day.
    if args.start_time is None and args.end_time is None:
        # Rolling 24 hours
        et = dt64.ceil(now, activity_cadence)
        st = et - day
        yield st, et, True
        
    else:
        if args.start_time is not None:
            st = dt64.parse_datetime64(args.start_time, 's', now=now)

        if args.end_time is None:
            et = st + day
        else:
            try:
                # Parse as date
                et = dt64.parse_datetime64(args.end_time, 's', now=now)
            except ValueError as e:
                try:
                    # Parse as a set of duration values
                    et = st + dt64.parse_datetime64(args.end_time, 's', 
                                                    now=now)
                except:
                    raise
            except:
                raise
        st = dt64.floor(st, day)
        et = dt64.ceil(et, day)
        t = st + 0*day
        while t < et:
            yield t, t+day, False
            t += day


def my_load_data(project, site,  data_type, st, et):
    kwargs = {}
    kwargs['archive'] = get_option(conf[project][site], data_type, 'archive',
                                   None)
    if project in archive and site in archive[project]:
        kwargs['archive'] = archive[project][site]
        
    try:
        md = ap.load_data(project, site, data_type, st, et, **kwargs)
        if md is None or md.data.size == 0:
            return None

        md = md.mark_missing_data(cadence=2*md.nominal_cadence)
        return md

    except Exception as e:
        logger.error(str(e))
        logger.debug(traceback.format_exc())
        return None


def get_aligned_qdc(md):
    '''Get QDC, aligned to input data

    When data crosses midnight this function selects the appropriate
    QDC for each day.
    '''
    assert md.end_time - md.start_time <= day, \
        'Data can cross >1 midnight'

    if md.end_time != dt64.get_date(md.end_time):
        # data crosses midnight. Get separate QDCs for each day
        md_list = [md.extract(end_time=dt64.floor(md.end_time, day)),
                   md.extract(start_time=dt64.floor(md.end_time, day))]
    else:
        md_list = [md]

    qdc_list = []
    for d in md_list:
        try:
            qdc = ap.magdata.load_qdc(d.project, d.site, d.start_time,
                                      tries=3,
                                      realtime=True)
            if qdc is not None and np.size(qdc.data):
                aligned_qdc = qdc.align(d, fit='realtime_baseline')
                if aligned_qdc is not None and np.size(aligned_qdc.data):
                    qdc_list.append(aligned_qdc)

        except Exception as e:
            logger.debug(traceback.format_exc())
            logger.error(str(e))

    if len(qdc_list) == 0:
        return None
    elif len(qdc_list) == 1:
        return qdc_list[0]
    else:
        return ap.concatenate(qdc_list, sort=False)


def make_aw_act_data(md, save=False):
    if md is None:
        return None
    act_list = []
    t1 = md.start_time
    # for t1 in dt64.dt64_range(md.start_time, md.end_time, day):
    while t1 < md.end_time:
        t2 = dt64.floor(t1 + day, day)
        try:
            qdc = ap.magdata.load_qdc(md.project, md.site, t1,
                                      tries=3,
                                      realtime=True)
            if qdc is not None and np.size(qdc.data):
                fitted_qdc = qdc.align(md, fit='realtime_baseline')
                act = ap.auroralactivity.AuroraWatchActivity(magdata=md,
                                                             magqdc=fitted_qdc,
                                                             range_=True)

                if act is not None and np.size(act.data):
                    act_list.append(act)

        except Exception as e:
            logger.debug(traceback.format_exc())
            logger.error(str(e))
        
        t1 = t2

    if len(act_list) == 0:
        return None
    r = ap.concatenate(act_list)
    if save:
        r.save(merge=True)
    return r


def make_filename(data, rolling, stackplot=False):
    if rolling:
        option = 'rolling_path'
    else:
        option = 'path'
    if stackplot:
        fstr = get_option(conf['master'], 'stackplot', option)
    else:
        fstr = get_option(conf[data.project][data.site],
                          type(data).__name__, option)
        if fstr is None:
            fstr = get_option(conf['master'], type(data).__name__, option)
    
    r = fstr.format(data=data)
    if not os.path.isabs(r):
        r = os.path.join(summary_base_dir, r)
    return r

def get_image_types(data_type, rolling):
    if roll:
        return conf['master'].get(data_type, 'rolling_image_types').split()
    else:
        return conf['master'].get(data_type, 'image_types').split()
        

def my_save_fig(fig, filename, ext_list, abbreviate=False, close=None):
    # Apply some standard format changes
    axes = fig.get_axes()
    if len(axes) > 1:
        num_ticks = int(20. / len(axes))
        for ax in axes:
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(num_ticks))

    for ax in axes:
        ax.grid(True)

        if abbreviate:
            # Shorten AURORAWATCHNET to AWN
            tll = ax.yaxis.get_ticklabels() # tick label list
            labels = [ tl.get_text() for tl in tll]
            labels = map(lambda x: x.replace('AURORAWATCHNET', 'AWN'), labels)
            ax.yaxis.set_ticklabels(labels)

        # Label axis by hour, set tick marks on 3h intervals
        if dt64.astype(np.diff(ax.get_xlim()), 
                       units=dt64.get_plot_units(ax.xaxis),
                       time_type='timedelta64') \
                == np.timedelta64(24, 'h'):
            ax.xaxis.set_major_formatter(dt64.Datetime64Formatter(fmt='%H'))
            ax.xaxis.set_major_locator(\
                dt64.Datetime64Locator(interval=np.timedelta64(3, 'h'),
                                       maxticks=10))
            
    # Create directory and save to the designated list of extensions
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        logger.debug('making directory %s', dirname)
        os.makedirs(dirname)
    for ext in ext_list:
        f = filename + '.' + ext
        logger.info('saving to %s', f)
        fig.savefig(f, dpi=80)

    if close or (close is None and not args.show):
        plt.close(fig)
        
    return
        

# Define command line arguments
parser = \
    argparse.ArgumentParser(description='Plot magnetometer data',
                            # formatter_class=argparse.RawTextHelpFormatter,
                            formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument('-a', '--archive', 
                    action='append',
                    nargs=2,
                    help='Select data archive used for project or site',
                    metavar=('PROJECT[/SITE]', 'ARCHIVE'))
parser.add_argument('--dataset',
                    nargs='*',
                    help='Import additional datset',
                    metavar='MODULE')
parser.add_argument('-e', '--end-time',
                    help='End time for data transfer (exclusive)',
                    metavar='DATETIME')
parser.add_argument('--log-format',
                    default='%(levelname)s:%(message)s',
                    help='Set format of log messages',
                    metavar='FORMAT')
parser.add_argument('--log-level', 
                    choices=['debug', 'info', 'warning', 'error', 'critical'],
                    default='warning',
                    help='Control how much detail is printed',
                    metavar='LEVEL')
parser.add_argument('--now',
                    help='Set current time for test mode',
                    metavar='DATETIME')
parser.add_argument('--rolling',
                    action='store_true',
                    default=False,
                    help='Make rolling 24 hour plot')
parser.add_argument('-s', '--start-time', 
                    help='Start time for data transfer (inclusive)',
                    metavar='DATETIME')
parser.add_argument('--show', 
                    action='store_true',
                    help='Show plots for final day')
parser.add_argument('--test-mode',
                    action='store_true',
                    help='Test mode for plots and jobs')




args = parser.parse_args()
if __name__ == '__main__':
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format=args.log_format)

logger = logging.getLogger(__name__)

# Set default values for figure
mpl.rcParams['figure.facecolor'] = 'w'
mpl.rcParams['figure.figsize'] = [8, 6]
mpl.rcParams['figure.subplot.bottom'] = 0.1
mpl.rcParams['figure.subplot.left'] = 0.12
mpl.rcParams['figure.subplot.right'] = 0.925
mpl.rcParams['figure.subplot.top'] = 0.85
mpl.rcParams['axes.grid'] = True
mpl.rcParams['legend.fontsize'] = 'medium'
mpl.rcParams['legend.numpoints'] = 1


# Set timezone appropriately to get intended np.datetime64 behaviour.
try:
    os.environ['TZ'] = 'UTC'
    time.tzset()
except Exception as e:
    # Cannot use tzset on windows
    logger.warn('Could not set time zone to UTC')


if args.dataset:
    for ds in args.dataset:
        new_module = 'auroraplot.datasets.' + ds
        try:
            import_module(new_module)
        except Exception as e:
            logger.critical('Could not import ' + new_module + ': ' + str(e))
            sys.exit(1)

# Process --archive options
archive = {}
if args.archive:
    archive = ap.parse_archive_selection(args.archive)


conf = read_config_files()
summary_base_dir = os.path.normpath(conf['master'].get('summary_plots', 
                                                       'base_dir'))
if args.test_mode:
    summary_base_dir = os.path.join(summary_base_dir, 'test')


logger.debug('summary plots directory: %s', summary_base_dir)
        
# Use a consistent value for current time, process any --now option
# first.
if args.now:
    now = dt64.parse_datetime64(args.now, 's')
else:
    now = np.datetime64('now', 's')
day = np.timedelta64(1, 'D')
activity_cadence = np.timedelta64(1, 'h')

default_data_types = conf['master'].get('status_plots', 'data_types')
stackplot_offset = conf['master'].getfloat('stackplot', 'offset') * 1e-9


stackplot_sites = {}
for project, site in get_sites(conf['master'], 'stackplot'):
    if project not in stackplot_sites:
        stackplot_sites[project] = {}
    if site not in stackplot_sites[project]:
        stackplot_sites[project][site] = None
        
# Load the data for each period/site. 

for st, et, rolling in get_start_end_times():
    if rolling:
        plot_st_et = [(st, et, True), 
                      (dt64.ceil(st, day), dt64.ceil(et, day), False)]
    else:
        plot_st_et = [(st, et, False)]

    cached_data = {} 

    plt.close('all')

    for project, site in get_sites(conf['master'], 'status_plots'):
        site_data_types = get_option(conf[project][site],
                                     'status_plots', 'data_types',
                                     default=default_data_types).split()
        mag_data = None
        aligned_qdc = None

        # Iterate over data types, but process AuroraWatchActivity
        # after MagData to reuse already loaded data
        for data_type in reversed(sorted(site_data_types)):
            if not get_option(conf[project][site], data_type, 'enabled',
                              default=True, get=RawConfigParser.getboolean):
                logger.debug('skipping %s for %s/%s', 
                             data_type, project, site)
                continue
            
            if data_type == 'AuroraWatchActivity':
                data = make_aw_act_data(mag_data)
            else:
                data = my_load_data(project, site, data_type, st, et)
                
            if data_type == 'MagData':
                mag_data = data
                if mag_data is not None:
                    aligned_qdc = get_aligned_qdc(mag_data)
                # Keep data which is needed for stackplot
                if project in stackplot_sites \
                        and site in stackplot_sites[project]:
                    if project not in cached_data:
                        cached_data[project] = {}
                    cached_data[project][site] = data

                

            if data is not None:
                for plot_st, plot_et, roll in plot_st_et:
                    d = data.extract(plot_st, plot_et)
                    d.plot(zorder=4)

                    if data_type == 'MagData':
                        # Add QDC to plot if possible
                        qdc = None
                        if aligned_qdc is not None:
                            qdc = aligned_qdc.extract(plot_st, plot_et)
                            if np.size(qdc.data):
                                fig = plt.gcf()
                                qdc.plot(figure=fig, color='grey', zorder=1)


                    my_save_fig(plt.gcf(), 
                                make_filename(d, roll),
                                get_image_types(data_type, roll))

            


    # Load data for stackplot (read from cached_data if possible)
    sp_data = []
    for project in stackplot_sites:
        for site in stackplot_sites[project]:
            if project in cached_data \
                    and site in cached_data[project]:
                logger.debug('stackplot data for %s/%s already loaded', 
                             project, site)
                data = cached_data[project][site]
            else:
                data = my_load_data(project, site, 'MagData', st, et)
                
            if data is not None:
                sp_data.append(data)

    # Make stackplot
    if len(sp_data) == 0:
        logger.warning('No stackplot data to process')
    else:
        for plot_st, plot_et, roll in plot_st_et:
            tmp_sp_data = []
            for d in sp_data:
                tmp_sp_data.append(d.extract(plot_st, plot_et))
            ap.magdata.stack_plot(tmp_sp_data, 
                                  offset=stackplot_offset,
                                  start_time=plot_st,
                                  end_time=plot_et)
            my_save_fig(plt.gcf(), 
                        make_filename(tmp_sp_data[0], roll, stackplot=True),
                        get_image_types(data_type, roll),
                        abbreviate=True)




# Make figure(s) visible.
if args.show:    
    plt.show()

