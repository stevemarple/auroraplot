#!/usr/bin/env python

# Plot magnetometer data from one or more sites.

import argparse
import copy
from importlib import import_module
import logging
import os 
import re
import sys
import time

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    # Try to force all times to be read as UTC
    os.environ['TZ'] = 'UTC'
    time.tzset()
except:
    pass

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



# For each project set the archive from which data is loaded. DTU and
# UIT are switched to hz_10s when AWN and SAMNET are
# included.
default_archive_selection = [['AWN', 'realtime'], 
                             ['DTU', 'xyz_10s'],
                             ['UIT', 'xyz_10s'],
                             ['INTERMAGNET', 'preliminary'],
                             ]

# Define command line arguments
parser = \
    argparse.ArgumentParser(description='Plot magnetometer data',
                            # formatter_class=argparse.RawTextHelpFormatter,
                            formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument('--aggregate', 
                    default='scipy.average',
                    help='Aggregate function used for setting cadence',
                    metavar='MODULE.NAME')
parser.add_argument('-a', '--archive', 
                    action='append',
                    nargs=2,
                    help='Select data archive used for project or site',
                    metavar=('PROJECT[/SITE]', 'ARCHIVE'))
parser.add_argument('--aurorawatch-activity', 
                    dest='plot_type',
                    action='store_const',
                    const='aurorawatch_activity',
                    help='Make AuroraWatch activity plot(s)')
parser.add_argument('--cadence', 
                    help='Set cadence (used when loading data)')
parser.add_argument('-s', '--start-time', 
                    default='today',
                    help='Start time for data transfer (inclusive)',
                    metavar='DATETIME')
parser.add_argument('-e', '--end-time',
                    help='End time for data transfer (exclusive)',
                    metavar='DATETIME')
parser.add_argument('--post-aggregate', 
                    default='scipy.average',
                    help='Aggregate function used for setting post-load cadence',
                    metavar='MODULE.NAME')
parser.add_argument('--post-cadence', 
                    help='Set cadence (after loading data)')
parser.add_argument('--rolling',
                    action='store_true',
                    default=False,
                    help='Make rolling 24 hour plot')
parser.add_argument('-c', '--channels',
                    default='H X',
                    help='Stack plot data channel(s)')
parser.add_argument('--dataset',
                    nargs='*',
                    help='Import additional datset',
                    metavar='MODULE')
parser.add_argument('--list-sites',
                    action='store_true',
                    help='List available sites and exit')
parser.add_argument('--highlight',
                    nargs=2,
                    action='append',
                    help='Highlight interval on plot',
                    metavar=('START_DATETIME', 'END_DATETIME'))
parser.add_argument('--highlight-color',
                    action='append',
                    help='Color used for plot highlights')
parser.add_argument('--log-level', 
                    choices=['debug', 'info', 'warning', 'error', 'critical'],
                    default='warning',
                    help='Control how much detail is printed',
                    metavar='LEVEL')
parser.add_argument('--log-format',
                    default='%(levelname)s:%(message)s',
                    help='Set format of log messages',
                    metavar='FORMAT')
parser.add_argument('--offset', 
                    default='auto',
                    type=lambda x: np.nan if x.lower() == 'auto' else float(x),
                    help='Offset between sites for stack plot (nT)')
parser.add_argument('--plot-function',
                    help='Name of matplotlib plot function',
                    metavar='FUNCTION_NAME')
parser.add_argument('--qdc-tries', 
                    default=2,
                    type=int,
                    help='Number of tries to load QDC',
                    metavar='NUM')
parser.add_argument('--k-index-cadence', 
                    default='3h',
                    help='Cadence for K index plot',
                    metavar='CADENCE')
parser.add_argument('--legend',
                    action='store_true',
                    help='Add legend to plot')
parser.add_argument('--save-filename',
                    help='Save plot',
                    metavar='FILE')
parser.add_argument('--dpi',
                    type=float,
                    default=80,
                    help='DPI when saving plot')

plot_type = parser.add_mutually_exclusive_group(required=False)
plot_type.add_argument('--stack-plot', 
                       dest='plot_type',
                       action='store_const',
                       const='stack_plot',
                       help='Display as a stack plot')
plot_type.add_argument('--k-index-plot', 
                       dest='plot_type',
                       action='store_const',
                       const='k_index_plot',
                       help='Make K index plot(s)')
plot_type.add_argument('--temperature-plot', 
                       dest='plot_type',
                       action='store_const',
                       const='temp_plot',
                       help='Make temperature plot(s)')
plot_type.add_argument('--voltage-plot', 
                       dest='plot_type',
                       action='store_const',
                       const='voltage_plot',
                       help='Make voltage plot(s)')


parser.add_argument('project_site',
                    nargs='*',
                    metavar="PROJECT/SITE")



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


if args.list_sites:
    ps_list = ['Projects/sites available:']
    for p in sorted(ap.projects):
        ps_list.append('  ' + p + ':')
        first = True
        slist = []
        for site in sorted(ap.projects[p]):
            if first:
                s = '      ' + site
                first = False
            else:
                s += ', ' + site
            if len(s) > 60:
                slist.append(s)
                s = ''
                first = True
        slist.append(s)
        ps_list.append(',\n'.join(slist))
    print('\n'.join(ps_list))
    sys.exit(0)

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
            print('Could not import ' + new_module + ': ' + str(e))
            sys.exit(1)
        
# Parse and process start and end times. If end time not given use
# start time plus 1 day.
if args.rolling:
    et = dt64.ceil(np.datetime64('now', 's'), np.timedelta64(1, 'h'))
    st = et - np.timedelta64(1, 'D')
else:
    st = dt64.parse_datetime64(args.start_time, 's')
    if args.end_time is None:
        et = st + np.timedelta64(86400, 's')
    else:
        try:
            # Parse as date
            et = dt64.parse_datetime64(args.end_time, 's')
        except ValueError as e:
            try:
                # Parse as a set of duration values
                et = st + np.timedelta64(0, 's')
                et_words = args.end_time.split()
                assert len(et_words) % 2 == 0, 'Need even number of words'
                for n in range(0, len(et_words), 2):
                    et += np.timedelta64(float(et_words[n]), et_words[n+1])
            except:
                raise
        except:
            raise

project_list, site_list = ap.parse_project_site_list(args.project_site)

if len(site_list) == 0:
    sys.stderr.write('No sites specified\n')
    sys.exit(1)

if 'AWN' in project_list or 'SAMNET' in project_list:
    # AWN and SAMNET are aligned with H so switch default
    # archive for DTU and UIT
    default_archive_selection.append(['DTU', 'hz_10s'])
    default_archive_selection.append(['UIT', 'hz_10s'])


# Requesting the UIT or DTU data from the UIT web site requires a
# password to be set. Try reading the password from a file called
# .uit_password from the user's home directory.
if ('UIT' in project_list or 'DTU' in project_list) \
        and ap.datasets.uit.uit_password is None:
    logger.warn('UIT password likely to be required but is not set')


# Get the default archives
archive = ap.parse_archive_selection(default_archive_selection)

# Process --archive options
if args.archive:
    archive = ap.parse_archive_selection(args.archive, defaults=archive)

if args.plot_type == 'temp_plot':
    data_type = 'TemperatureData'
elif args.plot_type == 'voltage_plot':
    data_type = 'VoltageData'
elif args.plot_type == 'aurorawatch_activity':
    data_type = 'AuroraWatchActivity'
else:
    data_type = 'MagData'


highlight_t = []
if args.highlight:
    for h_st, h_et in args.highlight:
        highlight_t.append([dt64.parse_datetime64(h_st, 's'),
                            dt64.parse_datetime64(h_et, 's')])
    if not args.highlight_color:
        args.highlight_color = ['#ffffaa']

if args.cadence:
    cadence = dt64.parse_timedelta64(args.cadence, 's')
    agg_mname, agg_fname = ap.tools.lookup_module_name(args.aggregate)
    agg_module = import_module(agg_mname)
    agg_func = getattr(agg_module, agg_fname)
else:
    cadence = None

if args.post_cadence:
    post_cadence = dt64.parse_timedelta64(args.post_cadence, 's')
    pa_mname, pa_fname = ap.tools.lookup_module_name(args.post_aggregate)
    pa_module = import_module(pa_mname)
    post_agg_func = getattr(pa_module, pa_fname)

else:
    post_cadence = None

# Load the data for each site. 
mdl = []
for n in range(len(project_list)):
    project = project_list[n]
    site = site_list[n]
    kwargs = {}
    if project in archive and site in archive[project]:
        kwargs['archive'] = archive[project][site]
    if cadence:
        kwargs['cadence'] = cadence
        kwargs['aggregate'] = agg_func

    if ap.is_operational(project, site, st, et):
        md = ap.load_data(project, site, data_type, st, et, **kwargs)
    else:
        logger.info('%s/%s not operational at this time', project, site)
        md = None
    # If result is None then no data available so ignore those
    # results.
    if (md is not None and md.data.size 
        and np.any(np.isfinite(md.data))):
        md = md.mark_missing_data(cadence=3*md.nominal_cadence)
        if post_cadence:
            md.set_cadence(post_cadence, aggregate=post_agg_func, inplace=True)
        mdl.append(md)

if args.plot_function:
    plot_args = dict(plot_func=getattr(plt, args.plot_function))
else:
    plot_args = {}

if len(mdl) == 0:
    print('No data to plot')
    sys.exit(0)

if args.plot_type is None or args.plot_type == 'stack_plot':
    if len(mdl) == 1:
        # No point in using a stackplot for a single site
        mdl[0].plot(**plot_args)
    else:
        # Create a stackplot.
        ap.magdata.stack_plot(mdl, 
                              offset=args.offset * 1e-9,
                              channel=args.channels.split(),
                              add_legend=args.legend)
else:
    # Every other plot type makes one figure per site
    for md in mdl:
        if args.plot_type == 'k_index_plot':
            try:
                qdc = ap.magdata.load_qdc(md.project, 
                                          md.site,
                                          dt64.mean(md.start_time, 
                                                    md.end_time),
                                          channels=md.channels,
                                          tries=args.qdc_tries)
            except Exception:
                qdc = None
            k_cadence = dt64.parse_timedelta64(args.k_index_cadence, 's')
            k = ap.auroralactivity.KIndex(magdata=md, magqdc=qdc,
                                          nominal_cadence=k_cadence)
            k.plot(**plot_args)
        else:
            md.plot(**plot_args)

# Override the labelling format for all figures
for fn in plt.get_fignums():
    fig = plt.figure(fn)
    for ax in fig.axes:
        # Set maxticks so that for an entire day the ticks are at 3-hourly
        # intervals (to correspond with K index plots).
        ax.xaxis.set_major_locator(dt64.Datetime64Locator(maxticks=9))

        # Have axis labelled with date or time, as appropriate. Indicate UT.
        ax.xaxis.set_major_formatter( \
            dt64.Datetime64Formatter(autolabel='%s (UT)'))
        h_color_n = 0
        for h_st, h_et in highlight_t:
            if h_color_n >= len(args.highlight_color):
                h_color_n = 0
            h_color = args.highlight_color[h_color_n]
            dt64.highlight(ax, h_st, h_et, 
                           facecolor=h_color, 
                           edgecolor=h_color, 
                           zorder=-2)
            h_color_n += 1


if args.save_filename:
    fig.savefig(args.save_filename, dpi=args.dpi)

# Make figure(s) visible.
plt.show()

