#!/usr/bin/env python

# Plot magnetometer data from one or more sites.

import argparse
import logging
import os 
import re
import sys
import time

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import auroraplot as ap
import auroraplot.dt64tools as dt64
import auroraplot.auroralactivity
import auroraplot.magdata
import auroraplot.datasets.aurorawatchnet
import auroraplot.datasets.samnet
import auroraplot.datasets.uit
import auroraplot.datasets.dtu


# For each network set the archive from which data is loaded. DTU and
# UIT are swich to HZ when AURORAWATCHNET and SAMNET are included.
archives = {
    'AURORAWATCHNET': 'realtime',
#    'SAMNET': '5s',
    'DTU': 'xyz_10s',
    'UIT': 'xyz_10s',
    }

# Define command line arguments
parser = \
    argparse.ArgumentParser(description='Plot magnetometer data')
parser.add_argument('-s', '--start-time', 
                    default='today',
                    help='Start time for data transfer (inclusive)',
                    metavar='DATETIME')
parser.add_argument('-e', '--end-time',
                    help='End time for data transfer (exclusive)',
                    metavar='DATETIME')
parser.add_argument('--rolling',
                    action='store_true',
                    default=False,
                    help='Make rolling 24 hour plot')
# parser.add_argument('-c', '--channel',
#                     default='H',
#                     help='Magnetometer data channel (axis)')
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
                    default=100,
                    type=float,
                    help='Offset between sites for stack plot (nT)')
parser.add_argument('--qdc-tries', 
                    default=2,
                    type=int,
                    help='Number of tries to load QDC',
                    metavar='NUM')
parser.add_argument('network_site',
                    nargs='+',
                    metavar="NETWORK/SITE")

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


# Parse and process start and end times. If end time not given use
# start time plus 1 day.
if args.rolling:
    et = dt64.ceil(np.datetime64('now', 'us'), np.timedelta64(1, 'h'))
    st = et - np.timedelta64(1, 'D')
else:
    st = np.datetime64(args.start_time) + np.timedelta64(0, 'us')
    if args.end_time is None:
        et = st + np.timedelta64(86400, 's')
    else:
        try:
            # Parse as date
            et = np.datetime64(args.end_time) + np.timedelta64(0, 'us')
        except ValueError as e:
            try:
                # Parse as a set of duration values
                et = st + np.timedelta64(0, 'us')
                et_words = args.end_time.split()
                assert len(et_words) % 2 == 0, 'Need even number of words'
                for n in range(0, len(et_words), 2):
                    et += np.timedelta64(float(et_words[n]), et_words[n+1])
            except:
                raise
        except:
            raise

# Parse and process list of networks and sites.
n_s_list = []
network_list = []
for n_s in args.network_site:
    m = re.match('^([a-z0-9]+)(/([a-z0-9]+))?$', n_s, re.IGNORECASE)
    assert m is not None, \
        'Magnetometer must have form NETWORK or NETWORK/SITE'
    network = m.groups()[0].upper()
    assert network in ap.networks, \
        'Network %s is not known' % network
    network_list.append(network)

    if m.groups()[2] is None:
        # Given just 'NETWORK'
        n_s_list.extend([network + '/' + x for x in list(ap.networks[network].keys())])
    else:
        site = m.groups()[2].upper()
        assert site in ap.networks[network], \
            'Site %s/%s is not known' % (network, site)
        n_s_list.append(network + '/' + site)

if 'AURORAWATCHNET' in network_list or 'SAMNET' in network_list:
    # AURORAWATCHNET and SAMNET are aligned with H so switch default
    # archive for DTU and UIT
    archives['DTU'] = 'hz_10s'
    archives['UIT'] = 'hz_10s'

# Requesting the UIT or DTU data from the UIT web site requires a
# password to be set. Try reading the password from a file called
# .uit_password from the user's home directory.
if ('UIT' in network_list or 'DTU' in network_list) \
        and ap.datasets.uit.uit_password is None:
    raise Exception('UIT password needed but could not be set')


# Load the data for each site. 
mdl = []
for n_s in n_s_list:
    network, site = n_s.split('/')
    kwargs = {}
    if network in archives:
        kwargs['archive'] = archives[network]
    md = ap.load_data(network, site, 'MagData', st, et, **kwargs)
                      # archive=archives[network])
    # Is result is None then no data available, so ignore those
    # results.
    if md is not None:
        md = md.mark_missing_data(cadence=2*md.nominal_cadence)
        mdl.append(md)


if len(mdl) == 0:
    print('No data to plot')
    sys.exit(0)

if args.plot_type is None or args.plot_type == 'stack_plot':
    if len(mdl) == 1:
        # No point in using a stackplot for a single site
        mdl[0].plot()
    else:
        # Create a stackplot.
        ap.magdata.stack_plot(mdl, offset=args.offset * 1e-9)
else:
    # Every other plot type makes one figure per site
    for md in mdl:
        if args.plot_type == 'k_index_plot':
            try:
                qdc = ap.magdata.load_qdc(md.network, 
                                          md.site,
                                          dt64.mean(md.start_time, 
                                                    md.end_time),
                                          channels=md.channels,
                                          tries=args.qdc_tries)
            except Exception:
                qdc = None
            k = ap.auroralactivity.KIndex(magdata=md, magqdc=qdc)
            k.plot()


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

        # Abbreviate AURORAWATCHNET to AWN
        ap.datasets.aurorawatchnet.abbreviate_aurorawatchnet(ax, title=False)

# Make figure(s) visible.
plt.show()

