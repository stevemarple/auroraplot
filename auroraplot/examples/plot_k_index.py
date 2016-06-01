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
import auroraplot.magdata
import auroraplot.auroralactivity
import auroraplot.datasets.aurorawatchnet
import auroraplot.datasets.samnet
import auroraplot.datasets.uit
import auroraplot.datasets.dtu


# For each project set the archive from which data is loaded 
archives = {
    'AURORAWATCHNET': 'realtime',
#    'SAMNET': '5s',
    'DTU': 'hz_10s',
    'UIT': 'hz_10s',
    }

# Define command line arguments
parser = \
    argparse.ArgumentParser(description='Plot local K index')
parser.add_argument('-s', '--start-time', 
                    default='today',
                    help='Start time for data transfer (inclusive)',
                    metavar='DATETIME')
parser.add_argument('-e', '--end-time',
                    help='End time for data transfer (exclusive)',
                    metavar='DATETIME')
parser.add_argument('--tries',
                    type=int,
                    default=3,
                    help='Number of tries to laod a QDC')
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
                    help='Offset between sites in nT')
parser.add_argument('project_site',
                    nargs='+',
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


# Set timezone appropriately to get intended np.datetime64 behaviour.
os.environ['TZ'] = 'UTC'
time.tzset()

# Parse and process start and end times. If end time not given use
# start time plus 1 day.
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

# Parse and process list of projects and sites.
n_s_list = []
project_list = []
for n_s in args.project_site:
    m = re.match('^([a-z0-9]+)(/([a-z0-9]+))?$', n_s, re.IGNORECASE)
    assert m is not None, \
        'Magnetometer must have form PROJECT or PROJECT/SITE'
    project = m.groups()[0].upper()
    assert project in ap.projects, \
        'Project %s is not known' % project
    project_list.append(project)

    if m.groups()[2] is None:
        # Given just 'PROJECT'
        n_s_list.extend([project + '/' + x for x in list(ap.projects[project].keys())])
    else:
        site = m.groups()[2].upper()
        assert site in ap.projects[project], \
            'Site %s/%s is not known' % (project, site)
        n_s_list.append(project + '/' + site)


# Requesting the UIT or DTU data from the UIT web site requires a
# password to be set. Try reading the password from a file called
# .uit_password from the user's home directory.
if ('UIT' in project_list or 'DTU' in project_list) \
        and ap.datasets.uit.uit_password is None:
    raise Exception('UIT password needed but could not be set')


# Load and plot the data for each site. 
for n_s in n_s_list:
    project, site = n_s.split('/')
    kwargs = {}
    if project in archives:
        kwargs['archive'] = archives[project]
    md = ap.load_data(project, site, 'MagData', st, et, **kwargs)
                      # archive=archives[project])
    # Is result is None then no data available, so ignore those
    # results.
    qdc = None
    if (md is not None and
        'MagQDC' in ap.get_site_info(project, site, 'data_types')):
        md = md.mark_missing_data(cadence=2*md.nominal_cadence)
        qdc_info = ap.magdata.load_qdc(project, site, dt64.mean(st, et),
                                       tries=args.tries, full_output=True)
        if qdc_info:
            qdc = qdc_info['magqdc']
        if qdc is not None and len(md.channels) != len(qdc.channels):
            qdc = None
    k = ap.auroralactivity.KIndex(magdata=md, magqdc=qdc)
    k.plot()
    fig = plt.gcf()
       
    # Override the labelling format.
    for ax in fig.axes:
        # Set maxticks so that for an entire day the ticks are at
        # 3-hourly intervals (to correspond with K index plots).
        ax.xaxis.set_major_locator(dt64.Datetime64Locator(maxticks=9))

        # Have axis labelled with date or time, as
        # appropriate. Indicate UT.
        ax.xaxis.set_major_formatter( \
            dt64.Datetime64Formatter(autolabel='%s (UT)'))

        # Abbreviate AURORAWATCHNET to AWN
        ap.datasets.aurorawatchnet.abbreviate_aurorawatchnet(ax, 
                                                             title=False)

        if qdc is None:
            ax.text(np.mean(ax.get_xlim()), np.mean(ax.get_ylim()), 
                    'Calculated\nwithout a QDC',
                    horizontalalignment='center', 
                    verticalalignment='center', 
                    fontsize=50, alpha=0.5, color='grey')
                    
        if qdc is None or qdc_info['tries'] > 1:
            # Have Y label say 'Estimated ...'
            s = ax.get_ylabel()
            if s.startswith('Local'):
                ax.set_ylabel(s.replace('Local', 'Estimated local', 1))

# Make the figure(s) visible.
plt.show()

