#!/usr/bin/env python
import argparse
from importlib import import_module
import logging
import os
import os.path
import sys
import time

import numpy as np

try:
    # Try to force all times to be read as UTC
    os.environ['TZ'] = 'UTC'
    time.tzset()
except:
    pass

import auroraplot as ap
import auroraplot.dt64tools as dt64
import auroraplot.magdata
import auroraplot.datasets.aurorawatchnet
import auroraplot.datasets.samnet

logger = logging.getLogger(__name__)



assert os.environ.get('TZ') == 'UTC', \
    'TZ environment variable must be set to UTC'


# ==========================================================================

# Parse command line options
parser = argparse.ArgumentParser(description\
                                     ='Make AuroraWatch quiet-day curve(s).')
parser.add_argument('-a', '--archive', 
                    action='append',
                    nargs=2,
                    help='Select data archive used for project or site',
                    metavar=('PROJECT[/SITE]', 'ARCHIVE'))
parser.add_argument('--dry-run',
                    action='store_true',
                    help='Test, do not save quiet day curves')
parser.add_argument('-e', '--end-time',
                    help='End time',
                    metavar='DATETIME')
parser.add_argument('--log-level', 
                    choices=['debug', 'info', 'warning', 'error', 'critical'],
                    default='warning',
                    help='Control how much details is printed',
                    metavar='LEVEL')
parser.add_argument('--log-format',
                    default='%(levelname)s:%(message)s',
                    help='Set format of log messages',
                    metavar='FORMAT')
parser.add_argument('--qdc-archive', 
                    action='append',
                    nargs=2,
                    help='Target data archive for project or site',
                    metavar=('PROJECT[/SITE]', 'QDC_ARCHIVE'))
parser.add_argument('--raise-all',
                    action='store_true',
                    help='No exception handling')
parser.add_argument('-s', '--start-time',
                    help='Start time',
                    metavar='DATETIME')
parser.add_argument('--smooth', 
                    action='store_true',
                    default=True,
                    help='Smooth QDC using truncated Fourier series')
parser.add_argument('--no-smooth', 
                    dest='smooth',
                    action='store_false',
                    help='Do not smooth QDC using truncated Fourier series')

parser.add_argument('project_site',
                    nargs='+',
                    metavar="PROJECT[/SITE]")

args = parser.parse_args()
if __name__ == '__main__':
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format=args.log_format)


# Parse and process start and end times. If end time not given use
# start time plus 1 day.
day = np.timedelta64(24, 'h')
st = dt64.parse_datetime64(args.start_time, 'D')
if args.end_time is None:
    et = st + day
else:
    try:
        # Parse as date
        et = dt64.parse_datetime64(args.end_time, 'D')
    except ValueError as e:
        try:
            # Parse as a set of duration values
            et = st
            et_words = args.end_time.split()
            assert len(et_words) % 2 == 0, 'Need even number of words'
            for n in range(0, len(et_words), 2):
                et += np.timedelta64(float(et_words[n]), et_words[n+1])
        except:
            raise
    except:
        raise

logger.debug('Start date: ' + str(st))
logger.debug('End date: ' + str(et))


# Get names of all projects and sites to be processed.
project_list, site_list = ap.parse_project_site_list(args.project_site)

# Process --archive options
if args.archive:
    archive = ap.parse_archive_selection(args.archive)
else:
    archive = {}

# Process --qdc-archive options for target data
if args.qdc_archive:
    qdc_archive = ap.parse_archive_selection(args.qdc_archive)
else:
    qdc_archive = {}

for n in range(len(project_list)):
    project = project_list[n]
    site = site_list[n]

    kwargs = {}
    if project in archive and site in archive[project]:
        kwargs['archive'] = archive[project][site]

    # Get mag data archive to use for source data
    if project in archive and site in archive[project]:
        archive = archive[project][site]
    else:
        archive = None

    # Get QDC archive to use for target data
    if project in qdc_archive and site in qdc_archive[project]:
        qdc_archive = qdc_archive[project][site]
    else:
        qdc_archive = None


    logger.debug('Processing %s/%s' % (project, site))

    # Attempt to import missing projects
    if project not in ap.projects:
        try:
            import_module('auroraplot.datasets.' + project.lower())
        except:
            pass
    
    ax = None
    t1 = dt64.get_start_of_month(st)

    while t1 < et:
        t2 = dt64.get_start_of_next_month(t1)
        try:
            archive, ad = ap.get_archive_info(project, site, 
                                              'MagData', 
                                              archive=archive)

            mag_data = ap.load_data(project, site, 'MagData', t1, t2,
                                    archive=archive,
                                    raise_all=args.raise_all)
            if mag_data is not None:
                mag_qdc = mag_data.make_qdc(smooth=args.smooth)
                qdc_archive, qdc_ad \
                    = ap.get_archive_info(project, site, 'MagQDC',
                                          archive=qdc_archive)

                filename = dt64.strftime(t1, qdc_ad['path'])
                p = os.path.dirname(filename)
                if not os.path.isdir(p):
                    os.makedirs(p)
                if args.dry_run:
                    logger.info('Dry run, not saving QDC to ' + filename)
                else:
                    fmt = ['%d']
                    fmt.extend(['%.3f'] * len(qdc_ad['channels']))
                    mag_qdc.savetxt(filename, fmt=fmt)

        finally:
            t1 = t2

