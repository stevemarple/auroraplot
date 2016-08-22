#!/usr/bin/env python

import argparse
import os
import time
import logging
import numpy as np
import traceback

import matplotlib as mpl
import matplotlib.pyplot as plt

if not os.environ.has_key('TZ') or \
   os.environ['TZ'] not in ('UTC', 'UT', 'GMT'):
    try:
        # Try to force all times to be read as UTC
        os.environ['TZ'] = 'UTC'
        time.tzset()
    except Exception as e:
        logger.error(e)
        pass


import auroraplot as ap
import auroraplot.magdata
import auroraplot.dt64tools as dt64
import auroraplot.datasets.aurorawatchnet
import auroraplot.datasets.samnet
import auroraplot.datasets.bgs_schools

# Define command line arguments
parser = \
    argparse.ArgumentParser(description='Compute baseline values')

parser.add_argument('-a', '--archive', 
                    action='append',
                    nargs=2,
                    help='Source data archive for project or site',
                    metavar=('PROJECT[/SITE]', 'ARCHIVE'))
parser.add_argument('--baseline-archive', 
                    action='append',
                    nargs=2,
                    help='Target data archive for project or site',
                    metavar=('PROJECT[/SITE]', 'ARCHIVE'))
parser.add_argument('-s', '--start-time', 
                    default='today',
                    help='Start time for data transfer (inclusive)',
                    metavar='DATETIME')
parser.add_argument('-e', '--end-time',
                    help='End time for data transfer (exclusive)',
                    metavar='DATETIME')
parser.add_argument('--dataset',
                    nargs='+',
                    help='Import additional dataset(s)',
                    metavar='MODULE')
parser.add_argument('--log-level', 
                    choices=['debug', 'info', 'warning', 'error', 'critical'],
                    default='warning',
                    help='Control how much detail is printed',
                    metavar='LEVEL')
parser.add_argument('--log-format',
                    # default='%(levelname)s:%(message)s',
                    help='Set format of log messages',
                    metavar='FORMAT')
parser.add_argument('--realtime-qdc',
                    action='store_true',
                    default=None,
                    help='Use realtime selection for loading QDCs')


parser.add_argument('project_site',
                    nargs='+',
                    metavar="PROJECT/SITE")



args = parser.parse_args()
if __name__ == '__main__':
    d = dict(level=getattr(logging, args.log_level.upper()))
    if args.log_format:
        d['format'] = args.log_format
    logging.basicConfig(**d)

logger = logging.getLogger(__name__)

if args.dataset:
    for ds in args.dataset:
        new_module = 'auroraplot.datasets.' + ds
        try:
            __import__(new_module)
        except Exception as e:
            logger.error('Could not import ' + new_module + ': ' + str(e))
            sys.exit(1)
        
# Parse and process start and end times. If end time not given use
# start time plus 1 day.
day = np.timedelta64(1, 'D')
st = dt64.parse_datetime64(args.start_time, 'D')
if args.end_time is None:
    et = st + day
else:
    try:
        # Parse as date
        et = dt64.parse_datetime64(args.end_time, 'us')
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

project_list, site_list = ap.parse_project_site_list(args.project_site)

# Process --archive options for source data
if args.archive:
    archive = ap.parse_archive_selection(args.archive)
else:
    archive = {}

# Process --baseline-archive options for target data
if args.baseline_archive:
    baseline_archive = ap.parse_archive_selection(args.baseline_archive)
else:
    baseline_archive = {}

st = dt64.floor(st, day)
et = dt64.ceil(et, day)


for n in range(len(project_list)):
    project = project_list[n]
    site = site_list[n]

    # Get baseline archive to use for target data
    if project in baseline_archive and site in baseline_archive[project]:
        bl_archive = baseline_archive[project][site]
    else:
        bl_archive = 'realtime_baseline'

    an, ai = ap.get_archive_info(project, site, 'MagData',
                                 archive=bl_archive)

    if 'qdc_fit_duration' not in  ai:
        logger.error('no qdc_fit_duration found in %s archive for %s/%s',
                     an, project, site)
        continue

    if 'realtime_qdc' not in ai:
        logger.warning('realtime_qdc option not in %s archive for %s/%s',
                       an, project, site)
        
    qdc_fit_duration = ai['qdc_fit_duration']
    qdc_fit_offset = ai.get('qdc_fit_offset', -qdc_fit_duration/2 - 1.5*day)
    qdc_tries = ai.get('qdc_tries', 3)

    # Get mag data archive to use for source data
    if project in archive and site in archive[project]:
        md_archive = archive[project][site]
    else:
        md_archive = None

    # Tune start/end times to avoid requesting data outside of
    # operational period
    site_st = ap.get_site_info(project, site, 'start_time')
    if site_st is None or site_st < st:
        site_st = st
    else:
        site_st = dt64.floor(site_st, day)

    site_et = ap.get_site_info(project, site, 'end_time')        
    if site_et is None or site_et > et:
        site_et = et
    else:
        site_et = dt64.ceil(site_et, day)

    logger.info('Processing %s/%s %s', project, site, dt64.
                fmt_dt64_range(site_st, site_et))
    for t1 in dt64.dt64_range(site_st, site_et, day):
        try:
            t2 = t1 + day
            # Calculate dates for data to be used for fitting
            md_mean_time = dt64.mean(t1, t2) + qdc_fit_offset
            md_st = md_mean_time - qdc_fit_duration/2
            md_et = md_st + qdc_fit_duration

            md = ap.load_data(project, site, 'MagData', md_st, md_et,
                              archive=md_archive)
            if md is None or np.size(md.data) == 0:
                logger.debug('no data')
                continue

            qdc = ap.magdata.load_qdc(project, site, t1, 
                                      tries=qdc_tries,
                                      realtime=ai['realtime_qdc'])
            if qdc is None or np.size(qdc.data) == 0:
                logger.debug('no QDC')
                continue

            # Ensure each channel is zero-mean
            for n in range(qdc.data.shape[0]):
                qdc.data[n] -= ap.nanmean(qdc.data[n])

            fitted_qdc, errors, fit_info = qdc.align(md, \
                fit=ap.data.Data.minimise_sign_error_fit,
                full_output=True,
                tolerance=1e-12)

            data = -np.reshape(errors, [len(errors), 1])

            tu = dt64.smallest_unit([t1,t2,ai['nominal_cadence']])
            t1a = dt64.astype(t1, units=tu)
            t2a = dt64.astype(t2, units=tu)
            cadence = dt64.astype(ai['nominal_cadence'], units=tu)
            bl = ap.magdata.MagData(project=project,
                                    site=site,
                                    channels=ai['channels'],
                                    start_time=t1a,
                                    end_time=t2a,
                                    sample_start_time=np.array([t1a]),
                                    sample_end_time=np.array([t2a]),
                                    nominal_cadence=cadence,
                                    data=data,
                                    units=ai['units'])
            bl.assert_valid()
            bl.save(archive=bl_archive, merge=True)
        except Exception as e:
            logger.error(e)
            logger.debug(traceback.format_exc())

logger.info('Done')
