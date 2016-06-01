#!/usr/bin/env python

import argparse
import os
import time
import logging
import numpy as np
import shutil
import traceback

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

# Define command line arguments
parser = \
    argparse.ArgumentParser(description='Create/update local data cache')

parser.add_argument('-a', '--archive', 
                    action='append',
                    nargs=2,
                    help='Source data archive for project or site',
                    metavar=('PROJECT[/SITE]', 'ARCHIVE'))
parser.add_argument('--dataset',
                    nargs='+',
                    help='Import additional dataset(s)',
                    metavar='MODULE')
parser.add_argument('-n', '--dry-run',
                    action='store_true',
                    help='Dry run, copy but do not save')
parser.add_argument('-s', '--start-time', 
                    default='today',
                    help='Start time for data transfer (inclusive)',
                    metavar='DATETIME')
parser.add_argument('-e', '--end-time',
                    help='End time for data transfer (exclusive)',
                    metavar='DATETIME')
parser.add_argument('--overwrite',
                    action='store_true',
                    help='Overwrite existing files')
parser.add_argument('--raise-all',
                    action='store_true',
                    help='Raise all exceptions, do not continue on error')
parser.add_argument('--log-level', 
                    choices=['debug', 'info', 'warning', 'error', 'critical'],
                    default='warning',
                    help='Control how much detail is printed',
                    metavar='LEVEL')
parser.add_argument('--log-format',
                    # default='%(levelname)s:%(message)s',
                    help='Set format of log messages',
                    metavar='FORMAT')

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


for n in range(len(project_list)):
    project = project_list[n]
    site = site_list[n]

    # Get archive to use for data
    if project in archive and site in archive[project]:
        archive = archive[project][site]
    else:
        archive = None

    dest_an, dest_ai = ap.get_archive_info(project, site, 'MagData',
                                           archive=archive)

    src_an, src_ai = ap.get_archive_info(project, site, 'MagData',
                                         archive='original_' + dest_an)
    src_path = src_ai['path']
    dest_path = dest_ai['path']
    print('src_ai ' + src_an)
    print(repr(src_ai))
    print('dest_ai ' + dest_an)
    print(repr(dest_ai))
    
    # Tune start/end times to avoid requesting data outside of
    # operational period
    site_st = ap.get_site_info(project, site, 'start_time')
    if site_st is None or site_st < st:
        site_st = st
    else:
        site_st = dt64.floor(site_st, day)
    site_st = dt64.floor(site_st, src_ai['duration'])
        
    site_et = ap.get_site_info(project, site, 'end_time')
    if site_et is None or site_et > et:
        site_et = et
    else:
        site_et = dt64.ceil(site_et, day)
    site_et = dt64.ceil(site_et, src_ai['duration'])
    

    logger.info('Processing %s/%s %s', project, site, dt64.
                fmt_dt64_range(site_st, site_et))
    for t in dt64.dt64_range(site_st, site_et, src_ai['duration']):
        temp_file_name = None
        try:
            if hasattr(dest_path, '__call__'):
                # Function: call it with relevant information to get
                # the dest_path
                dest_file_name = dest_path(t,
                                           project=project,
                                           site=site, 
                                           data_type=data_type,
                                           archive=dest_an,
                                           channels=channels)
            else:
                dest_file_name = dt64.strftime(t, dest_path)

            url_parts = urlparse(dest_file_name)
            if url_parts.scheme in ('ftp', 'http', 'https'):
                raise Exception('Cannot store to a remote location')
            elif url_parts.scheme == 'file':
                dest_file_name = url_parts.path

            if os.path.exists(dest_file_name) and not args.overwrite:
                logger.info('%s already exists', dest_file_name)
                continue
            
            
            if hasattr(src_path, '__call__'):
                # Function: call it with relevant information to get
                # the src_path
                file_name = src_path(t,
                                     project=project,
                                     site=site, 
                                     data_type=data_type,
                                     archive=src_an,
                                     channels=channels)
            else:
                file_name = dt64.strftime(t, src_path)

            url_parts = urlparse(file_name)
            if url_parts.scheme in ('ftp', 'http', 'https'):
                file_name = ap.download_url(file_name)
                if file_name is None:
                    continue
                temp_file_name = file_name
            elif url_parts.scheme == 'file':
                file_name = url_parts.path

            if not os.path.exists(file_name):
                logger.info('missing file %s', file_name)
                continue




            if os.path.exists(dest_file_name) and \
               os.path.samefile(file_name, dest_file_name):
                raise Exception('Refusing to overwrite source file')

            logger.info('creating %s', dest_file_name)
            if not args.dry_run:
                d = os.path.dirname(dest_file_name)
                if not os.path.exists(d):
                    logger.debug('creating directory %s', d)
                    os.makedirs(d)
                shutil.copyfile(file_name, dest_file_name)
            
            
        except Exception as e:
            if args.raise_all:
                raise
            logger.info('Could not cache ' + file_name)
            logger.debug(str(e))
            logger.debug(traceback.format_exc())

        finally:
            if temp_file_name:
                logger.debug('deleting temporary file ' + temp_file_name)
                os.unlink(temp_file_name)

            

logger.info('Done')
