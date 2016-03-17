# An example file to customise the load paths for auroraplot. To use
# copy this file to somewhere on the python module patch and rename to
# auroraplot_custom.py

import os.path
import sys
if sys.version_info[0] >= 3:
    import configparser
    from configparser import SafeConfigParser
else:
    import ConfigParser
    from ConfigParser import SafeConfigParser

import auroraplot as ap



def add_project_hook(project_name):
    '''This function is used to customise the paths to data files when
    datasets are imported.

    When project_name is "AURORAWATCHNET" /etc/awnet.ini defines the
    magnetometer site name then remap the URLs for the site to local
    filenames, eg for use on the AuroraWatchNet magnetometer data
    loggers.

    '''
    if project_name == 'AURORAWATCHNET':
        filename = '/etc/awnet.ini'
        if not os.path.exists(filename):
            return
        try:
            config = SafeConfigParser()
            config.read(filename)
            site = config.get('magnetometer', 'site').upper()
        except Exception as e:
            print('Bad config file ' + filename + ': ' + str(e))
            return
        
        if site not in ap.projects[project_name]:
            return # Unknown site

        # For data which matches this host's site convert all URLS to
        # local paths.
        for dtv in ap.projects[project_name][site]['data_types'].values():
            for an,av in dtv.iteritems(): # archive name/values
                if an == 'default':
                    continue
                av['path'] = av['path'].replace(
                    'http://aurorawatch.lancs.ac.uk/data/aurorawatchnet', 
                    '/data/aurorawatchnet')

