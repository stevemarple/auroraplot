# An example file to customise the load paths for auroraplot. To use
# Copy this file to somewhere on the python module patch and rename to
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



def add_network_hook(network_name):
    '''
    This function is used to customise the paths to data files when
    datasets are imported.

    When network_name is "AURORAWATCHNET" and the computer's hostname
    has the form awn-<site>, where <site> is a valid sitename (case
    ignored) for AURORAWATCHNET then the URLs are for the local site
    are mapped to local filenames, eg for use on the AuroraWatchNet
    magnetometer data loggers.
    '''
    if network_name == 'AURORAWATCHNET':
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
        
        if not ap.networks[network_name].has_key(site):
            return # Unknown site

        # For data which matches this host's site convert all URLS to
        # local paths.
        for dtv in ap.networks[network_name][site]['data_types'].values():
            for av in dtv.values(): # archive values
                av['path'] = av['path'].replace(
                    'http://aurorawatch.lancs.ac.uk/data/aurorawatchnet', 
                    '/data/aurorawatchnet')

