import auroraplot as ap
from socket import gethostname

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
        try:
            awn, site_lc = gethostname().split(':')
        except:
            # Hostname incorrect format
            return
        
        if not ap.networks[network_name].has_key(site):
            return # Unknown site

        # For data which matches the local hostname convert all URLS
        # to local paths.
        site = site_lc
        for dtv in ap.networks[network_name][site]['data_types'].values():
            for av in dtv.values(): # archive values
                av['path'] = av['path'].replace(
                    'http://aurorawatch.lancs.ac.uk/data/aurorawatchnet', 
                    '/data/aurorawatchnet')
