import copy
import logging
import os
import tempfile

import numpy as np
import scipy.signal

import auroraplot as ap

logger = logging.getLogger(__name__)
class NthLargest(object):
    '''
    Class to calculate the Nth largest value from a numpy array-like
    value. If 'smallest' is True then calculate the Nth smallest.
    '''

    def __init__(self, n, smallest=False):
        self.n = n
        self.smallest = smallest

    def __call__(self, a, weights=None):
        '''
        Calculate the largest (or smallest value). An optional
        argument, 'weights', is accepted for compatibility with
        scipy.average but is ignored.
        '''
        af = np.array(a).flatten()
        af = af[np.logical_not(np.isnan(af))] # Remove Nans
        if self.smallest:
            # Smallest first (standard sort order)
            b = af
        else:
            # Want largest first, so change sign
            b = -af

        idx = b.argsort()

        if np.max(self.n) >= af.size:
            return np.nan
        return np.mean(af[idx[self.n]])


class smart_open:
    """Smarter way to open files for writing.

    This class mimics the behaviour of 'with open(file) as handle:'
    but with two additional benefits:

    1. When opening a file for write access (including appending) the
    parent directory will be created automatically if it does not
    already exist.

    2. When opening a file for write access it is created with a
    temporary name ('.tmp' appended) and if there are no errors it is
    renamed automatically when closed. In the case of errors the
    default is to delete the temporary file (set delete_temp_on_error
    to False to keep the temporary file).

    The use of a temporary file is automatically disabled when the mode
    includes 'r', 'x', 'a', or '+'. It can be prevented manually by
    setting temp_ext to an empty string.

    Example use

        with smart_open('/tmp/somedir/somefile.txt', 'w') as f:
            f.write('some text')

    """

    def __init__(self,
                 filename,
                 mode='r',
                 temp_ext='.tmp',
                 delete_temp_on_error=True):
        self.filename = filename
        self.mode = mode
        # Modes which rely on an existing file should not change the name
        if (temp_ext is None or 'r' in self.mode or 'x' in self.mode
            or 'a' in self.mode or '+' in self.mode):
            self.temp_ext = ''
        else:
            self.temp_ext = temp_ext
        self.delete_temp_on_error = delete_temp_on_error
        self.file = None

    def __enter__(self):
        d = os.path.dirname(self.filename)
        if (not os.path.exists(d) and
            ('w' in self.mode or 'x' in self.mode
             or 'a' in self.mode or '+' in self.mode)):
            logger.debug('creating directory %s', d)
            os.makedirs(d)
        self.file = open(self.filename + self.temp_ext, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_value, traceback):
        if self.file is not None:
            if not self.file.closed:
                self.file.close()
            if self.file.name != self.filename:
                if exc_type is None:
                    logger.debug('renaming %s to %s',
                                 self.file.name, self.filename)
                    os.rename(self.file.name, self.filename)
                elif self.delete_temp_on_error:
                    os.remove(self.file.name)


def atomic_symlink(src, dst):
    """Create or update a symlink atomically."""
    dst_dir = os.path.dirname(dst)
    tmp = None
    try:
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        tmp = tempfile.mktemp(dir=dst_dir)
        os.symlink(src, tmp)
        os.rename(tmp, dst)
    except Exception as e:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)
        raise


def least_squares_error(a, b):
    e = np.power(a - b, 2)
    return np.sum(e[np.isfinite(e)]), False

def minimise_sign_error(a, b):
    e = np.sign(a - b)
    imbalance = np.sum(e[np.isfinite(e)])
    if abs(imbalance) <= 1:
        # Close enough, especially when total number is odd
        imbalance = 0
    return imbalance , True

def fit_data(data, ref_data, err_func=None, tolerance=None,
             max_iterations=50, full_output=False, plot_fit=False):

    # Assume datasets are aligned in time already
    assert not np.all(np.logical_or(np.isnan(data), np.isnan(ref_data))), \
        'No common samples which are not NaN'

    # Calculate initial starting values to iterate between
    e = data - ref_data
    data1 = data - np.nanmin(e); # upper limit
    data2 = data - np.nanmax(e); # lower limit

    # Compute initial errors
    e1, has_sign = err_func(data1, ref_data)
    e2, tmp      = err_func(data2, ref_data)


    non_nan_idx = np.where(np.isfinite(data)[0])[0][0]
    if tolerance is None:
        # Compute tolerance which will be solved in less than max iterations.
        tolerance = (data1[non_nan_idx] - data2[non_nan_idx]) \
            / 2**(max_iterations-3)

    iterations = 0
    while True:
        iterations += 1
        difference = data1[non_nan_idx] - data2[non_nan_idx]
        test_data = data1 - 0.5 * difference

        test_error, has_sign = err_func(test_data, ref_data)

        if test_error == 0:
            # On target!
            break
        elif np.abs(difference) < np.abs(tolerance):
            # Close enough, but which is to be used?
            if np.abs(e1) < np.min([np.abs(e2), np.abs(test_error)]):
                # data1 is closest
                test_data = data1
                test_error = e1
            elif np.abs(e2) < np.abs(test_error):
                # data2 is closest
                test_data = data2
                test_error = e2
            else:
                # test_data is closest
                pass
            break

        else:
            # Too high or low
            if not has_sign:
                # Compute the direction to move in, indicate this by
                # setting the sign of test_error accordingly
                test_error2, tmp = err_func(test_data + 0.5*tolerance,
                                            ref_data)

                if test_error > test_error2:
                    test_error = -test_error # Too low
                elif test_error < test_error2:
                    # Too high, so leave test_error as positive
                    pass
                else:
                    # The errors are equal? Could loop, trying a
                    # bigger step size, say for 10 times. Seems
                    # unlikely.
                    raise Exception('Cannot find direction to iterate')

            if test_error > 0:
                # Too high, go halfway between test_data and data2
                data1 = test_data
                e1 = test_error
            else:
                data2 = test_data
                e2 = test_error

        if iterations >= max_iterations:
            mesg = ('No solution after %(max_iterations)d iterations, ' +
                    'tolerance: %(tolerance)g, ' +
                    'difference: %(difference)g, ') % locals()
            raise Exception(mesg)


    if full_output:
        stats = {'iterations': iterations,
                 'error': data[non_nan_idx] - test_data[non_nan_idx],
                 'tolerance': tolerance,
                 'difference': difference,
                 }

        return test_data, data[non_nan_idx] - test_data[non_nan_idx], \
            stats
    else:
        return test_data



# Savitzky-Golay filter, from Scipy cookbook,
# http://wiki.scipy.org/Cookbook/SavitzkyGolay
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def sgolay_filt(data, window_size, order):
    '''Filter auroraplot.data objects.

    data: auroraplot.data object

    window_size: odd integer or numpy.timedelta64 value. When
        timedelta64 an off integer window size is derived using the
        nominal cadence of the data. No conversion to regular data is
        performed.
    order: order of fit
    
    returns: modified auroraplot.data object
    '''
    r = copy.deepcopy(data)
    if isinstance(window_size, np.timedelta64):
        window_size = int(window_size / data.nominal_cadence)
        if window_size % 2 == 0:
            window_size -= 1 # Must be odd

    for n in range(len(data.channels)):
        r.data[n] = savitzky_golay(data.data[n], window_size, order)
    return r


def walk_data_archives(callback, project,
                       site_list=None,
                       data_type_list=None,
                       archive_list=None):
    '''Walk through a project's data archives

    The callback function is called once for each data archive, with the parameters
    project, site, data type, archive name and archive data.
    '''

    if site_list is None:
        site_list = ap.get_sites(project)

    for site in site_list:
        if data_type_list is None:
            dt_list = ap.projects[project]['sites'][site]['data_types'].keys()
        else:
            dt_list = data_type_list

        for data_type in dt_list:
            dtv = ap.projects[project]['sites'][site]['data_types'][data_type]
            # If only one archive and default not set then make it the
            # default before adding copies
            dtv_keys = list(dtv.keys())
            if len(dtv_keys) == 1 and dtv_keys[0] != 'default':
                dtv['default'] = dtv_keys[0]

            if archive_list is None:
                a_list = dtv_keys
            else:
                a_list = archive_list

            for archive in a_list:
                if archive == 'default':
                    continue
                callback(project, site, data_type, archive, dtv[archive])


def change_load_data_paths(project,
                           replace,
                           site_list=None,
                           data_type_list=None,
                           archive_list=None,
                           load_converter=None,
                           cache_dir=None):
    '''Helper function for changing paths used when loading data

    change_load_data_paths is intended to be called from within
    auroraplot_custom.py to alter the paths used when loading data,
    for instance to modify the URLs to local file paths
    '''
    def callback(project, site, data_type, archive, archive_data):
        orig_archive = 'original_' + archive
        if orig_archive not in ap.projects[project]['sites'][site]['data_types'][data_type]:
            # Keep a copy of the original
            ap.projects[project]['sites'][site]['data_types'][data_type][orig_archive] = copy.deepcopy(archive_data)
        if not hasattr(archive_data['path'], '__call__'):
            archive_data['path'] = replace(archive_data['path'], project, site, data_type, archive)

    walk_data_archives(callback,
                       project,
                       site_list=site_list,
                       data_type_list=data_type_list,
                       archive_list=archive_list)

def cache_data_files(cache_dir, project,
                     site_list=None,
                     data_type_list=None,
                     archive_list=None):
    '''Helper function for caching data locally

    cache_data_files is intended to be called from within
    auroraplot_custom.py to alter the paths used when loading data.
    '''

    def callback(project, site, data_type, archive, archive_data):
        archive_data['cache_dir'] = os.path.join(cache_dir, project, site)

    walk_data_archives(callback,
                       project,
                       site_list=site_list,
                       data_type_list=data_type_list,
                       archive_list=archive_list)


def lookup_module_name(s):
    last_dot = s.rindex('.')
    module = s[:last_dot]
    name = s[(last_dot+1):]
    return (module, name)




