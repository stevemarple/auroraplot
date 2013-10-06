import numpy as np

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
        # Compute tolerance whihc will be solved in less than max iterations.
        tolerance = (data1[non_nan_idx] - data2[non_nan_idx]) \
            / 2**(max_iterations-1)
        
    iterations = 0
    while True:
        iterations += 1
        # e1, has_sign = err_func(data1, ref_data)
        # e2, has_sign = err_func(data2, ref_data)

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
                e1 = test_error # NEW
            else:
                data2 = test_data
                e2 = test_error # NEW
            
        if iterations >= max_iterations:
            raise Exception('No solution after ' + str(max_iterations)
                            + ' iterations.')
        
        
    if full_output:
        stats = {'iterations': iterations,
                 'error': data[non_nan_idx] - test_data[non_nan_idx],
                 'tolerance': tolerance,
                 }
        
        return test_data, data[non_nan_idx] - test_data[non_nan_idx], \
            stats
    else:
        return test_data

    
    
