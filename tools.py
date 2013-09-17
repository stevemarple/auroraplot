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
    
    
