import cachetools
import operator


class CachedFunc(object):
    '''Class to cache function return values (memoize).
    
    An object of this class can cache function return values using
    cachetools. However unlike the normal decorators this class is
    able to set the cache size at run-time. The __call__ operator is
    defined so that the object can be called in exactly the same way
    as the original function that was passed to the class constructor.
    '''
    
    def __init__(self, func, cache_class=cachetools.LRUCache, **kwargs):
        self.cache = cache_class(**kwargs)
        self.func = func

    @cachetools.cachedmethod(operator.attrgetter('cache'))
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

