import copy
import os
import re
import urllib2

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import urllib2

import scipy.stats
import warnings

import dt64tools as dt64

epoch64_ns = np.datetime64('1970-01-01T00:00:00Z','ns')

networks = { }


def testFunc(lst):
    lst.pop()
    return lst


NaN = float('nan')
NaT = np.timedelta64('NaT', 'ns')

colors = ['b', 'g', 'r']

# def safe_eval(s):
#     '''Like eval but without allowing the user to access builtin
#     functions or locals.'''
#     return eval(s, {'__builtins__': None}, {})


#class Instrument(object):
#

# def copy_dict(d, *keys):
#     '''Make a copy of only the `keys` from dictionary `d`.'''
#     return {key: d[key] for key in keys}


verbose = True
def add_network(network_name, sites):
    if networks.has_key(network_name):
        networks[network_name].update(sites)
    else:
        networks[network_name] = sites


def str_units(val, unit, prefix=None, sep=None, degrees_dir=None,
              fmt='%(adj)g%(sep)s%(prefix)s%(unit)s%(dir)s', ascii=True,
              wantstr=True):
    '''Return a string formatted with its units
    val: data value
    unit: SI or other unit
    prefix: standard prefix to use with unit, or None for automatic selection
    sep: separator, ' ' unless units are degrees when it is empty
    degrees_dir: direction indicator for degrees, length 2 iterable eg, ['N', 'S']
    fmt: format specifier
    ascii: if true use u as prefix for micro
    wantstr: if true return th formatted string, else return dict of info
    '''
    
    is_degrees = unicode(unit) in (u'deg', u'degrees', u'\N{DEGREE SIGN}')
    prefixes = {'y': -24, # yocto
                'z': -21, # zepto
                'a': -18, # atto
                'f': -15, # femto 
                'p': -12, # pico
                'n': -9,  # nano
                u'\N{MICRO SIGN}': -6, # micro
                'u': -6,  # micro
                'm': -3,  # milli
                'c': -2,  # centi
                'd': -1,  # deci
                '': 0,
                'da': 1,  # deca
                'h': 2,   # hecto
                'k': 3,   # kilo
                'M': 6,   # mega
                'G': 9,   # giga
                'T': 12,  # tera
                'P': 15,  # peta
                'E': 18,  # exa
                'Z': 21,  # zetta
                'Y': 24,  # yotta
                }

    d = {'sep': sep,
         'prefix': prefix,
         'unit': unit,
         'val': val}
    if is_degrees:
        if sep is None:
            d['sep'] = ''
        if prefix is None:
            d['prefix'] = '' # Do not calculate automatically
            d['mul'] = 1
            # prefix = ''
    elif sep is None:
        d['sep'] = ' '

    if d['prefix'] is None:
        if np.isfinite(val) and val != 0:
            logmul = int(np.floor((np.spacing(np.log10(val)) + 
                                           np.log10(val)) / 3.0) * 3)
        else:
            logmul = 1
        # Find matching prefix
        for k in prefixes:
            if prefixes[k] == logmul:
                d['prefix'] = k
                break
        assert d['prefix'] is not None
        d['mul'] = 10 ** logmul

    else:
        d['mul'] = 10 ** prefixes[d['prefix']]

    if ascii and d['mul'] >= 9e-07 and d['mul'] < 11e-7:
        d['prefix'] = 'u'

    if d['mul'] == 1:
        d['adj'] = val
    else:
        d['adj'] = val / d['mul']

    if is_degrees and degrees_dir is not None:
        # include N/S or other direction indicator
        if val >= 0:
            d['dir'] = degrees_dir[0] 
        else:
            d['dir'] = degrees_dir[1]
            d['adj'] = -d['adj']
    else:
        d['dir'] = ''

    if wantstr:
        return fmt % d
    else:
        d['fmt'] = fmt
        d['str'] = fmt % d
        d['fmtunit'] = d['prefix'] + unit
        return d


### TODO: allow load path to be overriden by function argument
# def load_data(network, site, data_type, 
#               start_time, end_time,
#               archive=None, channels=None, path=None, 
#               resolution=None, verbose=None):
def load_data(network, site, data_type, start_time, end_time, **kwargs):
    '''Load data. 
    network: name of the network (upper case)

    site: site abbreviation (upper case)

    data_type: class name of the data type to be loaded

    start_time: start time (inclusive) of the data set

    end_time: end time (exclusive) of the data set
    
    The following optional parameters are recognised: 
    
    archive: name of the archive. Required if more than one archive is
        present and there is not an archive called "default".

    channels: data channel(s) to load. All are loaded if not specified

    path: URL or file path, specified as a strftime format specifier. 
        Alternatively can be a function reference which is passed the time 
        and returns the filename.

    load_function: Pass responsibility for loading the data to the given
        function reference, after validating the input parameters.
        
    verbose: flag to indicate if verbose messages should be
        printed. If None then the global verbose parameter is checked.

    '''
    # Sanity checking
    if not networks.has_key(network):
        raise Exception('Unknown network')
    elif not networks[network].has_key(site):
        raise Exception('Unknown site')
    elif not networks[network][site]['data_types'].has_key(data_type):
        raise Exception('Unknown data_type')

    if kwargs.get('archive') is None:
        if len(networks[network][site]['data_types'][data_type]) == 1:
            # Only one archive, so default is implicit
            archive = networks[network][site]['data_types'][data_type].keys()[0]
        elif networks[network][site]['data_types'][data_type].has_key('default'):
            # Use explicit default
            archive = 'default'
        else:
            raise Exception('archive must be specified')
    else:
        archive = kwargs.get('archive')

    if not networks[network][site]['data_types'][data_type].has_key(archive):
        raise Exception('Unknown archive')

    # archive data
    ad = networks[network][site]['data_types'][data_type][archive] 
    channels = kwargs.get('channels')
    if channels:
        # Could be as single channel name or a list of channels
        if isinstance(channels, basestring):
            if channels not in ad['channels']:
                raise Exception('Unknown channel')
        else:
            for c in channels:
                if c not in ad['channels']:
                    raise Exception('Unknown channel')
    else:
        channels = ad['channels']

    verbose = kwargs.get('verbose', globals()['verbose'])
    path = kwargs.get('path', ad['path'])

    load_function=None
    if kwargs.has_key(load_function):
        load_function = kwargs['load_function']
    if ad.has_key('load_function'):
        load_function = ad['load_function']


    kwargs2 = kwargs.copy()
    kwargs2['archive'] = archive
    kwargs2['channels'] = channels
    kwargs2['load_function'] = load_function
    kwargs2['path'] = path
    kwargs2['verbose'] = verbose
        
    if load_function:
        # Pass responsibility for loading to some other
        # function. Parameters have already been checked and verbose
        # is set to the value the user desires.
        return load_function(network, site, data_type, start_time, 
                             end_time, **kwargs2)

    data = []
    t = start_time
    while t < end_time:
        t2 = t + ad['duration']
        if hasattr(path, '__call__'):
            # Function: call it with relevant information to get the path
            file_name = path(t, network=network, site=site, 
                             data_type=data_type, archive=archive,
                             channels=channels)
        else:
            file_name = dt64.strftime64(t, path)

        if verbose:
            print('loading ' + file_name)

        try:
            tmp = ad['converter'](file_name, 
                                  ad,
                                  network=network,
                                  site=site, 
                                  data_type=data_type, 
                                  start_time=t, 
                                  end_time=t2, **kwargs2)
            if tmp is not None:
                data.append(tmp)
        except Exception as e:
            if verbose:
                print('Could not load ' + file_name + ' ' + str(e))

                        
        
        t = t2
        
    r = concatenate(data).sort(inplace=True)
    r.extract(inplace=True, 
              start_time=start_time, 
              end_time=end_time, 
              channels=channels)
    return r


def concatenate(objs, sort=True):
    obj_type = type(objs[0])
    network = objs[0].network
    site = objs[0].site
    channels = objs[0].channels
    start_time = []
    end_time = []
    sam_st_list = [] # sample start times
    sam_et_list = [] # sample start times
    integration_interval = []
    cadence_list = []
    data_list = []
    units = objs[0].units
    for a in objs:
        assert(type(a) == obj_type)
        assert(a.network == network)
        assert(a.site == site)
        assert(np.all(a.channels == channels))
        assert(a.units == units)
        start_time.append(a.start_time)
        end_time.append(a.end_time)
        sam_st_list.append(a.sample_start_time)
        sam_et_list.append(a.sample_end_time)
        if a.integration_interval is None:
            # All integration intervals must be discarded
            integration_interval = None
        elif integration_interval is not None:
            integration_interval.append(a.integration_interval)
        cadence_list.append(a.nominal_cadence)
        data_list.append(a.data)

    if integration_interval is not None:
        integration_interval=np.concatenate(integration_interval, axis=1)
    return obj_type(network=network,
                    site=site,
                    channels=channels,
                    start_time=np.min(start_time),
                    end_time=np.max(end_time),
                    sample_start_time=np.concatenate(sam_st_list),
                    sample_end_time=np.concatenate(sam_et_list),
                    integration_interval=integration_interval,
                    nominal_cadence=np.max(cadence_list),
                    data=np.concatenate(data_list, axis=1),
                    units=units,
                    sort=sort)


class Data(object):
    '''Base class for time-series data.'''

    def __init__(self, 
                 network=None,
                 site=None,
                 channels=None,
                 start_time=None,
                 end_time=None,
                 sample_start_time=np.array([]), 
                 sample_end_time=np.array([]), 
                 integration_interval=None, 
                 nominal_cadence=None,
                 data=np.array([]),
                 units=None,
                 sort=True):
        self.network = network
        self.site = site
        self.channels = np.array(channels)
        self.start_time = start_time
        self.end_time = end_time
        self.sample_start_time = sample_start_time
        self.sample_end_time = sample_end_time
        self.integration_interval=integration_interval
        self.nominal_cadence = nominal_cadence
        self.data = data
        self.units = units
        if sort:
            self.sort(inplace=True)

    def __repr__(self):
        return (type(self).__name__ + ':\n' +
                '          network : ' + str(self.network) + '\n' +
                '             site : ' + str(self.site) + '\n' +
                '         channels : ' + str(self.channels) + '\n' +
                '       start_time : ' + str(self.start_time) + '\n' +
                '         end_time : ' + str(self.end_time) + '\n' +
                'sample_start_time : ' + repr(self.sample_start_time) + '\n' + 
                '  sample_end_time : ' + repr(self.sample_end_time) + '\n' + 
                'integration intv. : ' + repr(self.integration_interval)+'\n'+
                '   nominal cadence: ' + str(self.nominal_cadence) + '\n' +
                '             data : ' + repr(self.data) + '\n' + 
                '            units : ' + str(self.units))

    def data_description(self):
        return 'Data'

    def assert_valid(self):
        for n in ('network', 'site', 'channels', 'start_time', 'end_time', 
                  'sample_start_time', 'sample_end_time',
                  'nominal_cadence',
                  'data', 'units'):
            attr = getattr(self, n)
            assert (attr is not None and 
                    (not isinstance(attr, basestring) or attr != '')), \
                    n + ' not set'
        assert re.match('^[-A-Z0-9]+$', self.network), 'Bad value for network'
        assert re.match('^[-A-Z0-9]+$', self.site), 'Bad value for site'

        num_channels = len(self.channels)
        num_samples = len(self.sample_start_time)

        assert self.start_time <= self.end_time, \
            'start_time must not be after end_time'
        assert len(self.sample_end_time) == num_samples, \
            'lengths of sample_start_time and sample_end_time differ'
       
        if self.integration_interval is not None:
            assert self.integration_interval.shape == \
                (num_channels, num_samples), \
                'integration_interval incorrect shape'
        assert self.data.shape == (num_channels, num_samples), \
            'data incorrect shape'

        return True

    def extract(self, start_time=None, end_time=None, channels=None, 
                inplace=False):
        if inplace:
            r = self
        else:
            r = copy.deepcopy(self)
        if channels is not None:
            chan_tup = tuple(self.channels)
            cidx = []
            for c in channels:
                cidx.append(chan_tup.index(c))
            r.channels = r.channels[cidx]
            r.data = r.data[cidx, :]
            if r.integration_interval is not None:
                r.integration_interval = r.integration_interval[cidx, :]
            
        if start_time is not None or end_time is not None:
            if start_time is None:
                start_time = self.start_time
            if end_time is None:
                end_time = self.end_time
            tidx = (self.sample_start_time >= start_time) & \
                (self.sample_end_time < end_time)
            r.start_time = start_time
            r.end_time = end_time
            r.sample_start_time = r.sample_start_time[tidx]
            r.sample_end_time = r.sample_end_time[tidx]
            r.data = r.data[:, tidx]
            if r.integration_interval is not None:
                r.integration_interval = r.integration_interval[:, tidx]

        # elif channels is None and not inplace:
        #     # data matrix has not been copied
        #     r.data = r.data.copy()
        return r

    def sort(self, inplace=False):
        idx = np.argsort(self.sample_start_time)
        if inplace:
            new = self
        else:
            new = copy.deepcopy(self)

        new.sample_start_time = new.sample_start_time[idx]
        new.sample_end_time = new.sample_end_time[idx]

        # integration_interval may be None, or an array
        if self.integration_interval is not None:
            new.integration_interval = new.integration_interval[:, idx]

        new.data = new.data[:, idx]
        return new
            

    def plot(self, channels=None, figure=None, axes=None, subplot=None,
             units_prefix=None, subtitle=None):
        if channels is None:
            channels=self.channels
        elif isinstance(channels, basestring):
            channels=[channels]
        else:
            try:
                iterator = iter(channels)
            except TypeError:
                channels = [channels]
        
        if axes is not None:
            axes2 = copy.copy(axes)
            try:
                axes2.reverse()
            except:
                pass
        elif figure is None:
            figure=plt.figure()
        else:
            plt.figure(figure)
        

        if subplot is not None:
            try:
                iterator = iter(subplot)
                subplot2 = copy.copy(subplot) # will reverse in place
                subplot2.reverse()
            except:
                subplot2 = [subplot]
            if len(subplot2) == 1:
                subplot2 *= len(channels)
            else:
                assert len(subplot2) == len(channels), \
                    'subplot and channels must be same length'

        if subtitle is None:
            subtitle = self.data_description()

        chan_tup = tuple(self.channels)

        # Label the Y axis with channel if possible
        need_legend = not(len(channels) == 1 or
                          (hasattr(axes, '__iter__') and len(axes) > 1) or
                          (hasattr(subplot, '__iter__') and len(subplot) > 1))
        if need_legend:
            # Precompute the units information (prefix, multiplier etc)
            # since it must be the same for all channels
            allcidx = []
            for c in channels:
                allcidx.append(chan_tup.index(c))
            cu = str_units(np.nanmax(np.abs(self.data[allcidx])), self.units, 
                           prefix=units_prefix, wantstr=False)

        for c in channels:
            if axes is not None:
                try:
                    ax = plt.axes(axes2.pop())
                except:
                    ax = plt.axes(axes2)
            elif subplot is not None:
                try:
                    p = subplot2.pop()
                    ax = plt.subplot(p)
                except Exception as e:
                    print(str(e))
                    ax = plt.subplot(subplot2)
            else:
                ax = plt.gca()
            ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useOffset=False))
            cidx = chan_tup.index(c)
            u = str_units(np.nanmax(np.abs(self.data[cidx])), self.units, 
                          prefix=units_prefix, wantstr=False)
            xdata = dt64.mean(self.sample_start_time, self.sample_end_time)
            if u['mul'] == 1:
                # Can avoid copying
                ydata = self.data[cidx]
            else:
                ydata = self.data[cidx] / u['mul']

            r = dt64.plot_dt64(xdata, ydata)
            
            plt.xlim(xmin=dt64.dt64_to(self.start_time, 'ns'),
                     xmax=dt64.dt64_to(self.end_time, 'ns'))

            if not need_legend:
                # Lines plotted on different axes
                # u['channel'] = self.channels[cidx]
                # plt.ylabel('%(channel)s (%(prefix)s%(unit)s)' % u)
                plt.ylabel(self.channels[cidx] + ' (' + u['fmtunit'] + ')')

        if need_legend:
            lh = plt.legend(self.channels[allcidx], loc='best', fancybox=True)
            lh.get_frame().set_alpha(0.7)
            # Label Y axis
            plt.ylabel(str(subtitle) + ' (' + cu['fmtunit'] + ')')

        # Add title
        tstr = self.network + ' / ' + self.site
        # if len(channels) == 1:
        #    tstr += '\n' + self.channels[0]
        if subtitle:
            tstr += '\n' + subtitle
        # tstr += '\n' + dt64.fmt_dt64_range(self.start_time, self.end_time)
        plt.title(tstr)
        return r

    def set_cadence(self, cadence, method=scipy.stats.nanmean, 
                    offset_interval=np.timedelta64(0, 'ns'), inplace=False):
        if cadence > self.nominal_cadence:
            sam_st = np.arange(dt64.ceil(self.start_time, cadence) 
                               + offset_interval, self.end_time, cadence)
            sam_et = sam_st + cadence

            sample_time = dt64.mean(self.sample_start_time, 
                                    self.sample_end_time)
            d = np.empty([len(self.channels), len(sam_st)])
            if self.integration_interval is not None:
                integ_intv = np.empty([len(self.channels), len(sam_st)], 
                                      dtype=self.integration_interval.dtype)
            else:
                integ_intv = None
            for sn in range(len(sam_st)):
                tidx = np.where(np.logical_and(sample_time >= sam_st[sn],
                                               sample_time <= sam_et[sn]))[0]
                ### TODO: Resulting integration interval cannot take into
                ### account nans which occur in one channel but not
                ### another
                
                for cn in range(len(self.channels)):
                    if self.integration_interval is not None:
                        notnanidx = np.where(np.logical_not(np.isnan(self.data[cn, tidx])))[0]
                        integ_intv[cn,sn] = np.nansum(self.integration_interval[cn, tidx[notnanidx]])
                        
                    d[cn,sn] = method(self.data[cn, tidx])
                    
                

            if inplace:
                r = self
            else:
                r = copy.copy(self)
                for k in (set(self.__dict__.keys())
                          - set(['sample_start_time', 'sample_end_time', 
                                 'integration_interval', 'nominal_cadence',
                                 'data'])):
                    setattr(r, k, copy.deepcopy(getattr(self, k)))

            r.sample_start_time = sam_st
            r.sample_end_time = sam_et
            r.integration_interval = integ_intv
            r.nominal_cadence = copy.copy(cadence)
            r.data = d
            return r
        elif cadence < self.nominal_cadence:
            raise Exception('Interpolation to reduce cadence not implemented')
        else:
            # raise Warning('Cadence already at ' + str(cadence))
            warnings.warn('Cadence already at ' + str(cadence))
            if inplace:
                return self
            else:
                return copy.deepcopy(self)
        
        
    def mark_missing_data(self, cadence=None, inplace=False):
        if cadence is None:
            cadence = self.nominal_cadence
        sample_time = dt64.mean(self.sample_start_time, 
                                self.sample_end_time)
        idx = np.where(np.diff(sample_time) > cadence)[0]
        
        if inplace:
            r = self
        else:
            r = copy.deepcopy(self)
        
        data = np.ones([len(self.channels), len(idx)]) * NaN
        if self.integration_interval is None:
            ii = None
        else:
            ii = np.ones([len(self.channels), len(idx)]) * NaT

        missing = type(self)(network=self.network,
                             site=self.site,
                             channels=self.channels,
                             start_time=self.start_time,
                             end_time=self.end_time,
                             sample_start_time=self.sample_end_time[idx],
                             sample_end_time=self.sample_start_time[idx+1],
                             integration_interval=ii,
                             nominal_cadence=self.nominal_cadence,
                             data=data,
                             units=self.units,
                             sort=False)
        
        r = concatenate([self, missing])
        if inplace:
            self = r
        return r

class MagData(Data):
    '''Class to manipulate and display magnetometer data.'''

    def __init__(self,
                 network=None,
                 site=None,
                 channels=None,
                 start_time=None,
                 end_time=None,
                 sample_start_time=np.array([]),
                 sample_end_time=np.array([]),
                 integration_interval=np.array([]),
                 nominal_cadence=None,
                 data=np.array([]),
                 units=None,
                 sort=True):
        Data.__init__(self,
                      network=network,
                      site=site,
                      channels=channels,
                      start_time=start_time,
                      end_time=end_time,
                      sample_start_time=sample_start_time,
                      sample_end_time=sample_end_time,
                      integration_interval=integration_interval,
                      nominal_cadence=nominal_cadence,
                      data=data,
                      units=units,
                      sort=sort)

    def data_description(self):
        return 'Magnetic field'

    def save(self, filename):
        np.savetxt(filename, np.array([self.sample_start_time, self.data[0],
                                       self.data[1], self.data[2]]).transpose())

    def plot(self, channels=None, figure=None, axes=None, subplot=None, 
             units_prefix=None, subtitle=None):
        if channels is None:
            channels = self.channels
        if subplot is None:
            subplot = []
            for n in range(1, len(channels) + 1):
                subplot.append(len(channels)*100 + 10 + n)
        if units_prefix is None and self.units == 'T':
            # Use nT for plotting
            units_prefix = 'n'
        
        return Data.plot(self, channels=channels, figure=figure, axes=axes,
                         subplot=subplot, units_prefix=units_prefix,
                         subtitle=subtitle)

class MagQDC(MagData):
    '''Class to load and manipulate magnetometer quiet-day curves (QDC).'''
    def __init__(self,
                 network=None,
                 site=None,
                 channels=None,
                 start_time=None,
                 end_time=None,
                 sample_start_time=np.array([]),
                 sample_end_time=np.array([]),
                 integration_interval=np.array([]),
                 nominal_cadence=None,
                 data=np.array([]),
                 units=None,
                 sort=True):
        MagData.__init__(self,
                         network=network,
                         site=site,
                         channels=channels,
                         start_time=start_time,
                         end_time=end_time,
                         sample_start_time=sample_start_time,
                         sample_end_time=sample_end_time,
                         integration_interval=integration_interval,
                         nominal_cadence=nominal_cadence,
                         data=data,
                         units=units,
                         sort=sort)

    def data_description(self):
        return 'Magnetic field QDC'



class TemperatureData(Data):
    '''Class to manipulate and display temperature data.'''
    def __init__(self,
                 network=None,
                 site=None,
                 channels=None,
                 start_time=None,
                 end_time=None,
                 sample_start_time=np.array([]),
                 sample_end_time=np.array([]),
                 integration_interval=np.array([]),
                 nominal_cadence=None,
                 data=np.array([]),
                 units=None,
                 sort=True):
        Data.__init__(self,
                      network=network,
                      site=site,
                      channels=channels,
                      start_time=start_time,
                      end_time=end_time,
                      sample_start_time=sample_start_time,
                      sample_end_time=sample_end_time,
                      integration_interval=integration_interval,
                      nominal_cadence=nominal_cadence,
                      data=data,
                      units=units,
                      sort=sort)

    def data_description(self):
        return 'Temperature'

    def set_units(self, units, inplace=False):
        if inplace:
            r = self
        else:
            r = copy.deepcopy(self)
        # Accept various representations for degrees Celsius. The
        # final one uses the single-character unicode represention for
        # deg C.
        celsius = ['Celsius', 'C', u'\N{DEGREE SIGN}C', u'\u2103']
        if units in celsius:
            if r.units in celsius:
                return r
            elif r.units == 'K':
                # K -> Celsius
                r.data -= 273.15
                r.units = units
            else:
                raise Exception('Unknown units')
        elif units == 'K':
            if r.units in celsius:
                # Celsius -> K
                r.data += 273.15
                r.units = units
            elif r.units == 'K':
                return r
            else:
                raise Exception('Unknown units')
        else:
            raise Exception('Unknown units')
        return r

class VoltageData(Data):
    '''Class to manipulate and display voltage data.'''
    def __init__(self,
                 network=None,
                 site=None,
                 channels=None,
                 start_time=None,
                 end_time=None,
                 sample_start_time=np.array([]),
                 sample_end_time=np.array([]),
                 integration_interval=np.array([]),
                 nominal_cadence=None,
                 data=np.array([]),
                 units=None,
                 sort=True):
        Data.__init__(self,
                      network=network,
                      site=site,
                      channels=channels,
                      start_time=start_time,
                      end_time=end_time,
                      sample_start_time=sample_start_time,
                      sample_end_time=sample_end_time,
                      integration_interval=integration_interval,
                      nominal_cadence=nominal_cadence,
                      data=data,
                      units=units,
                      sort=sort)

    def data_description(self):
        return 'Voltage'
