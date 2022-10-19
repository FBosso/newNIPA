"""
Module for loading atmospheric and oceanic data necessary to run NIPA
"""

import os
from os import environ as EV
import sys
import resource


def load_climdata(**kwargs):
    data = load_clim_file(kwargs['fp'])
    from numpy import where, arange, zeros, inf
    from utils import slp_tf
    tran = slp_tf()
    startmon = int(tran[kwargs['months'][0]])
    startyr = kwargs['startyr']
    idx_start = where((data.index.year == startyr) & (data.index.month == startmon))
    idx = []
    [idx.extend(arange(len(kwargs['months'])) + idx_start + 12*n) for n in range(kwargs['n_year'])]
    climdata = zeros((kwargs['n_year']))
    for year, mons in enumerate(idx):
        climdata[year] = data.values[mons].mean()
    return climdata



def load_clim_file(fp, debug = False):
    # This function takes a specified input file, and
    # creates a pandas series with all necessary information
    # to run NIPA
    import numpy as np
    import pandas as pd

    #First get the description and years
    f = open(fp)
    description = f.readline()
    years = f.readline()
    startyr, endyr = years[:4], years[5:9]
    print( description)

    #First load extended index
    data = np.loadtxt(fp, skiprows = 2)
    nyrs = data.shape[0]
    data = data.reshape(data.size) # Make data 1D
    timeargs = {'start'     : startyr + '-01',
                'periods'    : len(data),
                'freq'        : 'M'}
    index = pd.date_range(**timeargs)
    clim_data = pd.Series(data = data, index = index)

    return clim_data



def loadFiles(data_path, version = '3b', debug = False, anomalies = True, **kwargs):
    """
    This function load the file containing the data in the script
    """
    from utils import int_to_month
    from os.path import isfile
    from numpy import arange
    from numpy import squeeze
    import pickle
    import re
    from collections import namedtuple
    
    # FRANCESCO: added imports
    import xarray as xr
    from pydap.model import BaseType
    import numpy as np

    i2m = int_to_month()
    # Keyword arguments for setting up the loading process
    DLargs = {
        'startmon'    : i2m[kwargs['months'][0]],
        'endmon'    : i2m[kwargs['months'][-1]],
        'startyr'    : str(kwargs['startyr']),
        'endyr'        : str(kwargs['endyr']),
        'nbox'         : str(kwargs['n_mon'])
            }
    seasonal_var = namedtuple('seasonal_var', ('data','lat','lon'))

    # ------------------------------------------------------------------------
    
    print('Start loading...')
    
    years = [i for i in range(int(DLargs['startyr'])+1,int(DLargs['endyr'])+2)]
    months = ['01','02','03','04','05','06','07','08','09','10','11','12']
    lista = []
    for year in years:
        for month in months:
            lista.append(xr.open_dataset(f'{data_path}/{year}-{month}_SST.nc', engine='netcdf4').mean(dim='time'))
    tot = xr.concat(lista, dim='time')
    
    print(tot.sst.values.shape)
    
    x = tot.sst.values
    dataset = tot
    #convert Kelvin into Celsius
    grid = x-273.15
    t = dataset.time.values
    
    sstlat = BaseType(data = dataset.lat.values)
    sstlon = BaseType(data = dataset.lon.values)

    print('Loading finished')
    
    # ------------------------------------------------------------------------

    #_Grid has shape (ntim, nlat, nlon)

    nseasons = 12 / kwargs['n_mon']
    if debug:
        print('Number of seasons is %i, number of months is %i' % (nseasons, kwargs['n_mon']))
    ntime = len(t)
    idx = arange(0, ntime, nseasons).astype(int)
    sst = grid[idx]
    sstdata = {'grid':sst, 'lat':sstlat, 'lon':sstlon}
    var = seasonal_var(sst, sstlat, sstlon)
    return var



def create_phase_index2(**kwargs):
    from copy import deepcopy
    import numpy as np
    from numpy import sort
    index = load_clim_file(kwargs['fp'])
    from numpy import where, arange, zeros, inf
    from utils import slp_tf
    tran = slp_tf()
    startmon = int(tran[kwargs['months'][0]])
    startyr = kwargs['startyr']
    idx_start = where((index.index.year == startyr) & (index.index.month == startmon))
    idx = []
    [idx.extend(arange(kwargs['n_mon']) + idx_start + 12*n) for n in range(kwargs['n_year'])]
    index_avg = zeros((kwargs['n_year']))
    for year, mons in enumerate(idx):
        index_avg[year] = index.values[mons].mean()

    idx = np.argsort(index_avg)
    nyrs = kwargs['n_year']
    nphase = kwargs['n_phases']
    phases_even = kwargs['phases_even']
    p = np.zeros((len(index_avg)), dtype = 'bool')
    p1 = deepcopy(p)
    p2 = deepcopy(p)
    p3 = deepcopy(p)
    p4 = deepcopy(p)
    p5 = deepcopy(p)
    phaseind = {}
    if nphase == 1:
        p[idx[:]] = True
        phaseind['allyears'] = p
    if nphase == 2:
        x = nyrs / nphase
        p1[idx[:int(x)]] = True; phaseind['neg'] = p1
        p2[idx[int(x):]] = True; phaseind['pos'] = p2
    if nphase == 3:
        if phases_even:
            x = nyrs / nphase
            x2 = nyrs - x
        else:
            x = nphase / 4
            x2 = nyrs - x
        p1[idx[:x]] = True; phaseind['neg'] = p1
        p2[idx[x:x2]] = True; phaseind['neutral'] = p2
        p3[idx[x2:]] = True; phaseind['pos'] = p3

    if nphase == 4:
        if phases_even:
            x = nyrs / nphase
            x3 = nyrs - x
            xr = (x3 - x) / 2
            x2 = x+xr
        else:
            half = nyrs / 2
            x = int(round(half*0.34))
            x3 = nyrs - x
            xr = (x3 - x) / 2
            x2 = x + xr
        p1[idx[:x]] = True; phaseind['neg'] = p1
        p2[idx[x:x2]] = True; phaseind['neutneg'] = p2
        p3[idx[x2:x3]] = True; phaseind['netpos'] = p3
        p4[idx[x3:]] = True; phaseind['pos'] = p4
    if nphase == 5:
        if phases_even:
            x = nyrs / nphase
            x4 = nyrs - x
            xr = (x4 - x) / 3
            x2 = x+xr
            x3 = x4-xr
        else:
            half = nyrs / 2
            x = int(round(half*0.3))
            x4 = nyrs - x
            xr = (x4 - x) / 3
            x2 = x+xr
            x3 = x4-xr
        p1[idx[:x]] = True; phaseind['neg'] = p1
        p2[idx[x:x2]] = True; phaseind['neutneg'] = p2
        p3[idx[x2:x3]] = True; phaseind['neutral'] = p3
        p4[idx[x3:x4]] = True; phaseind['neutpos'] = p4
        p5[idx[x4:]] = True; phaseind['pos'] = p5
    # if nphase == 6:
    return index_avg, phaseind











