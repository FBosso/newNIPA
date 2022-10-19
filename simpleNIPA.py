"""
New NIPA module
"""

import os
import pandas as pd
import numpy as np
from collections import namedtuple
seasonal_var = namedtuple('seasonal_var', ('data','lat','lon'))
from os import environ as EV


class NIPAphase(object):
    """
    Class and methods for operations on phases as determined by the MEI.

    _INPUTS
    phaseind:    dictionary containing phase names as keys and corresponding booleans as index vectors
    clim_data:    n x 1 pandas time series of the climate data (predictands)
    sst:        dictionary containing the keys 'data', 'lat', and 'lon'
    slp:        dictionary containing the keys 'data', 'lat', and 'lon'
    mei:        n x 1 pandas time series containing averaged MEI values

    _ATTRIBUTES
    sstcorr_grid
    slpcorr_grid

    """
    def __init__(self, clim_data, sst, mei, phaseind, alt = False):
        if alt:
            self.clim_data = clim_data
        else:
            self.clim_data = clim_data[phaseind]
        self.sst = seasonal_var(sst.data[phaseind], sst.lat, sst.lon)
        self.mei = mei[phaseind]
        self.flags = {}

        """
        sst is a named tuple

        """
        return
    def categorize(self, ncat = 3, hindcast = False):
        from pandas import Series
        from numpy import sort

        if hindcast:
            data = self.hindcast.copy()
        else:
            data = self.clim_data.copy()
        x = sort(data)
        n = len(x)
        upper = x[((2 * n) / ncat) - 1]
        lower = x[(n / ncat) - 1]


        cat_dat = Series(index = data.index)


        for year in data.index:
            if data[year] <= lower:
                cat_dat[year] = 'B'
            elif data[year] > upper:
                cat_dat[year] = 'A'
            else:
                cat_dat[year] = 'N'
        if hindcast:
            self.hcat = cat_dat
        else:
            self.cat = cat_dat
        return

    def bootcorr(    self, ntim = 1000, corrconf = 0.95, bootconf = 0.80,
                    debug = False, quick = True    ):
        from numpy import meshgrid, zeros, ma, isnan, linspace
        from utils import vcorr, sig_test

        corrlevel = 1 - corrconf

        fieldData = self.sst.data
        clim_data = self.clim_data

        corr_grid = vcorr(X = fieldData, y = clim_data)

        n_yrs = len(clim_data)

        p_value = sig_test(corr_grid, n_yrs)

        #Mask insignificant gridpoints
        corr_grid = ma.masked_array(corr_grid, ~(p_value < corrlevel))
        #Mask land
        corr_grid = ma.masked_array(corr_grid, isnan(corr_grid))
        #Mask northern/southern ocean
        corr_grid.mask[self.sst.lat > 60] = True
        corr_grid.mask[self.sst.lat < -30] = True
        nlat = len(self.sst.lat)
        nlon = len(self.sst.lon)

        if quick:
            self.corr_grid = corr_grid
            self.n_pre_grid = nlat * nlon - corr_grid.mask.sum()
            if self.n_pre_grid == 0:
                self.flags['noSST'] = True
            else:
                self.flags['noSST'] = False
            return
        ###INITIALIZE A NEW CORR GRID####

        count = np.zeros((nlat,nlon))


        dat = clim_data.copy()


        for boot in xrange(ntim):

            ###SHUFFLE THE YEARS AND CREATE THE BOOT DATA###
            idx = np.random.randint(0, len(dat) - 1, len(dat))
            boot_fieldData = np.zeros((len(idx), nlat, nlon))
            boot_fieldData[:] = fieldData[idx]
            boot_climData = np.zeros((len(idx)))
            boot_climData = dat[idx]

            boot_corr_grid = vcorr(X = boot_fieldData, y = boot_climData)

            p_value = sig_test(boot_corr_grid, n_yrs)

            count[p_value <= corrlevel] += 1
            if debug:
                print( 'Count max is %i' % count.max())


        ###CREATE MASKED ARRAY USING THE COUNT AND BOOTCONF ATTRIBUTES
        corr_grid = np.ma.masked_array(corr_grid, count < bootconf * ntim)

        self.corr_grid = corr_grid
        self.n_pre_grid = nlat * nlon - corr_grid.mask.sum()
        if self.n_pre_grid == 0:
            self.flags['noSST'] = True
        else:
            self.flags['noSST'] = False
        return

    def gridCheck(self, lim = 5, ntim = 2, debug = False):
        if self.n_pre_grid < 50: lim = 6; ntim = 2
        for time in range(ntim):
            dat = self.corr_grid.mask
            count = 0
            for i in np.arange(1,dat.shape[0]-1):
                for j in np.arange(1,dat.shape[1]-1):
                    if not dat[i,j]:
                        check = np.zeros(8)
                        check[0] = dat[i+1,j]
                        check[1] = dat[i+1,j+1]
                        check[2] = dat[i+1,j-1]
                        check[3] = dat[i,j+1]
                        check[4] = dat[i,j-1]
                        check[5] = dat[i-1,j]
                        check[6] = dat[i-1,j+1]
                        check[7] = dat[i-1,j-1]
                        if check.sum() >= lim:
                            dat[i,j] = True
                            count += 1
            if debug: print( 'Deleted %i grids' % count)

            self.corr_grid.mask = dat
            self.n_post_grid = dat.size - dat.sum()

        return

    def crossvalpcr(self, xval = True, debug = False):
        #Must set phase with bootcorr, and then use crossvalpcr, as it just uses the corr_grid attribute
        import numpy as np
        from numpy import array
        from scipy.stats import pearsonr as corr
        from scipy.stats import linregress
        from matplotlib import pyplot as plt
        from utils import weightsst
        predictand = self.clim_data
        
        if self.corr_grid.mask.sum() >= len(self.sst.lat) * len(self.sst.lon) - 4:
            yhat = np.nan
            e = np.nan
            #index = self.clim_data.index
            index = self.mei
            hindcast = pd.Series(data = yhat, index = index)
            error = pd.Series(data = e, index = index)
            self.correlation = np.nan
            self.hindcast = np.nan
            self.hindcast_error = np.nan
            self.flags['noSST'] = True
            return

        self.flags['noSST'] = False
        sstidx = self.corr_grid.mask == False
        n = len(predictand)
        yhat = np.zeros(n)
        e = np.zeros(n)
        idx = np.arange(n)

        params = []
        std_errs = []
        p_vals = []
        t_vals = []
        if not xval:
            # weight the SST values with the cosine of their latitude
            rawSSTdata = weightsst(self.sst).data
            # take only the SST values considering the correlation mask created before
            rawdata = rawSSTdata[:, sstidx]
            # compute the covariance matrix
            cvr = np.cov(rawdata.T)
            # compute eigenvalues and eigenvectors
            eigval, eigvec = np.linalg.eig(cvr)
            # sort the eigenvalues from the smallest (0) to the largest (-1) and then 
            # flip the order to obtain them from the largest to the smalles ([::-1] does exactly this)
            eigvalsort = np.argsort(eigval)[::-1]
            eigval = eigval[eigvalsort]
            # transformatipn of eigval in real number
            eigval = np.real(eigval)
            # definition of the number of principal components to consider
            ncomp = 1
            #######
            eigvec = eigvec[:,eigval.argsort()[::-1]]
            #######
            # selection of the eigvec related to the largest eigval 
            eof_1 = eigvec[:,:ncomp] #_fv stands for Feature Vector, in this case EOF-1
            # conversion of eigvec in real umber
            eof_1 = np.real(eof_1)
            # dot product between original matrix (nxm) and the eigenvector matrix (mxk) to obtain the PC matrix (nxk)
            pc_1 = eof_1.T.dot(rawdata.T).squeeze()
            slope, intercept, r, p, err = linregress(pc_1, predictand)
            yhat = slope * pc_1 + intercept
            self.pc1 = pc_1
            self.correlation = r
            self.hindcast = yhat
            return

        for i in idx:
            test = idx == i
            train = idx != i
            rawSSTdata = weightsst(self.sst).data[train]
            droppedSSTdata = weightsst(self.sst).data[test]
            rawdata = rawSSTdata[:, sstidx]#
            dropped_data = droppedSSTdata[:,sstidx].squeeze()

            #U, s, V = np.linalg.svd(rawdata)
            #pc_1 = V[0,:] #_Rows of V are principal components
            #eof_1 = U[:,0].squeeze() #_Columns are EOFS
            #EIGs = s**2 #_s is square root of eigenvalues

            cvr = np.cov(rawdata.T)
            #print cvr.shape
            eigval, eigvec = np.linalg.eig(cvr)
            eigvalsort = np.argsort(eigval)[::-1]
            eigval = eigval[eigvalsort]
            eigval = np.real(eigval)
            ncomp = 1
            eof_1 = eigvec[:,:ncomp] #_fv stands for Feature Vector, in this case EOF-1
            eof_1 = np.real(eof_1)
            pc_1 = eof_1.T.dot(rawdata.T).squeeze()

            slope, intercept, r_value, p_value, std_err = linregress(pc_1, predictand[train])
            predictor = dropped_data.dot(eof_1)
            yhat[i] = slope * predictor + intercept
            e[i] = predictand[i] - yhat[i]
            params.append(slope); std_errs.append(std_err); p_vals.append(p_value)
            t_vals.append(slope/std_err)

        r, p = corr(predictand, yhat)

        hindcast = yhat
        error = e
        self.hindcast = hindcast
        self.hindcast_error = error
        self.correlation = round(r, 2)
        self.reg_stats = {    'params' : array(params),
                            'std_errs' : array(std_errs),
                            't_vals' : array(t_vals),
                            'p_vals' : array(p_vals)}

        return

    def genEnsemble(self):
        from numpy import zeros
        from numpy.random import randint
        sd = self.hindcast_error.std()
        mn = self.hindcast_error.mean()
        n = len(self.hindcast)
        ensemble = zeros((1000,n))
        for i in range(n):
            for j in range(1000):
                idx = randint(0,n)
                ensemble[j,i] = self.hindcast[i] + self.hindcast_error[idx]
        self.ensemble = ensemble
        return

    def simple_skillscores(self):
        n = len(self.clim_data)
        lower_ind = n/3
        upper_ind = ( 2*n / 3 )

        maxima = max(self.clim_data[:lower_ind])
