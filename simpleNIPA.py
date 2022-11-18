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
        self.valid = None

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

    def bootcorr(    self, var, ntim = 1000, corrconf = 0.95, bootconf = 0.80,
                    debug = False, quick = True    ):
        from numpy import meshgrid, zeros, ma, isnan, linspace
        from utils import vcorr, sig_test

        corrlevel = 1 - corrconf
        
        fieldData = self.sst.data
        
        if var == 'Z500':
            #finding geopotential height from geopotential by dividing for 
            #gravity acceleration --> (m^2 / s^2) / (m / s^2)
            fieldData = fieldData/9.80665
        elif var == 'MSLP':
            #converting MSLP from Pa to hPa (because of coherence with 
            #geopotential that is in hPa)
            fieldData = fieldData/100
        #subtract mean pixel by pixel to find the anomalies
        for i in range(fieldData.shape[1]):
            for j in range(fieldData.shape[2]):
                fieldData[:,i,j] = fieldData[:,i,j] - fieldData[:,i,j].mean()
        
        clim_data = self.clim_data
        

        corr_grid = vcorr(X = fieldData, y = clim_data)

        n_yrs = len(clim_data)

        p_value = sig_test(corr_grid, n_yrs)
        
        '''
        this mask will take from the correlation map only the pixels that:
            - have a correlation > 0.6 and lower than < -0.6
            - have a p_value  < than 0.05
            
        NOTE: the operation are the opposite with respect to the ones described 
        in the above comment because the mask must have True values for the pixels
        to be masked (not cosidered) and False values for the pixels not to be 
        masked (considered)
        '''
        #Prepare significance level mask
        significance_mask =  (~(p_value < corrlevel))
        #Mask insignificant gridpoints
        corr_grid = ma.masked_array(corr_grid, significance_mask)
        #Mask land
        corr_grid = ma.masked_array(corr_grid, isnan(corr_grid))
        #Prepare correlation mask
        min_corr = 0.6
        corr_mask = (corr_grid >-min_corr) & (corr_grid < min_corr)
        #Mask not highly correlated gridpoints
        corr_grid = ma.masked_array(corr_grid, corr_mask)
        #Mask northern/southern ocean
        corr_grid.mask[self.sst.lat > 60] = True
        corr_grid.mask[self.sst.lat < -30] = True
        nlat = len(self.sst.lat)
        nlon = len(self.sst.lon)
        #Count the number of visualized pixels
        count = 0
        for row in corr_grid.mask:
            for item in row:
                if item == False:
                    count += 1
                    
        #Check area with convolutions
        import tensorflow as tf
        # boolean map to be checked
        x_in = (~corr_grid.mask).astype(int).reshape(1,121,240,1)
        x = tf.constant(x_in, dtype=tf.float32)
        # convolution kernel definition
        l = 3
        kernel_in = np.ones((l,l,1,1)) # [filter_height, filter_width, in_channels, out_channels]
        kernel = tf.constant(kernel_in, dtype=tf.float32)
        # execution of the convolutions
        result = tf.nn.conv2d(x, kernel, strides=[1, l, l, 1], padding='VALID').numpy()
        
        # check
        
        if (l**2 not in result):
            self.valid = False
        elif (l**2 in result):
            self.valid = True

        if quick:
            self.corr_grid = corr_grid
            self.n_pre_grid = nlat * nlon - corr_grid.mask.sum()
            self.count = count
            self.min_corr = min_corr
            
            if self.n_pre_grid == 0:
                self.flags['noSST'] = True
            else:
                self.flags['noSST'] = False
            return
        ###INITIALIZE A NEW CORR GRID####

        count = np.zeros((nlat,nlon))


        dat = clim_data.copy()


        for boot in range(ntim):

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

    def crossvalpcr(self, M, var, phase, n_comp, xval = True, debug = False):
        #Must set phase with bootcorr, and then use crossvalpcr, as it just uses the corr_grid attribute
        import numpy as np
        from numpy import array
        from scipy.stats import pearsonr as corr
        from scipy.stats import linregress
        from matplotlib import pyplot as plt
        from utils import weightsst
        from sklearn import linear_model
        from sklearn.feature_selection import r_regression
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
            rawdata = self.sst.data[:, sstidx]
            if var == 'Z500':
                #finding geopotential height from geopotential by dividing for 
                #gravity acceleration --> (m^2 / s^2) / (m / s^2)
                rawdata = rawdata/9.80665
            elif var == 'MSLP':
                #converting MSLP from Pa to hPa (because of coherence with 
                #geopotential that is in hPa)
                rawdata = rawdata/100
            #subtract mean column by column to center the data for the PCA
            for i in range(rawdata.shape[1]):
                rawdata[:,i] = rawdata[:,i] - rawdata[:,i].mean()
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
            ncomp = n_comp
            #######
            eigvec = eigvec[:,eigval.argsort()[::-1]]
            #######
            '''
            #################################################################
            ########### Trend of cumulative explained variance ##############
            x = eigval.argsort()[::-1]
            y = [eigval[:i].sum()/eigval.sum() for i,val in enumerate(eigval)]
            plt.plot(x[:10],y[:10])
            plt.show()
            ################################################################
            '''
            # selection of the eigvec related to the largest eigval 
            eof_1 = eigvec[:,:ncomp] #_fv stands for Feature Vector, in this case EOF-1
            # conversion of eigvec in real umber
            eof_1 = np.real(eof_1)
            # dot product between original matrix (nxm) and the eigenvector matrix (mxk) to obtain the PC matrix (nxk)
            pc_1 = eof_1.T.dot(rawdata.T).squeeze()
            if n_comp == 1:
                slope, intercept, r, p, err = linregress(pc_1, predictand)
                yhat = slope * pc_1 + intercept
            elif n_comp == 2:
                reg = linear_model.LinearRegression()
                reg.fit(pc_1.T, predictand)
                slope = reg.coef_
                intercept = reg.intercept_
                r = r_regression(pc_1.T, predictand)
                yhat = slope[0]*pc_1[0] + slope[1]*pc_1[1] + intercept
            self.pc1 = pc_1
            self.correlation = r
            self.hindcast = yhat
            # create boolean vector representing the phase in the dataset (0 neg 1 pos)
            if M == 2:
                label_n = [1 for i in range(len(pc_1))]
                label_p = [2 for i in range(len(pc_1))]
                if phase == 'neg':
                    label = np.array(label_n)
                elif phase == 'pos':
                    label = np.array(label_p)
                ###
                if n_comp == 1:
                    data = pd.DataFrame([pc_1,predictand,label]).T
                    data.columns = ['pc1','target','phase_label']
                    # coverting label column from float to integer (we want 0 and 1 not 0.0 and 1.0)
                    data['phase_label'] = pd.to_numeric(data['phase_label'], downcast='integer')
                    self.dataset = data
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
