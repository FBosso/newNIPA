#import section
import pandas as pd
from matplotlib import cm, pyplot as plt
import numpy as np
from climdiv_data import *
from simpleNIPA import *
from atmos_ocean_data import *
from utils import sstMap
import matplotlib as mpl
from utils import *
import mpl_toolkits
import csv
import math
import xarray as xr




#### USER INPUT ####
local_datas = ['tp_netherlands_cumul', 't2m_netherlands']
aggrs = [3,2,3]
months_complete = [i+1 for i in range(12)]
indices = ['SCA','EA','ENSO-mei','NAO']
global_datas = ['SST','MSLP','Z500']
# Experimental
n_comp = 1
#############

'''
ind = ['SCA']
global_data = ['Z500']
local_data = ['t2m_netherlands']
aggr = [3]
month = [3]
'''
'''
the dimensions are respectively:
    - 3: Aggregation level of global variable
    - 2: Considered local variable
    - 12: Month
    - 4: Cimate Indices
    - 3: Global variables
'''

long = indices*2
for i,sig in enumerate(long[:4]):
    long[i] = sig + '_N'
    
for i,sig in enumerate(long[4:]):
    long[i+4] = sig + '_P'
    
pearson_table = np.zeros((2,3,12,8,3))
#replace the zero with a number which cannot be confused with a pearson correlation number (for debugging purposes)
pearson_table[pearson_table == 0] = 2

for local_data_i, local_data in enumerate(local_datas):
    for aggr_i, aggr in enumerate(aggrs):
        for month_i, month in enumerate(months_complete):
            for ind_i, ind in enumerate(indices):
                for global_data_i, global_data in enumerate(global_datas):
                    # Select the input-output files:
                    #climate index for "pos"/"neg" segmentation (in this case NAO)
                    #path to the input data for North Atlantic Oscillation (NAO)
                    index_file = f'./DATA/{ind}.txt'
                    index_label = index_file.split('/')[-1].split('.')[0]
                    ## NB:  the file in this path contains the values of a specific climate index 
                    #       (such as NAO, MJO, ecc). Thus, there is not lat/lon involved because 
                    #       Climate Indices are single values that represent the behavior of a 
                    #       climate signal. In conclusion: this is a simple temporal 
                    #       serie.
                    
                    #path to the dataset of daily precipitation covering the working area. 
                    #This file allows to know the precipitation for the correlation analysis 
                    #between location-specific precipitation and golbal vars (such as SST, Z200, 
                    #Z500, Z850, ...)
                    clim_file = f'./DATA/{local_data}.txt'
                    loc_var = clim_file.split('/')[-1].split('.')[0].split('_')[0]
                    ## NB:  the file in this path contains the precipitation data only for the 
                    #       desired location (single cell). 
                    #       Thus, there is not lat/lon involved; this is a simple temporal serie 
                    #       located in the working area.
                    ####
                    data_var = f'/Users/francesco/Documents/data_1,5x1,5/{global_data}'
                    var = data_var.split('/')[-1]
                    ####
                    
                    # Original Settings:
                    M = 2              # number of climate signal's phases
                    n_obs = aggr           # number of observations (months)
                    lag = aggr             # lag-time (months) --> 3 = seasonal
                    months = [month]        # months to consider [1,2,3] = (J,F,M)
                    # if month = [12] we will obtain the PC1 to build the NN for the forecast of
                    # December precipitation (with november data)
                    startyr = 1980      # beginning of the time period to analyze
                    n_yrs = 40          # number of years to analyze
                    
                    #defining the name of the output PNG file
                    filename = f'{index_label}_{var}-{n_obs}_{loc_var}-{month}'
                    
                    
                    # creation of an array of years from the starting year to the ending year of the analysis
                    years = np.arange(startyr, startyr+n_yrs)
                    
                    
                    # Select the type of experiment:
                    # crv_flag:
                    #   True  = runs NIPA with crossvalidation
                    #   False = runs NIPA without crossvalidation and save the first SST-Principal Component for multi-variate model
                    #
                    crv_flag = False
                    map_flag = True
                    
                    ####################
                    
                    
                    
                    
                    #this function takes information about the seasons, years, and type of divisional
                    #data to look at, and creates appropriate kwgroups (parameters) to load the data
                    kwgroups = create_kwgroups(debug = True, climdata_months = months,
                                            climdata_startyr = startyr, n_yrs = n_yrs,
                                            n_mon_sst = n_obs, n_mon_index = n_obs, sst_lag = lag,
                                            n_phases = M, phases_even = True,
                                            index_lag = lag,
                                            index_fp = index_file,
                                            climdata_fp = clim_file,
                                            var = var)
                    
                    
                    
                    #here we call the get_data function giving the just created kwgroups as input
                    #with this function we download the missing data: SST (in lat,lon,time space)
                    
                    #After we will perform a PCA in order to reduce the dimensionality of this 
                    #dataset to a single dimension in order also to compare the SST with the other 
                    #monodimensional vars (NAO,precipitation)
                    climdata, sst, index, phaseind = get_data(kwgroups, data_var)
                    
                    # create a specific folder
                    if os.path.exists(f'./maps/{index_label}_{var}-{n_obs}_{loc_var}') == False:
                        os.mkdir(f'./maps/{index_label}_{var}-{n_obs}_{loc_var}')
                    #here we set up where to save the output map
                    fp = f'./maps/{index_label}_{var}-{n_obs}_{loc_var}/{filename}'
                    
                    #here the plot settings are initialized
                    fig, axes = plt.subplots(M, 1, figsize = (6, 12))
                    
                    #here a dictionary with 3 keys (year, data, hindcast) is created to be then
                    #filled during the process
                    timeseries = {'years' : [], 'data' : [], 'hindcast': []}
                    
                    #here is where the PC1 will be stored
                    pc1 = {'pc1':[]}
                    #here is where the dataset will be stored
                    dataset = []
                    
                    print('\n\nNIPA running...')
                    
                    
                    #START OF THE ANALYTICAL PART
                    
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    #this section will be executed only if we consider the climate signal as "single phased" (M=1)
                    if M == 1:
                        phase = 'allyears'
                        model = NIPAphase(climdata, sst, index, phaseind[phase])
                        model.phase = phase
                        model.years = years[phaseind[phase]]
                        model.bootcorr(corrconf = 0.95)
                        model.gridCheck()
                        model.crossvalpcr(xval = crv_flag)
                        timeseries['years'] = model.years
                        timeseries['data'] = model.clim_data
                        timeseries['hindcast'] = model.hindcast
                        print( timeseries['years'])
                        print( timeseries['data'])
                        
                        if map_flag:
                            fig, axes, m = sstMap(model, fig = fig, ax = axes)
                            axes.set_title('%s, %.2f' % (phase, model.correlation))
                            fig.savefig(fp)
                            plt.close(fig)
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    #this section will be executed only with pluri-phased climate signals 
                    #(it actually works only for 2 phases)
                    else:
                        save_condition = []
                        for phase, ax in zip(phaseind, axes):  
                            model = NIPAphase(climdata, sst, index, phaseind[phase])
                            
                            #assignment of the phase of the climate signal to the key named "phase" in the i-esima iteration
                            model.phase = phase
                            
                            #this line does the following operations:
                            
                            #   0. The  "phaseind" contains 2 items (neg & pos) composed by 
                            #   boolean values related to the accordance/discordance wrt the 
                            #   name of the item itself (a 'False' boolean value in the 'neg' 
                            #   item corrisponds to a postive phase)
                            
                            #   1. "phaseind[phase]" selects the item corresponding to the 
                            #   considered phase for this iteration (if the considered phase 
                            #   is 'neg' --> "phaseind[phase]" == "phaseind['neg']" ).
                            
                            #   2. The pattern of boolen values obtained at step 1. is then used 
                            #   to select the subset of years corresponding to the true values 
                            #   with "years[phaseind[phase]]" (if the result of "phaseind[phase]" 
                            #   is "[True, False, True, False] and the full set of years is [
                            #   2001, 2002, 2003, 2004], then the result will be [2001,2003]")
                            model.years = years[phaseind[phase]]
                            
                            #here the minimum correlation values between preseasonal SST and 
                            #precipitation is set (if the correlation is lower than 0.95 the 
                            #SST values of that location are not considered)
                            model.bootcorr(corrconf = 0.95)
                            save_condition.append(model.valid)
                            
                            #given that "gridCheck()" is a function inside the "NIPAphase" class 
                            #and given that we assigned to the variable "model" an instance of 
                            #the NIPAphase class, now we can simply call the "gridCheck()" 
                            #function (and any other function inside that class) by simply using
                            #the notation "model.<function>"
                            model.gridCheck()
                            
                            # this fuction is the one that actually computes the PC1 and the pearson coefficient
                            model.crossvalpcr(global_data, phase, n_comp, xval = crv_flag)
                            
                            ##### Assignment of pearson value to the right element of the matrix "pearson_table"
                            if (phase == 'neg') and (save_condition[0] == True):
                                pearson_table[local_data_i,aggr_i,month_i,ind_i,global_data_i] = model.correlation
                            elif (phase == 'neg') and (save_condition[0] == False):
                                pearson_table[local_data_i,aggr_i,month_i,ind_i,global_data_i] = 0
                            elif (phase == 'pos') and (save_condition[1] == True):
                                pearson_table[local_data_i,aggr_i,month_i,ind_i+4,global_data_i] = model.correlation
                            elif (phase == 'pos') and (save_condition[1] == False):
                                pearson_table[local_data_i,aggr_i,month_i,ind_i+4,global_data_i] = 0
                            
                            #append model.years values identified before to the key "years" in the dictionary "timeseries"
                            timeseries['years'].append(model.years)
                            #append model.data values to the key "data" in the dictionary "timeseries"
                            timeseries['data'].append(model.clim_data)
                            #append model.hindcast values to the key "hindcast" in the dictionary "timeseries"
                            timeseries['hindcast'].append(model.hindcast)
                            
                            #if the crossvalidation value is set to False the first principal 
                            #component value is stored into the "pc1" key of the "pc1" vriable
                            #NOT SURE ABOUT THE "WHY" OF THIS
                            if not crv_flag:
                                if hasattr(model,'pc1'):
                                    pc1['pc1'].append(model.pc1)
                                    dataset.append(model.dataset)
                                    
                            # Select the maximum absolute value of correlation
                            maximum = model.corr_grid.max()
                            minimum = model.corr_grid.min()
                            strongest = 0
                            if max(abs(maximum), abs(minimum)) == abs(maximum):
                                strongest = round(maximum,2)
                            elif max(abs(maximum), abs(minimum)) == abs(minimum):
                                strongest = round(minimum,2)
                                
                                
                            
                            #if "map_flag" variable is set on "true", then the resultin plot is produced
                            if map_flag:
                                #the data for the plot are stored in the model variable (passed in the sstMap function)
                                fig, ax, m = sstMap(model, fig = fig, ax = ax)
                                #ax.set_title('%s, %.2f' % (phase, model.correlation))
                                if n_comp == 1:
                                    ax.set_title(f'phase:{phase}   pearson:{round(model.correlation,2)}  \n\n  min_possible_corr:±{model.min_corr}   max_corr:{strongest}  count:{model.count}')
                                elif n_comp == 2:
                                    ax.set_title(f'phase:{phase}   pearson:{[round(model.correlation[0],2), round(model.correlation[1],2)]}  \n\n  min_possible_corr:±{model.min_corr}   max_corr:{strongest}  count:{model.count}')
                                fig.savefig(fp)
                                plt.close(fig)
                              
                    if map_flag:
                        if (save_condition[0] == False) or (save_condition[1] == False):
                            import os
                            os.remove(fp+'.png')
                                
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++          
                    
                    
                    ### ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ###
                    
                    # save timeseries (exceptions handled only for 2 phase analysis)
                    # this part of the code is ment to handle exceptions in the case in which 
                    # the analysis is executed with 2 phaes
                    
                    #if the hindcast for the first instant in the timeseries is only 1
                    if np.size(timeseries['hindcast'][0]) == 1:
                        #and is also NaN
                        if math.isnan(timeseries['hindcast'][0]):
                            # it means that there is no result for the first phase -> consider 
                            # only the second set of results
                            timeseries['years'] = timeseries['years'][1]
                            timeseries['data'] = timeseries['data'][1]
                            timeseries['hindcast'] = timeseries['hindcast'][1]
                            
                    ## SAME FOR THE SECOND INSTANT
                    #if the hindcast for the second instant in the timeseries is only 1
                    elif np.size(timeseries['hindcast'][1]) == 1: ##### QUA DA UN ERRORE FORSE ...
                        #and is also NaN
                        if math.isnan(timeseries['hindcast'][1]):
                            # it means that there is no result for the second phase -> consider 
                            # only the first set of results
                            timeseries['years'] = timeseries['years'][0]
                            timeseries['data'] = timeseries['data'][0]
                            timeseries['hindcast'] = timeseries['hindcast'][0]
                         
                    #if both the first and the second phase have results, do a concatenation 
                    #and use them both
                    
                    else:
                        timeseries['years'] = np.concatenate(timeseries['years'])
                        timeseries['data'] = np.concatenate(timeseries['data'])
                        timeseries['hindcast'] = np.concatenate(timeseries['hindcast'])
                    
                    
                    ### ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ###
                    
                    # once the exceptions have been handled (only first phase, only second phase 
                    # or both phses present) the data are saved in a pandas dataframe
                    df = pd.DataFrame(timeseries)
                    
                    if (save_condition[0] == True) and (save_condition[1] == True):
                    
                        # creation of a direcotry
                        if os.path.exists(f'./output/{index_label}_{var}-{n_obs}_{loc_var}') == False:
                            os.mkdir(f'./output/{index_label}_{var}-{n_obs}_{loc_var}')
                        
                        # the dataframe is saved in a csv file
                        ts_file = f'./output/{index_label}_{var}-{n_obs}_{loc_var}/{filename}_timeseries.csv'
                        df.to_csv(ts_file)
                    
                    # if "crv_flag" was set to False a csv file containing the first principal 
                    # components of the SSTs is also saved (NOT CLEAR WHY IT IS NOT CREATED IF 
                    # "crv_flag" is set to false)
                    #if (not crv_flag) and (not model.flags['noSST']) :
                    if (not crv_flag) and ((save_condition[0] == True) and (save_condition[1] == True)) :
                        # save PC
                        pc1['pc1'] = np.concatenate(pc1['pc1'])
                        pc_file = f'./output/{index_label}_{var}-{n_obs}_{loc_var}/{filename}_pc1SST.csv'
                        dataset_file = f'./output/{index_label}_{var}-{n_obs}_{loc_var}/{filename}_dataset.csv'
                        dataset = pd.concat(dataset)
                        if n_comp == 1:
                            df1 = pd.DataFrame(pc1)
                        elif n_comp == 2:
                            pc = {'pc1':np.concatenate((pc1['pc1'].T[:,0],pc1['pc1'].T[:,2])),
                                  'pc2':np.concatenate((pc1['pc1'].T[:,1],pc1['pc1'].T[:,3]))}
                            df1 = pd.DataFrame(pc)
                            
                        df1.to_csv(pc_file)
                        dataset.to_csv(dataset_file)
                        
                    ### SAVING PEARSON TABLE ###
                    
                    if (ind == indices[-1]) and (global_data == global_datas[-1]):
                        
                        #this section replaces the pearson coefficients with zero 
                        #for all the combination in which the minimum areal 
                        #constraint is not met even only for one of the two phases 
                        #(the areal constraint could be ok for negative phase 
                        #but not for the positive one --> in that case we do 
                        #not want consider both pearson coefficients)
                        for i in range(len(indices)):
                            for j in range(len(global_datas)):
                                if (pearson_table[local_data_i,aggr_i,month_i][i,j] == 0) or (pearson_table[local_data_i,aggr_i,month_i][i+len(indices),j] == 0):
                                    pearson_table[local_data_i,aggr_i,month_i][i,j] = 0
                                    pearson_table[local_data_i,aggr_i,month_i][i+len(indices),j] = 0
                    
                        df_mon = pd.DataFrame(pearson_table[local_data_i,aggr_i,month_i])
                        df_mon.columns = global_datas
                        df_mon.index = long
                        
                        df_mon.to_excel(f'./pearson_tables/{local_data}-{aggr}-{month}.xlsx')
                    
                    ############################
                    
                    print( 'NIPA run completed ✅')  
                    
                    
                    filename = f'{index_label}_{var}-{n_obs}_{loc_var}-{month}'
                        
                               
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
