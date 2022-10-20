
"""
Module for loading climate division data for running NIPA
"""

import os
from atmos_ocean_data import *
from os import environ as EV

def get_data(kwgroups, data_path):
	clim_data = load_climdata(**kwgroups['climdata'])
	sst = loadFiles(data_path, newFormat = True, anomalies = True, **kwgroups['sst'])
	index, phaseind = create_phase_index2(**kwgroups['index'])
	return clim_data, sst, index, phaseind


def create_kwgroups(debug = False, climdata_startyr = 1979, n_yrs = 43, \
	climdata_months = [1,2,3], n_mon_sst = 3, sst_lag = 3, n_mon_slp = 3, \
	slp_lag = 3, n_mon_index = 3, index_lag = 3, n_phases = 2, phases_even = True, \
	index_fp = 'mei.txt', climdata_fp = 'APGD_prcp.txt', var = 'SST'):
	print(climdata_months)
	"""
	This function takes information about the seasons, years, and type of divisional
	data to look at, and creates appropriate kwgroups (parameters) to be input into
	data loading.
	"""
    
	#_Check a few things
	assert climdata_months[0] >= 1, 'Divisonal data can only wrap to the following year'
	assert climdata_months[-1] <= 15, 'DJFM (i.e. [12, 13, 14, 15]) is the biggest wrap allowed'

	#_Following block sets the appropriate start month for the SST and SLP fields
	#_based on the input climdata_months and the specified lags
	sst_months = []
	slp_months = []
	index_months = []
	sst_start = climdata_months[0] - sst_lag
	sst_months.append(sst_start)
	slp_start = climdata_months[0] - slp_lag
	slp_months.append(slp_start)
	index_start = climdata_months[0] - index_lag
	index_months.append(index_start)

	#_The for loops then populate the rest of the sst(slp)_months based n_mon_sst(slp)
	for i in range(1, n_mon_sst):
		sst_months.append(sst_start + i)
	for i in range(1, n_mon_slp):
		slp_months.append(slp_start + i)
	for i in range(1, n_mon_index):
		index_months.append(index_start + i)

	assert sst_months[0] >= -8, 'sst_lag set too high, only goes to -8'
	assert slp_months[0] >= -8, 'slp_lag set too high, only goes to -8'
	assert index_months[0] >= -8, 'index_lag set too high, only goes to -8'

	#_Next block of code checks start years and end years and sets appropriately.
	#_So hacky..
	#########################################################
	#########################################################
	if climdata_months[-1] <= 12:
		climdata_endyr = climdata_startyr + n_yrs - 1
		if sst_months[0] < 1 and sst_months[-1] < 1:
			sst_startyr = climdata_startyr - 1
			sst_endyr = climdata_endyr - 1
		elif sst_months[0] < 1 and sst_months[-1] >= 1:
			sst_startyr = climdata_startyr - 1
			sst_endyr = climdata_endyr
		elif sst_months[0] >=1 and sst_months[-1] >= 1:
			sst_startyr = climdata_startyr
			sst_endyr = climdata_endyr
	elif climdata_months[-1] > 12:
		climdata_endyr = climdata_startyr + n_yrs
		if sst_months[0] < 1 and sst_months[-1] < 1:
			sst_startyr = climdata_startyr - 1
			sst_endyr = climdata_endyr - 2
		elif sst_months[0] < 1 and sst_months[-1] >= 1:
			sst_startyr = climdata_startyr - 1
			sst_endyr = climdata_endyr - 1
		elif sst_months[0] >=1 and 1 <= sst_months[-1] <= 12:
			sst_startyr = climdata_startyr
			sst_endyr = climdata_endyr - 1
		elif sst_months[0] >=1 and sst_months[-1] > 12:
			sst_startyr = climdata_startyr
			sst_endyr = climdata_endyr
	if climdata_months[-1] <= 12:
		climdata_endyr = climdata_startyr + n_yrs - 1
		if index_months[0] < 1 and index_months[-1] < 1:
			index_startyr = climdata_startyr - 1
			index_endyr = climdata_endyr - 1
		elif index_months[0] < 1 and index_months[-1] >= 1:
			index_startyr = climdata_startyr - 1
			index_endyr = climdata_endyr
		elif index_months[0] >=1 and index_months[-1] >= 1:
			index_startyr = climdata_startyr
			index_endyr = climdata_endyr
	elif climdata_months[-1] > 12:
		climdata_endyr = climdata_startyr + n_yrs
		if index_months[0] < 1 and index_months[-1] < 1:
			index_startyr = climdata_startyr - 1
			index_endyr = climdata_endyr - 2
		elif index_months[0] < 1 and index_months[-1] >= 1:
			index_startyr = climdata_startyr - 1
			index_endyr = climdata_endyr - 1
		elif index_months[0] >=1 and 1 <= index_months[-1] <= 12:
			index_startyr = climdata_startyr
			index_endyr = climdata_endyr - 1
		elif index_months[0] >=1 and index_months[-1] > 12:
			index_startyr = climdata_startyr
			index_endyr = climdata_endyr
	if climdata_months[-1] <= 12:
		climdata_endyr = climdata_startyr + n_yrs - 1
		if slp_months[0] < 1 and slp_months[-1] < 1:
			slp_startyr = climdata_startyr - 1
			slp_endyr = climdata_endyr - 1
		elif slp_months[0] < 1 and slp_months[-1] >= 1:
			slp_startyr = climdata_startyr - 1
			slp_endyr = climdata_endyr
		elif slp_months[0] >=1 and slp_months[-1] >= 1:
			slp_startyr = climdata_startyr
			slp_endyr = climdata_endyr
	elif climdata_months[-1] > 12:
		climdata_endyr = climdata_startyr + n_yrs
		if slp_months[0] < 1 and slp_months[-1] < 1:
			slp_startyr = climdata_startyr - 1
			slp_endyr = climdata_endyr - 2
		elif slp_months[0] < 1 and slp_months[-1] >= 1:
			slp_startyr = climdata_startyr - 1
			slp_endyr = climdata_endyr - 1
		elif slp_months[0] >=1 and 1 <= slp_months[-1] <= 12:
			slp_startyr = climdata_startyr
			slp_endyr = climdata_endyr - 1
		elif slp_months[0] >=1 and slp_months[-1] > 12:
			slp_startyr = climdata_startyr
			slp_endyr = climdata_endyr
	#########################################################
	#########################################################

	if debug:
		from utils import int_to_month
		i2m = int_to_month()
		print('Precip starts in %s-%d, ends in %s-%d' % \
			(i2m[climdata_months[0]], climdata_startyr, i2m[climdata_months[-1]], climdata_endyr))
		print(f'{var} starts in %s-%d, ends in %s-%d' % \
			(i2m[sst_months[0]], sst_startyr, i2m[sst_months[-1]], sst_endyr))
		#print('SLP starts in %s-%d, ends in %s-%d' % \
			#(i2m[slp_months[0]], slp_startyr, i2m[slp_months[-1]], slp_endyr))
		print('INDEX starts in %s-%d, ends in %s-%d' % \
			(i2m[index_months[0]], index_startyr, i2m[index_months[-1]], index_endyr))

	#_Create function output
	kwgroups = {
		'climdata'	: {	'fp'		: climdata_fp,
						'startyr'	: climdata_startyr,
						'endyr'		: climdata_endyr,
						'months'	: climdata_months,
						'n_year'	: n_yrs
						},

		'sst'		: {	'n_mon'		: n_mon_sst,
						'months'	: sst_months,
						'startyr'	: sst_startyr,
						'endyr'		: sst_endyr
						},

		'slp'		: {	'n_mon'		: n_mon_slp,
						'months'	: slp_months,
						'startyr'	: slp_startyr,
						'endyr'		: slp_endyr,
						'n_year'	: n_yrs
						},
		'index'		: {	'n_mon'		: n_mon_index,
						'months'	: index_months,
						'startyr'	: index_startyr,
						'endyr'		: index_endyr,
						'n_year'	: n_yrs,
						'fp'		: index_fp,
						'n_phases'	: n_phases,
                        'phases_even': phases_even
						}
				}



	return kwgroups




