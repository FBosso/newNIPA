"""
Module containing utility functions for running NIPA.
"""
from matplotlib import cm, pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np

def sstMap(nipaPhase,  cmap = cm.jet, fig = None, ax = None):
	from mpl_toolkits.basemap import Basemap
	if fig == None:
		fig = plt.figure()
		ax = fig.add_subplot(111)
	m = Basemap(ax = ax, projection = 'cyl', lon_0 = 270, resolution = 'i')
	m.drawmapboundary(fill_color='#ffffff',linewidth = 0.15)
	m.drawcoastlines(linewidth = 0.15)
	#m.fillcontinents(color='#eeeeee',lake_color='#ffffff')
	parallels = np.linspace(m.llcrnrlat, m.urcrnrlat, 4)
	meridians = np.linspace(m.llcrnrlon, m.urcrnrlon, 4)
	m.drawparallels(parallels, linewidth = 0.3, labels = [0,0,0,0])
	m.drawmeridians(meridians, linewidth = 0.3, labels = [0,0,0,0])

	lons = nipaPhase.sst.lon
	lats = nipaPhase.sst.lat

	data = nipaPhase.corr_grid
	levels = np.linspace(-1.0,1.0,41)

	lons, lats = np.meshgrid(lons,lats)

	im1 = m.pcolormesh(lons,lats,data, vmin = np.min(levels), \
	vmax=np.max(levels), cmap = cmap, latlon=True)
	divider = make_axes_locatable(ax)
	cax = divider.append_axes('bottom', size='5%', pad=0.05)
	fig.colorbar(im1, cax=cax, orientation='horizontal')
	return fig, ax, m 


def int_to_month():
	"""
	This function is used by data_load.create_data_parameters
	"""
	i2m = {
	-8	: 'Apr',
	-7	: 'May',
	-6	: 'Jun',
	-5	: 'Jul',
	-4	: 'Aug',
	-3	: 'Sep',
	-2	: 'Oct',
	-1	: 'Nov',
	0	: 'Dec',
	1	: 'Jan',
	2	: 'Feb',
	3	: 'Mar',
	4	: 'Apr',
	5	: 'May',
	6	: 'Jun',
	7 	: 'Jul',
	8	: 'Aug',
	9	: 'Sept',
	10	: 'Oct',
	11	: 'Nov',
	12	: 'Dec',
	13	: 'Jan',
	14	: 'Feb',
	15	: 'Mar',
		}
	return i2m


def slp_tf():
	d = {
		-4	: '08',
		-3	: '09',
		-2	: '10',
		-1	: '11',
		0	: '12',
		1	: '01',
		2	: '02',
		3	: '03',
		4	: '04',
		5	: '05',
		6 	: '06',
		7	: '07',
		8	: '08',
		9	: '09',
		10	: '10',
		11	: '11',
		12	: '12',
		13	: '01',
		14	: '02',
		15	: '03',
		}
	return d


def vcorr(X, y):
    # Function to correlate a single time series with a gridded data field
    # X - Gridded data, 3 dimensions (ntim, nlat, nlon)
    # Y - Time series, 1 dimension (ntim)

	ntim, nlat, nlon = X.shape
	ngrid = nlat * nlon

	y = y.reshape(1, ntim)
	X = X.reshape(ntim, ngrid).T
	Xm = X.mean(axis = 1).reshape(ngrid,1)
	ym = y.mean()
	r_num = np.sum((X-Xm) * (y-ym), axis = 1)
	r_den = np.sqrt(np.sum((X-Xm)**2, axis = 1) * np.sum((y-ym)**2))
	r = (r_num/r_den).reshape(nlat, nlon)

	return r

def sig_test(r, n, twotailed = True):
	import numpy as np
	from scipy.stats import t as tdist
	df = n - 2

	# Create t-statistic
	# Use absolute value to be able to deal with negative scores
	t = np.abs(r * np.sqrt(df/(1-r**2)))
	p = (1 - tdist.cdf(t,df))
	if twotailed:
		p = p * 2
	return p


def weightsst(sst):
    # SST needs to be downloaded using the openDAPsst function
	from numpy import cos, radians
	weights = cos(radians(sst.lat))
	for i, weight in enumerate(weights):
		sst.data[:,i,:] *= weight
        # se vogliamo il dataset SST meno la media dobbiamo sommare i valori 
        # ottenuti dalla moltiplicazione della SST con i relativi pesi (in modo 
        # da ottenere il valore della media pesata ... è solo necessaria la 
        # somma in quanto i pesi sommano già ad 1, quindi non serve dividere)
	return sst

