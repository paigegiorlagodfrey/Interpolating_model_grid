from matplotlib import pyplot as plt
import pickle
from BDNYCdb import BDdb, utilities as u
ma_db = BDdb.get_db('/Users/paigegiorla/Code/Python/BDNYC/model_atmospheres.db')
from pylab import *
from matplotlib.colors import ListedColormap
import numpy as np
import modules as m
import itertools
cmap=cm.get_cmap('cool')
rc('axes',color_cycle=[cmap(i) for i in np.linspace(0,0.9,30)])

def showme(model_grid_name, comparison_filepath, g_param,tmin,tmax,scale=None):
	res=np.arange(0.65794802,2.5604601,0.00033973429884229386)
	mg = ma_db.dict("SELECT * from {} where logg={} and teff between {} and {}".format(model_grid_name,g_param,tmin,tmax)).fetchall()
	fb = open('{}'.format(comparison_filepath),'rb')
	img = pickle.load(fb)
	fb.close()
	
	f, (ax1,ax2) = plt.subplots(2, 1)
	
# 	ax1.set_color_cycle([cmap(i) for i in np.linspace(0, 0.9, len(mg))])
	print len(mg)
	for i in range(len(mg)):
		f = mg[i]['flux']
		w = mg[i]['wavelength']	
		unc = np.ones(len(f))	
		[w,f,unc] = u.rebin_spec([w,f,unc],res)
		f = u.smooth(f,1)
		ax1.plot(w,f)
	ax1.set_xlim(0.9,2.3)	
	ax1.set_title('Original')
	if scale == 'log':
		ax1.set_yscale("log")	
	ax1.axis('tight')

	n=1
	for i in range(len(img['logg'])):
		w = img['wavelength']		
		if img['logg'][i]==g_param:
			f = img['flux'][i]
			unc = np.ones(len(f))	
			[w,f,unc] = u.rebin_spec([w,f,unc],res)
			f_smooth = u.smooth(f,1)
			ax2.plot(w,f_smooth)
			n+=1
	print n		
	ax2.set_title('Recalculated')
	if scale == 'log':
		ax2.set_yscale("log")	
	ax2.axis('tight')

	plt.savefig('/Users/paigegiorla/Code/Python/BDNYC/interpolation/compare_grid_versions_{}_{}'.format(model_grid_name,scale)+'.png')
	plt.clf()