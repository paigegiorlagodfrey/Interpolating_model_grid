from BDNYCdb import BDdb, utilities as u
import modules as m
from matplotlib import pyplot as plt
import numpy as np
from pylab import *
rcParams['figure.figsize'] = 10,8
import mcmc_fit as mc
from scipy.optimize import curve_fit
from scipy.stats import mode
import logging, cPickle, SEDfit.synth_fit, itertools, astropy.units as q, numpy as np, matplotlib.pyplot as plt, pandas as pd

 
def make_model_db(model_grid, param_lims, rebin_models=True, use_pandas=False,model_atmosphere_db='/Users/paigegiorla/Code/Python/BDNYC/model_atmospheres.db'): 
    
  # Load the model atmospheres into a data frame and define the parameters
  models = pd.DataFrame(model_grid)
  params = [p for p in models.columns.values.tolist() if p in ['teff','logg','f_sed','k_zz']]
  
  # Get the upper bound, lower bound, and increment of the parameters
  plims = {p[0]:p[1:] for p in param_lims} if param_lims else {}
  for p in params:
    if p not in plims: 
      plims[p] = (min(models.loc[:,p]),max(models.loc[:,p]),max(np.diff(np.unique(np.asarray(models.loc[:,p]))), key=list(np.diff(np.unique(np.asarray(models.loc[:,p])))).count))
  
  # Choose template wavelength array to rebin all other spectra
  W = rebin_models if isinstance(rebin_models,(list,np.ndarray)) else models['wavelength'][0]
  
  # Rebin model spectra
  models['flux'] = pd.Series([u.rebin_spec([w*q.um, f*q.erg/q.s/q.cm**2/q.AA], W*q.um)[1].value for w,f in zip(list(models['wavelength']),list(models['flux']))])
  models['wavelength'] = pd.Series([W]*len(models['flux']))
  
  # Get the coordinates in parameter space of each existing grid point
  coords = models.loc[:,params].values
  
  # Get the coordinates in parameter space of each desired grid point
  template = np.asarray(list(itertools.product(*[np.arange(l[0],l[1]+l[2],l[2]) for p,l in plims.items()])))

  # Find the holes in the grid based on the defined grid resolution without expanding the grid borders
  def find_holes(coords, template=''):
    # Make a grid of all the parameters
    coords = np.asanyarray(coords)
    uniq, labels = zip(*[np.unique(c, return_inverse=True) for c in coords.T])
    grid = np.zeros(map(len, uniq), bool)
    # if template!='':
    #   temp = np.asanyarray(template)
    #   uniqT, labelsT = zip(*[np.unique(c, return_inverse=True) for c in temp.T])
    #   gridT = np.zeros(map(len, uniqT), bool)
    grid[labels] = True
    candidates = np.zeros_like(grid)
    
    # Test if there are neighboring models for interpolation
    for dim in range(grid.ndim):
      grid0 = np.rollaxis(grid, dim)
      inside = np.logical_or.accumulate(grid0, axis=0) & np.logical_or.accumulate(grid0[::-1], axis=0)[::-1]
      candidates |= np.rollaxis(inside, 0, dim+1)
    holes = candidates & ~grid
    hole_labels = np.where(holes)
    return np.column_stack([u[h] for u, h in zip(uniq, hole_labels)])
  
  grid_holes = find_holes(coords, template=template)
  
  # Interpolate the grid to fill in the holes
  for h in grid_holes:
    print 'Filling grid hole at {}'.format(h)
    new_spectrum = mc.pd_interp_models(params, h, models, smoothing=False)
    new_row = {k:v for k,v in zip(params,h)}
    new_row.update({'wavelength':new_spectrum[0], 'flux':new_spectrum[1], 'comments':'interpolated'})
    new_row.update({'wavelength':new_spectrum[0], 'flux':new_spectrum[1], 'comments':'interpolated', 'metallicity':0, 'id':None})
    models = models.append(new_row, ignore_index=True)
    
  # Sort the DataFrame by teff and logg?
  models.sort(list(reversed(params)), inplace=True)

  # Turn Pandas DataFrame into a dictionary of arrays if not using Pandas
  if not use_pandas:
    M = {k:models[k].values for k in models.columns.values}
    M['flux'] = q.erg/q.AA/q.cm**2/q.s*np.asarray(M['flux'])
    M['wavelength'] = q.um*M['wavelength'][0]
    return M

  else: return models
  
def model_grid_interp_test(mg,param_lims):
	'''
	Perform a test of the model grid interpolation by specifying the teff, logg and grid resolution of the given model grid.
  
	Parameters
	----------
	model_grid: dict
		The model grid object that results from running make_model_db(). Full and complete.
	teff: tuple, list
		A sequence of the teff value and increment over which to interpolate, e.g. (1200,100) tests the model at 1200K by interpolating between the 1100K and 1300K models
	teffs:
		a list of the wanted teff tuples, e.g. [(1200,100),(1300,100)]
	logg: tuple, list
		A sequence of the logg value and increment over which to interpolate, e.g. (4.5,0.5) tests the model at 4.5dex by interpolating between the 4.0dex and 5.0dex models
		
		old = [k[p][idx] for p in range(len(k))]
		oldies.append(old)
			
		L = M.values()

	for param in params:
		idx2 = zip(M['teff'],M['logg']).index((param[0],param[1]))
		
		newbies.append(new)
	Returns
	-------
	None
	'''
	# Load the model atmospheres into a data frame and define the parameters
	ranges = [np.arange(param_lims[0][1]+50,param_lims[0][2],param_lims[0][3]), np.arange(param_lims[1][1]+1 ,param_lims[1][2],param_lims[1][3])]
	params = list(itertools.product(*ranges))
	
 	interpd_models = []
 	d = mg.values()
 	
 	for param in params:
		# Find the indexes of the models with the appropriate parameters
		idx = zip(d[0],d[1]).index((param[0],param[1]))
		# Pop the true models out of the grid
		k=[np.delete(r,idx) for r in d]
		waves = [mg['wavelength']]*len(k[1])
		m = {'id':k[6],'metallicity':k[4],'teff':k[0],'logg':k[1],'wavelength':waves,'flux':k[3],'comments':k[2]}
		MG = make_model_db(m,param_lims)
		interpd_models.append([MG.values()[p][zip(MG['teff'],MG['logg']).index((param[0],param[1]))] for p in range(len(MG.values()))])
	teff,logg,comments, flux, wavelength = [],[],[],[],[]
	for i in interpd_models:
		teff.append(i[0])
		logg.append(i[1])
		comments.append(i[2])
		flux.append(i[3])
		wavelength.append(i[4])
	img = {'teff':teff,'logg':logg,'wavelength':wavelength,'flux':flux,'comments':comments}
	return interpd_models, img, mg, params

def plot_residuals(mg, interpd_models, params):	
	l=mg.values()
	wave = l[5].value

	for i in range(len(params)):
		f = [l[p][zip(mg['teff'],mg['logg']).index((params[i][0],params[i][1]))] for p in range(len(l))][3]
		f_up = [l[p][zip(mg['teff'],mg['logg']).index((params[i][0]+50,params[i][1]))] for p in range(len(l))][3]
		f_dn = [l[p][zip(mg['teff'],mg['logg']).index((params[i][0]-50,params[i][1]))] for p in range(len(l))][3]
		F = interpd_models[i][3]
	
		[w,f,unc] = m.wavelength_band('all',[wave,f,np.ones(len(f))])
		[w_up,f_up,unc] = m.wavelength_band('all',[wave,f_up,np.ones(len(f_up))])
		[w_dn,f_dn,unc] = m.wavelength_band('all',[wave,f_dn,np.ones(len(f_dn))])		
		[W,F,U] = m.wavelength_band('all',[wave,F,np.ones(len(F))])
	
		difference = ((np.array(f)-np.array(F))/np.array(f))*100
		f = u.smooth(list(f),1)
		f_up = u.smooth(list(f_up),1)
		f_dn = u.smooth(list(f_dn),1)
		F = u.smooth(list(F),1)
	
		fig1 = figure(1)
		frame1=fig1.add_axes((.1,.3,.8,.6))
		plt.scatter(w,f,s=0.5,c='b',edgecolor='b',label='old')
		plt.scatter(W,F,s=0.5,c='g',edgecolor='g',label='interp-d')
		plt.scatter(w_up,f_up,s=0.5,c='y',edgecolor='y',label='upper')
		plt.scatter(w_dn,f_dn,s=0.5,c='lime',edgecolor='lime',label='lower')		
		plt.legend()
		plt.ylabel('Flux')
		frame1.set_xticklabels([]) #Remove x-tic labels for the first frame
		grid()

		frame2=fig1.add_axes((.1,.1,.8,.2))        
		plt.scatter(w,difference, s=1, c='r',edgecolor='r')
# 		plt.ylim(-100,100)
		grid()
		plt.xlabel('Wavelength ($mu$m)')
		plt.ylabel('Percent Difference')
		plt.savefig('/Users/paigegiorla/Code/Python/BDNYC/interpolation/residuals/trimmed_{}_{}'.format(params[i][0],params[i][1])+'.png')
		clf()		
		
def plot_comparison(mg, interpd_models, params):	

	divdiff,teffs,loggs = [],[],[]
	l=mg.values()
	wave = l[5].value

	for i in range(len(params)):
		f = [l[p][zip(mg['teff'],mg['logg']).index((params[i][0],params[i][1]))] for p in range(len(l))][3]
		F = interpd_models[i][3]
		
		[w,f,unc] = m.wavelength_band('all',[wave,f,np.ones(len(f))])
		[W,F,U] = m.wavelength_band('all',[wave,F,np.ones(len(F))])

		n = (abs(np.array(f)-np.array(F))/np.array(f))*100
		
		divdiff.append(list(mode(n)[0])[0])
		teffs.append(params[i][0])
		loggs.append(params[i][1])
	cmap=cm.get_cmap('RdPu',7)
	labels = np.arange(2.5,6.0,0.5)	
	for i in range(len(teffs)):
		rand=np.random.random()
		foo = plt.scatter(teffs[i],divdiff[i]+rand/10, c=loggs[i],cmap=cmap, vmin=2.5, vmax=5.5)
	plt.colorbar(foo,ticks=labels)
 	plt.xlim(430,1600)
#  	plt.ylim(min(divdiff),max(divdiff))
 	plt.yscale('log')
 	plt.xlabel('$T_{eff}$',fontsize = 'large')
	plt.ylabel('Mode of Percent Difference (log scale)',fontsize = 'large')
	plt.savefig('/Users/paigegiorla/Code/Python/BDNYC/interpolation/goodness_test_mcmc_mode.png')
	plt.clf()

def plot_diff_wavelength(mg, MG, params):
	
	for i in range(len(oldies)):
		f=oldies[i][3]
		F=newbies[i][3]

		f=m.wavelength_band()
		