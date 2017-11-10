import os
import astropy.table as at
import numpy as np
import scipy.interpolate as sintrp
# import sklearn
from sklearn import linear_model
import scipy.signal as ss
import scipy.optimize as so
import extinction

class modelBC03(object):
	def __init__(self, **kwargs):
		""" 
		modelBC03
	
		Please set the directory path to the BC03 model data or use the environment variable MODELBC03_DIR. For example, in ~/.bashrc file, add line:

		export MODELBC03_DIR='path_to_model/'

		The fitting does not handle emission lines, one needs to mask out emission lines before putting the spectrum in. One can choose to fit extinction law using the calzetti00 law. No spectral resolution adjustment is made. Fitted age or metallicity is not reliable, this class is only to be used for extrapolation of continuum spectrum. 

		Params
		------
		directory (str): 
			path to the director containing BC03 templates. If it is not supplied then take value from environmental variable. 

		w_max = 15000 (float):
			max rest frame wavelength used in the template in unit of Angstrom

		extinction_law = 'none' (str):
			either 'linear' or 'calzetti00' or 'none'
			extinction_law will be applied unless its set to 'none'

		Attributes
		----------

		n_age (int)
		n_met (int)
		n_temp (int)
		n_ws (int)

		temps
		ws

		(optional)
		temps_regrid
		ws_regrid 
		"""

		self.directory = kwargs.pop('directory', os.environ.get('MODELBC03_DIR'))
		if self.directory is None:
			raise Exception("Please provide directory path to the BC03 model files. ")

		self.w_max = kwargs.pop('w_max', 15000.)
		self.extinction_law = kwargs.pop('extinction_law', 'none')

		# initialize extinction_params
		if self.extinction_law == 'linear':
			self.extinction_params = (0., 0.) # slope, intercept
		elif self.extinction_law == 'calzetti00':
			self.extinction_params = (0.1, 0.1, 1.) # a_v, r_v, scaling
		elif self.extinction_law != 'none': 
			raise Exception("extinction_law not recognized")

		self.__init_param()
		self.__init_temps()


	def fit(self, ws, spec, z=0.): 
		""" 
		fit linear model to spectrum

		Params
		------
		ws (arr)
		spec (arr)
		z=0. (float)

		Return
		------
		None

		Set Attr
		--------
		regression (linear_model regression object)
		coef (best fit coefficients)
		intercept (best fit intercept)
		bestfit (best fit model in the grid of the original model self.ws but redshifted)
		ws_bestfit (redshifted wavelengths)
		bestfit_regrid (best fit model in the grid of the input data self.ws_regrid)
		"""
		if not np.all(ws == self.ws):
			self.regrid(ws=ws, z=z)

		if self.extinction_law == 'none':
			temps = self.temps
			temps_regrid = self.temps_regrid
		else:
			self._fit_extinction(ws=ws, spec=spec, z=z)
			temps = self._apply_extinction(self.ws*(1.+z), self.temps)
			temps_regrid = self._apply_extinction(self.ws_regrid, self.temps_regrid)
			self.temps_ext = temps
			self.temps_regrid_ext = temps_regrid

		reg, bestfit_regrid = linear_reg(data=spec, temps=temps_regrid, type='lasso', alpha=1.e-6, positive=True)

		self.regression = reg
		self.coef = reg.coef_
		self.intercept = reg.intercept_

		self.bestfit = np.dot(reg.coef_, temps) + reg.intercept_
		self.ws_bestfit = self.ws*(1.+z)
		self.bestfit_regrid = bestfit_regrid


	def predict(self, ws):
		"""
		Return the predicted bestfit spectrum at the observed wavelengths ws as an array of the same size as ws
		"""
		f = sintrp.interp1d(self.ws_bestfit, self.bestfit, kind='linear')
		return f(ws)


	def _fit_extinction(self, ws, spec, z=0.):
		"""
		Get the extinciton law of spectrum. Currently using calzetti00 law parametrized by a_v, r_v. 

		Params
		------
		self
		ws (array)
		spec (array)
		z=0. (float)

		Set Attr
		--------
		self.reddening_ratio (ratio of bestfit unreddened over data on ws_regrid grid)
		self.extinction_params (depending on the extinction_law)
		"""
		if not np.all(ws == self.ws):
			self.regrid(ws=ws, z=z)

		spec_norm = spec/np.mean(spec)
		# highpass filter both the spec and the temp
		kernel_size=299
		spec_highpass = spec_norm - ss.medfilt(spec_norm, kernel_size=kernel_size)
		temps_regrid_highpass = [temp - ss.medfilt(temp, kernel_size=kernel_size) for temp in self.temps_regrid]
		spec_highpass = smooth(spec_highpass, n=10)
		temps_regrid_highpass = np.array([smooth(temp, n=10) for temp in temps_regrid_highpass])

		# find the bestfit template for highpass, which gives prediction on unreddened spec. 
		reg_highpass, __ = linear_reg(spec_highpass, temps_regrid_highpass, type='lasso', alpha=1.e-6, positive=True, max_iter=5000)
		predicted = np.dot(self.temps_regrid.T, reg_highpass.coef_) + reg_highpass.intercept_

		# get empirial reddening ratio
		spec_med = ss.medfilt(spec_norm, kernel_size=kernel_size)
		predict_med = ss.medfilt(predicted, kernel_size=kernel_size)

		self.reddening_ratio = spec_med/predict_med

		if self.extinction_law == 'linear':
			# sklearn RANSAC robust linear regression to find best fit params slope and intercept
			x = self.ws_regrid.reshape(-1, 1)
			y = self.reddening_ratio.reshape(-1, 1)
			reg_ransac = linear_model.RANSACRegressor()
			reg_ransac.fit(X=x, y=y)
			slope = reg_ransac.estimator_.coef_[0]
			intercept = reg_ransac.estimator_.intercept_
			self.extinction_params = (slope, intercept)

		elif self.extinction_law == 'calzetti00':
			# least square curve_fit to find bestfit params a_v, r_v, scaling
			popt, __ = so.curve_fit(self.extinction_curve, xdata=self.ws_regrid, ydata=self.reddening_ratio, p0=self.extinction_params) 
			self.extinction_params = popt
		else:
			raise Exception("extinction_law not recognized")


	def extinction_curve(self, ws, *params):
		"""
		return the reddening curve given wavlengths and params. 
		the type of extinction curve is determined by attribute self.extinction_law. 

		For example, for 'calzetti00' the params are a_v, r_v, and scaling.  
		For example, for 'linear' the params are slope and intercept.  

		Params
		------
		self
		ws (array)
		*params
		"""
		if self.extinction_law == 'linear':
			slope, intercept = params
			return slope*ws + intercept

		elif self.extinction_law == 'calzetti00':
			a_v, r_v, scaling = params
			flux = np.ones(len(ws))
			extlaw = extinction.calzetti00(ws.astype('double'), a_v=a_v, r_v=r_v)
			return scaling*extinction.apply(extlaw, flux)


	def _apply_extinction(self, ws, data):
		"""
		apply fitted extinction curve to input

		Params
		------
		self
		ws (array)
		data (array)

		Return
		------
		extincted (array)
			extinction applied to data
		"""
		return data * self.extinction_curve(ws, *self.extinction_params)


	def regrid(self, ws, z=0.):
		""" 
		creating new attributes temps_regrid and ws_regrid that is regridded onto a new wavelength grids given by the parameter ws. Optionally, one can set the redshift z to be non-zero to have templates matched to the one of the object. 

		Params
		------
		ws (arr)
		z=0. (float)

		Return
		------
		None

		New Attr
		--------
		temps_regrid
		ws_regrid
		"""
		temps_regrid = regrid_temps(ws=self.ws, temps=self.temps, ws_to=ws, z=z)
		self.z = z
		self.temps_regrid = temps_regrid
		self.ws_regrid = ws


	def __init_temps(self):
		# set attribute temps as an array of size (number of templates, number of spexels)
		# the templates are normalized such that the mean is 1.

		fn = self.get_fp_temp(self.mettab['mettag'][0], self.agetab['agetag'][0])
		data = np.genfromtxt(fn).T

		ws = data[0]
		sel = [ws < self.w_max]

		self.ws = ws[sel]
		self.n_wave = len(self.ws)

		temps = np.empty([self.n_temp, self.n_wave])
		for im, mettag in enumerate(self.mettab['mettag']):
			for ia, agetag in enumerate(self.agetab['agetag']):
				fn = self.get_fp_temp(mettag, agetag)
				data = np.genfromtxt(fn).T
				temp = data[1]/np.mean(data[1])
				temps[im*self.n_age+ia] = temp[sel]
		self.temps = temps


	def __init_param(self):
		""" 
		assign attributes for the template params:
		n_age (int)
		n_met (int)
		n_temp (int)

		agetab
				age  agetag
				float	int
				---- ------
				 0.1    0.1
				 0.2    0.2

		mettab (like agetab)

		pars 
				number metallicity age  mettag agetag
				------ ----------- ---- ------ ------
				     0        0.02  0.1    002    0.1
				     1        0.02  0.2    002    0.2
				     2        0.02  0.3    002    0.3
				     3        0.02  0.4    002    0.4
				     4        0.02  0.5    002    0.5
		"""
		agetags = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.05', '0.7', '0.8', '0.08', '0.9', '0.55', '0.65', '1.5', '1', '2.5', '2', '3.5', '3', '4.5', '4', '5.5', '5', '6.5', '6', '7.5', '7', '8.5', '8', '9.5', '9', '10.5', '10', '11.5', '11', '12.5', '12', '13.5', '13', '14',]
		ages = np.array(agetags).astype(float)
		self.agetab = at.Table([ages, agetags], names=['age', 'agetag'])
		self.n_age = len(self.agetab)

		mettags = ['002', '0004', '005', '0008']
		mets = [0.02, 0.004, 0.05, 0.008]
		self.mettab = at.Table([mets, mettags], names=['met', 'mettag'])
		self.n_met = len(self.mettab)

		# set up meshgrid par file
		self.n_temp = self.n_met * self.n_age
		a, m = np.meshgrid(ages, mets)
		atag, mtag = np.meshgrid(agetags, mettags)
		self.pars = at.Table([np.arange(self.n_temp), m.flatten(), a.flatten(), mtag.flatten(), atag.flatten()], names=['number', 'metallicity', 'age', 'mettag', 'agetag'])


	def get_fp_temp(self, mettag, agetag):
		return self.directory+'spectrum_BC03_stelib_chab_Z{mettag}_age{agetag}.dat'.format(mettag=mettag, agetag=agetag)


def linear_reg(data, temps, type='lasso', alpha=1.e-3, positive=True, max_iter=5000): 
	"""
	fit temps to data using general linear regression. 

	Params
	------
	data (1d np array)
	temps (2d np array) (n_temp * n_datapoint)
	type = 'ridge' (str):
		ridge or lasso regression
	alpha=0.001 (float)
		weighting of the regularization term

	Return
	------
	reg (linear_model.Ridge/ or Lasso instance)
		with attributes coef_, intercept_
	bestfit (1d np array of length n_datapoint)
		best fit model to data on the same grid as data
	"""
	if type == 'lasso':
		reg = linear_model.Lasso(alpha=alpha, positive=positive, max_iter=max_iter, fit_intercept=False)
	elif type == 'ridge':
		reg = linear_model.Ridge(alpha=alpha, max_iter=max_iter, fit_intercept=False)
	else:
		raise Exception("regression type not recognized")

	reg.fit(temps.T, data)

	bestfit = np.dot(reg.coef_, temps) + reg.intercept_

	return reg, bestfit



def butter_highpass_param(lowcut=0.001, fs=1., order=3):
	""" 
	Create params to make Butterworth filter. The cut off is at a scale of (0.5 * fs)/lowcut in the unit of pixel. 

	lowcut [1/Angstrom]
	fs [pix per Angstrom]

	"""
	nyq = 0.5 * fs
	low = lowcut / nyq
	b, a = ss.butter(order, low, btype='highpass')
	return b, a


def butter_highpass_filter(data, lowcut=0.001, fs=1., order=3):
	""" 
	filter the data with butterwoth highpass filter. see butter_highpass_param for details. 
	"""
	b, a = butter_highpass_param(lowcut, fs, order=order)
	y = ss.lfilter(b, a, data)
	return y


def regrid_temps(ws, temps, ws_to, z=0.):
	n_temp = len(temps)
	temps_regrid = np.empty([n_temp, len(ws_to)])
	for i in range(n_temp):
		f = sintrp.interp1d(ws*(1.+z), temps[i], kind='linear')
		temp_intrp = f(ws_to)
		temps_regrid[i] = temp_intrp

	return temps_regrid


def smooth(data, n=10):
	return np.convolve(data, np.ones((n,))/n, mode='same')


