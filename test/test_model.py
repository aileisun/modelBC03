import pytest
import numpy as np
import astropy.table as at
import bubbleimg
from .. import modelBC03


# dir_model = '/Users/aisun/Documents/astro/projects/feedback/survey/hsc/sandbox/sps/sklearnlinear/spectra_bc03_downgraded/'

@pytest.fixture
def get_model():
	m = modelBC03(extinction_law='none')
	return m


def test_modelBC03_init(get_model):

	m = get_model

	temps = m.temps

	# check that the length is correct
	assert temps.shape[0] == len(m.pars)
	assert temps.shape[1] == len(m.ws)

	# check that the content is correct on one of the templates (randomly chosen)
	i = np.random.randint(m.n_temp)
	mettag = m.pars['mettag'][i]
	agetag = m.pars['agetag'][i]
	data = np.genfromtxt(m.get_fp_temp(mettag, agetag)).T
	temp_from_file = data[1][data[0]<m.w_max]
	spec = data[1]/np.mean(data[1])
	temp_from_file = spec[data[0]<m.w_max]
	assert np.all(temp_from_file == temps[i])


def test_modelBC03_init_wmax():
	w_max = 20000.
	m = modelBC03(w_max=w_max)
	assert np.max(m.ws) < w_max
	assert m.temps.shape[1] == len(m.ws)


def test_modelBC03_regrid(get_model):

	m = get_model

	dir_obj = '/Users/aisun/Documents/astro/projects/feedback/survey/hsc/sandbox/sps/data/SDSSJ084240+005347/'
	obj = bubbleimg.obsobj.obsObj(ra=130.6686971883207, dec=0.8966510710968996, dir_obj=dir_obj, obj_naming_sys='sdss_precise')

	s = bubbleimg.spector.Spector(obj=obj, survey='hsc')

	spec, ws = s.get_spec_ws()
	m.regrid(ws, s.z)

	assert m.temps_regrid.shape[0] == len(m.pars)
	assert m.temps_regrid.shape[1] == len(m.ws_regrid)

	assert np.all(m.ws_regrid == ws)


def test_modelBC03_fit(get_model):

	m = get_model
	
	dir_obj = '/Users/aisun/Documents/astro/projects/feedback/survey/hsc/sandbox/sps/data/SDSSJ084240+005347/'
	obj = bubbleimg.obsobj.obsObj(ra=130.6686971883207, dec=0.8966510710968996, dir_obj=dir_obj, obj_naming_sys='sdss_precise')

	s = bubbleimg.spector.Spector(obj=obj, survey='hsc')

	spec, ws = s.get_spec_ws()
	m.fit(ws, spec, z=s.z)

	assert len(m.bestfit) == m.n_wave
	assert len(m.bestfit_regrid) == len(s.ws)
	assert len(m.coef) == m.n_temp

	ws_topredict = ws[:10]
	spec_predict = m.predict(ws_topredict)
	assert len(spec_predict) == len(ws_topredict)


def test_modelBC03_fit_extinction():

	m = modelBC03(extinction_law='linear')
	# m = modelBC03(extinction_law='calzetti00')


	dir_obj = '/Users/aisun/Documents/astro/projects/feedback/survey/hsc/sandbox/sps/data/SDSSJ084240+005347/'
	s = bubbleimg.spector.Spector(dir_obj=dir_obj, survey='hsc')

	m._fit_extinction(s.ws, s.spec, z=s.z)
	assert m.extinction_params[0] > 0. # a_v
	assert m.extinction_params[1] > 0. # r_v

	temp_ext = m._apply_extinction(m.ws, m.temps[0])
	assert np.all(temp_ext.shape == m.temps[0].shape)

	m.fit(s.ws, s.spec, z=s.z)

	assert len(m.bestfit) == m.n_wave
	assert len(m.bestfit_regrid) == len(s.ws)
	assert len(m.coef) == m.n_temp
