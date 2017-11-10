# modelBC03

modelBC03 is a python module to fit Brussel and Charlotte stellar population synthesis model to spectrum. 

```
import modelBC03

m = modelBC03.modelBC03()

m.fit(wavelengths, spectrum)

m.predict(wavelengths2)
```

# fitting

It uses generalized linear models from sklearn -- either Lasso or Ridge regression. It is recommanded to use the Lasso regression option, which finds a linear combination of a small number of templates to fit the data. It is reasonably fast. 

# extinction

modelBC03 can incorporate extinction but it is not well tested. There are three options: 'none', 'linear', 'calzetti00'. To determine the amount of extinction, it first tries to fit only the high frequency (absorption line) part of the spectrum to predict a dereddened spectrum. It takes the ratio between the dereddenedd and the original spectrum and fits a extinction curve to it. This best fit extinction curve is then applied to the templates. 

# emission lines

It does not handle emission lines, please mask them before fitting. 


# to link to model files
Please set the directory path to the BC03 model data or use the environment variable MODELBC03_DIR. For example, in ~/.bashrc file, add line:

    export MODELBC03_DIR='path/spectra_bc03_downgraded/'


## installing

The recommended method is to clone or download the source code and add the path to your .bashrc.. 

~~~~
export "PYTHONPATH=$PYTHONPATH:/where/you/put/the/code/" 
~~~~
