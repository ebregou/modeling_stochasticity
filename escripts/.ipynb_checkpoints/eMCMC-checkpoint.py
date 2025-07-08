# Purpose: interface with Zeus to run MCMC on models of the UVLF with time-evolving sigma_UV
# Author: Emily Bregou, Julian Muñoz

# Standard packages
import numpy as np
import emcee
import pandas as pd

# Local packages
import zeus21

print('dust is currently turned off at high z')

# MCMC class
class UVLF():
    def __init__(self, file_names, data_labels = None, param_data = None):
        self.sorted_data = read_data(file_names, data_labels)
        self.data = reduce_data(self.sorted_data)
        if type(self.data[0]) is not list: # Make sure the format is correct for the rest of the script if only one redshift is input
            self.data = [self.data] 
        self.zs = [dat[0] for dat in self.data]
        if param_data is None:
            param_data = get_default_dict(table=False)
        self.UserParams = zeus21.User_Parameters()
        self.param_data = param_data
        self.lowers = [p['lower'] for p in list(self.param_data.values()) if p.get('fit', True)]
        self.uppers = [p['upper'] for p in list(self.param_data.values()) if p.get('fit', True)]
        self.ndim = len(self.lowers)
        self.nwalkers = 2*self.ndim # Walkers = twice the number of parameters

        # Get cosmological parameters, construct HMF from Zeus
        CosmoParams_input = zeus21.Cosmo_Parameters_Input(zmin_CLASS=0.0)
        self.CosmoParams,ClassyCosmo, CorrFclass, self.HMFintclass =  zeus21.cosmo_wrapper(self.UserParams, CosmoParams_input)

        # Create MCMC sampler
        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_prob) 

    def generate_ICs(self):
        """
        Generate different ICs, run a short MCMC to spread them out a bit
        Returns: 
            ICs [array]: ICs to run MCMC with
        """
        step_size = [(upper-lower)/100 for upper, lower in zip(self.uppers, self.lowers)]

        # Start each walker at a different place
        params1 = [p['start'] for p in list(self.param_data.values()) if p.get('fit', True)]
        p0 = params1 + step_size * (np.random.randn(self.nwalkers, self.ndim))

        # Run a short MCMC to spread the walkers out a bit
        ICs = self.sampler.run_mcmc(p0, 100)
        
        return ICs

    def log_prob(self, paramvector): 
        """
        Calculate the log probability, taking into account the likelihood & the prior
        Inputs:
            paramvector [1darray]: values of parameters
        Outputs:
            log probability [float]
        """
        lprior=self.log_prior(paramvector)
        if (lprior > -np.inf): #only run if in prior range. avoid weird behavior for negative logs etc
            lpost=self.log_like(paramvector)
        else:
            lpost = 0.0 #doesn't matter, added to -inf
        #print(paramvector)
        return lprior + lpost

    
    def log_prior(self, paramvector):
        """
        Calculate the log prior for a flat prior: 0 if within range; negative infinity if outside
        Inputs:
            paramvector [1darray]: values of parameters
        Returns:
            prior [float]: (0 if within range; negative infinity if outside)
        """
        if all(lower < t < upper for t, lower, upper in zip(paramvector, self.lowers, self.uppers)):
            return 0.0  # log(1)
        return -np.inf  # log(0)
        
    def log_like(self, paramvector, alt_data = None):
        """
        Calculate log likelihood
        Inputs:
            paramvector [1darray]: values of parameters
            alt_data [Ndarray]: alternative data to input if you don't want to use self.data (in case you want to calculate log_like over a diff.
                                redshift range than you originally fit to)
        Returns:
            loglike_curr [float]: log likelihood of the data given the model parameters
        """
        
        #now bin it appropriately at each z -- data part
        loglike_curr = 0.0

        # self.data has all the z & UVLF data
        if alt_data is not None:
            data = alt_data
        else:
            data = self.data
            
        for dataarrayz in data: # Add the log likelihoods together for each redshift. The log likelihood is just a sum over all the points
            # anyways, so this makes sense
            
            zdat, zerr, xdat, ydat, yerr_upper, yerr_lower, xerr = decompose_data(dataarrayz)
    
            uvlftheory = self.UVLF_wrapper(zdat,zerr,xdat,xerr, paramvector)
            
            # This applies the correct sigma to the Gaussian distribution of the likelihood, based on whether the point returned is above or below
            # the mean, for when the datapoints have asymmetrical errorbars. 
            yerr_asymmetrical = np.array([yerr_upper[i] if uvlftheory[i] > ydat[i] else yerr_lower[i] for i in range(len(ydat))])

            # this formula is just proportional to the natural log of a Gaussian
            loglike_curr += -np.sum((ydat - uvlftheory)**2/(2.0 * yerr_asymmetrical**2))
        
        return loglike_curr

    def UVLF_wrapper(self, zcenter, zwidth, MUVcenters, MUVwidths, paramvector):
        """
        Computes and returns the UVLF at z=zcenters, with width zwidths, in bins centered at MUVcenters with width MUVwidths
        Inputs:
            zcenters [float]: center redshift value for binned UVLF data
            zwdith [float]: width of the redshift bin
            MUVcenters [1darray]: the central UV magnitude in each bin
            MUVwidths [1darray]: the width of the UV magnitude bins
            paramvector [1darray]: parameters
        Outputs:
            PhiUV [1darray]: In units of mag^-1 Mpc^-3
        """

        params = self.time_evolution(paramvector, zcenter)
        astroparams = self.param_wrapper(params)
        UVLFs_std = zeus21.UVLFs.UVLF_binned(astroparams,self.CosmoParams,self.HMFintclass,zcenter,zwidth,MUVcenters,MUVwidths)
        
        return UVLFs_std

    def time_evolution(self, paramvector, zcenter):
        """
        Applies the time evolution of each parameter so that we feed the evolved value, matching the given redshift, to the UVLF wrapper
        Inputs:
            paramvector [1darray]: parameters
            zcenter [float]: center redshift value for binned UVLF data
        Returns:
            [log10epsstar, log10Mcstar, alphastar, betastar, sigmaUV]: values of these parameters that match the given redshift
        """
        param_data = list(self.param_data.values())
        assert len(paramvector) == len([p for p in param_data if p['fit']]), 'The length of paramvector does not match the number of parameters you want to fit'
        
        # Deal with constant parameters, deal with piecewise
        params = np.zeros(len(param_data))
        j = 0 # This keeps track of how far we are into paramvector
        for i, param in enumerate(param_data): 
            if param['fit']: # If this is a fit parameter
                value = paramvector[j] 
                j+=1
            else: # If this parameter is held constant
                value = param['value']

            if not np.isscalar(value): # Piecewise, get the value that corresponds to that redshift
                index = np.where(self.zs == zcenter)[0][0]
                value = value[index]
                
            params[i] = value

        # Apply time evolution
        final_values = []
        for i in range(5): # This is alpha*, beta*, M_h, eps*, sigmaUV, the 5 base parameters
            base_idx = 2 * i
            value = params[base_idx]
            time_deriv = params[base_idx+1] # The order of the parameters are alternating base & time derivative, so this works
            final_values.append(value+(time_deriv*(zcenter-8)))

        # Apply mass dependence of sigma
        sig = final_values[-1]
        dsigdM = params[-1]
        final_values[-1] = sig + (dsigdM*(np.log10(self.HMFintclass.Mhtab)-10))

        return final_values

    def param_wrapper(self, params):
        """
        Puts paramvector into a format that Zeus can read
        Inputs:
            params [1darray]: parameters
        Outputs:
            astroparams [zeus Astro_Parameters object]: parameters for the UVLF, wrapped so that Zeus can read them
        """
        alphastar, betastar, log10Mcstar, log10epsstar, sigmaUV = params
        astroparams = zeus21.Astro_Parameters(self.UserParams, self.CosmoParams, epsstar=10**log10epsstar, Mc=10**log10Mcstar,alphastar=alphastar, 
                                              betastar=betastar, sigmaUV = sigmaUV) 
        
        return astroparams

    def get_best_fit(self, samples, excluded_params = []):
        """
        Get an array of the best fit values from MCMC samples (or, if the parameter is not fit, return the default value). This is for use in 
        conjunction with make_table
        Inputs:
            samples [Ndarray]: Sample chain returned by MCMC
            excluded_params [list]: any parameters you don't want to return
        Outputs:
            all_vals [list]: ordered list of best fit parameters (or default parameters where applicable)
            all_labels [list]: TeX representation of parameters
        """
        best_fit = np.average(samples, axis=0) # Get best fit values from the samples
        
        i = 0
        all_vals = []
        all_labels = []
    
        for name, param in self.param_data.items():
            if name in excluded_params:
                continue
            else: 
                if param.get('fit', True): # Get best fit value if the parameter is fit by MCMC
                    value = best_fit[i]
                    i += 1
                else: # Otherwise, take the default value
                    value = param.get('value', None)
                    
            all_vals.append(round(value, 2))
            all_labels.append(param.get('label', None))
    
        return all_vals, all_labels

def build_param_data(custom_params):
    """
    Builds the metadata around each parameter. You need only specify the parameters & the keys you want to change; anything unspecified will assume
    its default value. The order in which you pass the modifcations is not important, as long as you match the keys.
    Inputs:
        custom_params [dict of dicts]: nested dictionary that describes how you want to modify parameters from their default values
    Returns:
        default_values [dict of dicts]: dictionary with updated data set by custom_params and default data otherwise                        
    """
    
    # Master dictionary with defaults for each parameter label
    default_values = get_default_dict(table = False)

    if custom_params is None:
        return default_values
        
    for label in custom_params:
        if label not in default_values:
            raise ValueError(f"No default values found for label: {label}")

        default_values[label].update(custom_params[label])

    return default_values

def get_default_dict(table = True):
    """
    All supported parameters are included in this nested dictionary, including the following keys:
    'fit': when True, this will be fit with MCMC. When False, this value will be held fixed at the value provided under the 'value' key.
    'value': value to assign this parameter when 'fit' is False
    'start': starting value for this parameter in the MCMC
    'lower': lower bound for the flat prior in MCMC
    'upper': upper bound for the flat prior in MCMC
    'label': label, in mathematical format, for plotting
    Note that you may assign the 'value' of base parameters (alpha, beta, logMc, loge, sig) to be arrays. The code will interpret these as
    piecewise values across redshift (so the length of the array must match the number of redshifts you have data to fit for). This will only work
    if 'fit' is labeled False; there is currently not support for using MCMC to fit parameters in a piecewise fashion. 

    If table is true, it will return a readable pandas dataframe. If not, it will return as a dictionary.
    """
    default_values = {
        'alpha':   {'fit': True, 'value': 0.6, 'start': 0.6, 'lower': 0, 'upper': 4, 'label': r"$\alpha$"},
        'dalphadz':{'fit': True, 'value': 0, 'start': 0.01, 'lower': -0.5, 'upper': 0.5, 'label': r"$d\alpha/dz$"},
        'beta':    {'fit': True, 'value': -0.5, 'start': -0.5, 'lower': -1, 'upper': 0, 'label': r"$\beta$"},
        'dbetadz': {'fit': True, 'value': 0, 'start': 0.01, 'lower': -0.5, 'upper': 0.5, 'label': r"$d\beta/dz$"},
        'logMc':   {'fit': True, 'value': 12, 'start': 12, 'lower': 9, 'upper': 16, 'label': r'$\log(M_c)$'},
        'dlogMcdz':{'fit': True, 'value': 0, 'start': 0.01, 'lower': -0.5, 'upper': 0.5, 'label': r'$d\log(M_c)/dz$'},
        'loge':    {'fit': True, 'value': -0.5, 'start': -1, 'lower': -1, 'upper': 1, 'label': r"$\log(\epsilon_0)$"},
        'dlogedz': {'fit': True, 'value': 0, 'start': 0.01, 'lower': -0.5, 'upper': 0.5, 'label': r'$d\log(\epsilon_0)/dz$'},
        'sig':     {'fit': True, 'value': 0, 'start': 0.5, 'lower': 0, 'upper': 6, 'label': r'$\sigma_{\rm{UV}, 10}$'},
        'dsigdz':  {'fit': True, 'value': 0, 'start': 0.01, 'lower': -0.5, 'upper': 0.5, 'label': r'$d\sigma_{\rm{UV}}/dz$'},
        'dsigdlogM':  {'fit': True, 'value': 0, 'start': 0.01, 'lower': -0.5, 'upper': 0.5, 
                    'label': r'$d\sigma_{\rm{UV}}/d\log(M_{h})$'}
    }

    if table:
        return pd.DataFrame(default_values).T
    else:
        return default_values

def make_table(best_fits, param_labels, fit_labels):
    """
    Make a table to compare best fit values of parameters
    Inputs:
        best_fits [list of lists]: list of best fit parameters
        param_labels [list of strs]: names of parameters (rows of the table)
        fit_labels [list]: names of each type of fit (columns of the table)
    Outputs:
        dataframe table with labeled parameters for comparison
    """
    df = pd.DataFrame(best_fits, columns = param_labels)
    df.index = fit_labels
    
    return df.T

def decompose_data(data, MINRELERROR = 0.2):
    """
    Decompose data corresponding to a single redshift into its components, ensure that the error bars are reasonable
    Inputs:
        data [Nx6 array]: data corresponding to a single redshift
        MINRELERROR [float]: minimum relative error (prevents the error bars from being unrealistically small)
    Outputs:
        zdat, zerr, xdat, ydat, modified yerr_upper, yerr_lower, xerr (all sorted)
    """
    zdat = data[0] # redshift
    zerr = data[1] # delta redshift (redshift bin)
    xdat = data[2] # MUV
    sorting = np.argsort(xdat) # Make sure the data is ordered by MUV
    xdat = xdat[sorting]
    ydat = data[3][sorting] # phiUV 
    yerr_upper = data[4][sorting]
    yerr_lower = data[5][sorting]
    xerr = data[6][sorting] # error on MUV (MUV bins)

    yerr_upper, yerr_lower = np.fmax(yerr_upper, ydat * MINRELERROR), np.fmax(yerr_lower, ydat*MINRELERROR) # Make sure error bars aren't any
                                                                                                            # smaller than the relative error

    return zdat, zerr, xdat, ydat, yerr_upper, yerr_lower, xerr

def read_data(file_names, data_labels = None): 
    """
    Read in data and organize it by dataset and redshift. We want the data sorted by dataset for plotting purposes (so we can assign credit to the
    studies that observed different things). 
    Inputs:
        file_names [list of strs]: file names to read data from
        data_labels [list of strs]: names of datasets, for plotting purposes
    Returns:
        data, organized by dataset and redshift
    """
    if data_labels is None:
        data_labels = np.full('', len(file_names))
        
    all_data = []
    for fn in file_names:
        all_data.append(np.loadtxt(fn, skiprows=2, unpack = True))
                        
    redshifts = np.unique(np.concatenate([data[0] for data in all_data])) # Get a list of all redshifts that exist within the files
    dredshifts = np.ones_like(redshifts)/2. #approximate, there are true window functions to use

    # Organize data by redshift
    sorted_data = []
    #format is     zdat = data[0] zerr = data[1] xdat = data[2],  ydat = data[3]  yerr_upper = data[4]  yerr_lower = data [5] xerr = data[6] 
    
    for iz,z in enumerate(redshifts):
        z_separated = []
        for data, label in zip(all_data, data_labels):
            zlistindex = data[0] == 1.0*z
            datarr = [z,dredshifts[iz], data[1][zlistindex], data[3][zlistindex], data[4][zlistindex], np.abs(data[5][zlistindex]), 
                      #Use absolute value because the lower error bar is given as negative (centered on ydat)
                      data[2][zlistindex], label]
            z_separated.append(datarr)
        sorted_data.append(z_separated)

    return sorted_data

def reduce_data(sorted_data): 
    """
    Remove the sorting by dataset and just sort by redshift. The MCMC doesn't care where the data comes from, so this is for that.
    Inputs:
        sorted_data [list]: from read_data, data sorted by dataset and redshift.
    Returns:
        reduced_data [list]: data sorted by just redshift
    """
    reduced_data = []
    for zbin in sorted_data:
        z = zbin[0][0]
        dz = zbin[0][1]
        MUVs = np.concatenate([dat[2] for dat in zbin])
        phis = np.concatenate([dat[3] for dat in zbin])
        yerr_upper = np.concatenate([dat[4] for dat in zbin])
        yerr_lower = np.concatenate([dat[5] for dat in zbin])
        dMUV = np.concatenate([dat[6] for dat in zbin])
        reduced_data.append([z, dz, MUVs, phis, yerr_upper, yerr_lower, dMUV])

    return reduced_data