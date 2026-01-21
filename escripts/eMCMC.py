# Purpose: interface with Zeus to run MCMC on models of the UVLF with time-evolving sigma_UV
# Authors: Emily Bregou, Julian Muñoz

# Standard packages
import numpy as np
import emcee
import pandas as pd
import zeus21
import h5py

# Local packages
from escripts import eplots
from escripts import edata

# MCMC class
class UVLF():
    def __init__(self, sorted_data, param_data, Mhpivot = 11, min_sig = 0.3, backend_filename = None, precisionboost = 1.5, cut_sig = True):
        self.sorted_data = sorted_data # This isn't used in the MCMC at all but it's useful to have for plotting results since it divides data
                                        # by redshift & author
        self.data = edata.reduce(self.sorted_data)
        if type(self.data[0]) is not list: # Make sure the format is correct for the rest of the script if only one redshift is input
            self.data = [self.data] 
        self.zs = [dat[0] for dat in self.data]
        self.param_data = param_data
        self.UserParams = zeus21.User_Parameters(precisionboost = precisionboost) # Increase the resolution of the HMF to get smooth UVLFs
        self.fit = np.where(self.param_data['fit'])[0]
        self.notfit = np.where(self.param_data['fit'] == False)[0]
        self.lowers = self.param_data['lower'].to_numpy()
        self.uppers = self.param_data['upper'].to_numpy()
        self.ndim = np.sum(self.param_data['fit']) # Count the number of parameters to fit
        self.nwalkers = 2*self.ndim # Walkers = twice the number of parameters
        self.Mhpivot = Mhpivot # The central halo mass that corresponds to sigma_0
        self.min_sig = min_sig # Minimum value sigma can take (in case of time evolving / halo mass evolving sigma). If you set it any lower than
                            # the default value, you may have a lumpy UVLF
        self.backend_filename = backend_filename
        self.cut_sig = cut_sig # Whether or not to force a small sigma for halos smaller than the atomic cooling limit
        if self.backend_filename is not None:
            self.backend = emcee.backends.HDFBackend(self.backend_filename)
            self.backend.reset(self.nwalkers, self.ndim)
        else:
            self.backend = None

        # Get cosmological parameters, construct HMF from Zeus
        CosmoParams_input = zeus21.Cosmo_Parameters_Input(zmin_CLASS=0.0)
        self.CosmoParams,ClassyCosmo, CorrFclass, self.HMFintclass =  zeus21.cosmo_wrapper(self.UserParams, CosmoParams_input)

        # Get baseline astronomical parameters
        self.Astro_Parameters = zeus21.Astro_Parameters(self.UserParams, self.CosmoParams)

        # Create MCMC sampler
        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_prob, backend = self.backend) 

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
        return lprior + lpost
 
    def log_prior(self, paramvector):
        """
        Calculate the log prior for a flat prior: 0 if within range; negative infinity if outside
        Inputs:
            paramvector [1darray]: values of parameters
        Returns:
            prior [float]: (0 if within range; negative infinity if outside)
        """
        if all(lower < t < upper for t, lower, upper in zip(paramvector, self.lowers[self.fit], self.uppers[self.fit])):
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
        loglike_curr = 0.0

        # self.data has all the z & UVLF data
        if alt_data is not None:
            data = alt_data
        else:
            data = self.data
            
        for dataarrayz in data: # Add the log likelihoods together for each redshift. The log likelihood is just a sum over all the points
            # anyways, so this makes sense
            
            zdat, zerr, xdat, ydat, yerr_upper, yerr_lower, xerr = edata.decompose(dataarrayz)
    
            uvlftheory = self.UVLF_wrapper(zdat,zerr,xdat,xerr, paramvector)
            
            # This applies the correct sigma to the Gaussian distribution of the likelihood, based on whether the point returned is above or below
            # the mean, for when the datapoints have asymmetrical errorbars. 
            yerr_asymmetrical = np.array([yerr_upper[i] if uvlftheory[i] > ydat[i] else yerr_lower[i] for i in range(len(ydat))])

            # this formula is just proportional to the natural log of a Gaussian
            loglike_curr += -np.sum((ydat - uvlftheory)**2/(2.0 * yerr_asymmetrical**2))

        return loglike_curr

    def UVLF_wrapper(self, zcenter, zwidth, MUVcenters, MUVwidths, paramvector, get_bias = False):
        """
        Computes and returns the UVLF at z=zcenters, with width zwidths, in bins centered at MUVcenters with width MUVwidths
        Inputs:
            zcenters [float]: center redshift value for binned UVLF data
            zwdith [float]: width of the redshift bin
            MUVcenters [1darray]: the central UV magnitude in each bin
            MUVwidths [1darray]: the width of the UV magnitude bins
            paramvector [1darray]: parameters
            get_bias [bool]: whether or not to calculate the bias-weighted UVLF from Zeus & then divide by the ULVF to get back the bias
        Outputs:
            PhiUV [1darray]: In units of mag^-1 Mpc^-3
            bias [1darray]: b(MUV) (unitless), returned if get_bias is True
        """
        params = self.time_evolution(paramvector, zcenter)
        astroparams = self.param_wrapper(params)
        UVLFs_std = zeus21.UVLFs.UVLF_binned(astroparams,self.CosmoParams,self.HMFintclass,zcenter,zwidth,MUVcenters,MUVwidths)
        if get_bias:
            bias_weighted_UVLF = zeus21.UVLFs.UVLF_binned(astroparams,self.CosmoParams,self.HMFintclass,zcenter,zwidth,MUVcenters,MUVwidths, 
                                                     RETURNBIAS = True)
            return UVLFs_std, bias_weighted_UVLF/UVLFs_std
        else:
            return UVLFs_std

    def time_evolution(self, paramvector, zcenter):
        """
        Applies the time evolution of each parameter so that we feed the evolved value, matching the given redshift, to the UVLF wrapper
        Applies the halo mass evolution of sigmaUV & ensures that halos smaller than the atomic cooling limit are given small sigmaUV
        Inputs:
            paramvector [1darray]: parameters
            zcenter [float]: center redshift value for binned UVLF data
        Returns:
            [alphastar, betastar, log10epsstar, log10Mcstar, sigmaUV]: values of these parameters that match the given redshift
        """

        # Deal with piecewise parameters if necessary
        if np.isscalar(all(self.param_data['value'].iloc[self.notfit])):
            insert_vals = self.param_data['value'].iloc[self.notfit]
        else:
            insert_vals = np.zeros(self.notfit) 
            for i, val in enumerate(self.param_data['value'][self.notfit]):
                if np.isscalar(val):
                    insert_vals[i] = val
                else:
                    zindex = np.where(self.zs == zcenter)[0][0]
                    insert_vals[i] = val[zindex]

        full_paramvector =  np.zeros(len(self.param_data['label'])) # All parameters accounted for
        full_paramvector[self.notfit] = insert_vals
        full_paramvector[self.fit] = paramvector

        # Apply time evolution
        base_idx = np.arange(0, 10, 2) # The order of the parameters are alternating base & time derivative
        param_base = full_paramvector[base_idx]
        time_derivs = full_paramvector[base_idx + 1]
        param_values = param_base + (time_derivs*(zcenter-self.Astro_Parameters._zpivot))
        # Limit alpha to be positive and beta to be negative regardless of redshift evolution to preserve the double power law structure of SFE
        param_values[2] = max(param_values[2], 0) # alpha should be positive
        param_values[3] = min(param_values[3], 0) # beta should be negative

        # Apply mass dependence of sigma
        sig = param_values[-1]
        dsigdM = full_paramvector[-1]
        param_values = list(param_values) # Need to convert to a list such that the last element can be an array
        sig_array = sig + (dsigdM*(np.log10(self.HMFintclass.Mhtab)-self.Mhpivot))
        sig_array = sig_array.clip(min=self.min_sig)

        # Assign small sigma to halos below the atomic cooling limit
        Matom = zeus21.sfrd.Matom(zcenter) # Get the atomic cooling limit at this redshift
        if self.cut_sig:
            below_limit = np.where(self.HMFintclass.Mhtab < Matom)[0]
            sig_array[below_limit] = 1e-4

        param_values[-1] = sig_array 

        return param_values

    def param_wrapper(self, params):
        """
        Puts paramvector into a format that Zeus can read
        Inputs:
            params [1darray]: parameters
        Outputs:
            astroparams [zeus Astro_Parameters object]: parameters for the UVLF, wrapped so that Zeus can read them
        """
        alphastar, betastar, log10Mcstar, log10epsstar, sigmaUV = params
        astroparams = zeus21.Astro_Parameters(self.UserParams, self.CosmoParams, epsstar=10**log10epsstar, 
                                              Mc=10**log10Mcstar,alphastar=alphastar, betastar=betastar, sigmaUV = sigmaUV)
        return astroparams

    def get_fit(self, backend_file = None, burn_in = None, exclude_unfit = True, excluded_params = [],):
        """
        Get samples and best fit values from MCMC samples (or, if the parameter is not fit, return the default value).
        Inputs:
            backend_file [str]: .h5 file containing previously saved chain
            burn_in [int]: number of steps to discard as burnin
            exclude_unfit [bool]: whether or not to exclude parameters that are set by default (and not by the MCMC chain) 
            excluded_params [list]: any parameters you don't want to return
        Outputs:
            samples [Ndarray]: MCMC chain samples
            best_fit [list]: ordered list of best fit parameters (or default parameters where applicable)
            bounds [Nx2 array]: Upper and lower bounds on parameter values that correspond to the 16th & 84th percentile 
            all_labels [list]: TeX representation of parameters, used with make_table()
        """
        if burn_in is None: # Standard value of burn in
            burn_in = 8000
        
        if backend_file is None:
            samples = self.sampler.get_chain(discard = burn_in, flat=True)
            log_prob = self.sampler.get_log_prob(discard=burn_in, flat=True)
        else:
            reader = emcee.backends.HDFBackend(backend_file)
            samples = reader.get_chain(discard = burn_in, flat=True)
            log_prob = reader.get_log_prob(discard=burn_in, flat = True)

        # Get highest probability sample
        best_fit_data = samples[np.argmax(log_prob)]
        i = 0
        best_fit = []
        all_labels = []
        exclude_indices = []
    
        for index, key in enumerate(self.param_data.T.keys()):
            if key in excluded_params:
                if index not in self.notfit: # Make sure you don't double count if the parameter wasn't fit anyways
                    exclude_indices.append(i)
                    i+=1
                continue
            else: 
                if self.param_data.T[key].fit: # Get best fit value if the parameter is fit by MCMC
                    value = best_fit_data[i]
                    i += 1
                else: # Otherwise, take the default value
                    if exclude_unfit:
                        continue
                    else:
                        value = self.param_data.T[key].value
            best_fit.append(round(value, 2))
            all_labels.append(self.param_data.T[key].label)

        # Get the bounds on best fit data
        bounds = np.zeros((len(samples[0]),2))
        for i in range(len(samples[0])):
            bounds[i] = np.percentile(samples[:, i], [16, 84])

        return np.delete(samples, exclude_indices, axis=1), best_fit, bounds, all_labels

    def run_MCMC(self, Nsteps = 100000, ICs = None):
        """
        Run MCMC, store the chain that's created
        Inputs:
            Nsteps [int]: number of steps in the chain. This will be multiplied by self.ndim for the total number of samples
        Outputs:
            None
        """
        self.sampler.reset()
        
        # Get ICs for running MCMC
        if ICs is None:
            ICs = self.generate_ICs(self.lowers[self.fit], self.uppers[self.fit])

        # Run MCMC with a progress bar
        _ = self.sampler.run_mcmc(ICs, Nsteps, progress = True)

        return
    
    def generate_ICs(self, lower_bounds, upper_bounds):
        """
        Generate initial conditions
        Inputs: 
            lower_bounds [1darray]: lower bounds for ICs
            upper_bounds [1darray]: upper bounds for ICs
            nwalkers [int]: number of walkers
        Returns: 
            ICs [array]: ICs to run MCMC with
        """
        nwalkers = self.nwalkers
        default_df = get_default_df()

        # Make sure the input initial condition values are within the default upper and lower limits
        assert (all((base_lower <= input_lower for (base_lower, input_lower) in zip(default_df['lower'][self.fit], lower_bounds))) & 
                all((base_upper >= input_upper for (base_upper, input_upper) in zip(default_df['upper'][self.fit], upper_bounds)))), \
                    'the input uppers and lowers are outside the default bounds'

        ordered_ICs = np.linspace(lower_bounds, upper_bounds, nwalkers, dtype = 'float') 
        # Need to set datatype or it won't work
        rand_inds = np.random.rand(*ordered_ICs.shape).argsort(axis=0) # Shuffled indices (this is so that a walker that starts at a lower for a
                                                            # given parameter doesn't also start at the lower bound for all other parameters)
        ICs = np.take_along_axis(ordered_ICs,rand_inds,axis=0) # Apply the randomization

        return ICs
        

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
    default_values = get_default_df()

    if custom_params is None:
        return default_values
        
    for label in list(custom_params.keys()): # Label refers to things like 'alpha', 'dsigdz'
        if label not in default_values.index:
            raise ValueError(f"No default values found for label: {label}")
        for sublabel in list(custom_params[label]): # Sublabel refers to things like 'fit' or 'lower'
            if sublabel not in default_values.keys():
                raise ValueError(f"No default values found for sublabel: {parameter}")
            default_values.loc[label, sublabel] = custom_params[label][sublabel]


    return default_values



def get_default_df():
    """
    All supported parameters are included in this nested dictionary, including the following keys:
    'fit': when True, this will be fit with MCMC. When False, this value will be held fixed at the value provided under the 'value' key.
    'value': value to assign this parameter when 'fit' is False
    'lower': lower bound for the flat prior in MCMC
    'upper': upper bound for the flat prior in MCMC
    'label': label, in mathematical format, for plotting
    Note that you may assign the 'value' of base parameters (alpha, beta, logMc, loge, sig) to be arrays. The code will interpret these as
    piecewise values across redshift (so the length of the array must match the number of redshifts you have data to fit for). This will only work
    if 'fit' is labeled False; there is currently not support for using MCMC to fit parameters in a piecewise fashion. 
    """
    default_values = {
        'alpha':   {'fit': True, 'value': 0.6, 'lower': 0, 'upper': 4, 'label': r"$\alpha$"},
        'dalphadz':{'fit': True, 'value': 0,  'lower': -0.5, 'upper': 0.5, 'label': r"$d\alpha/dz$"},
        'beta':    {'fit': True, 'value': -0.5,  'lower': -3, 'upper': 0, 'label': r"$\beta$"},
        'dbetadz': {'fit': True, 'value': 0,  'lower': -1, 'upper': 1.5, 'label': r"$d\beta/dz$"},
        'logMc':   {'fit': True, 'value': 12, 'lower': 9, 'upper': 16, 'label': r'$\log(M_c)$'},
        'dlogMcdz':{'fit': True, 'value': 0,  'lower': -0.5, 'upper': 0.5, 'label': r'$d\log(M_c)/dz$'},
        'loge':    {'fit': True, 'value': -0.5, 'lower': -4.5, 'upper': 0, 'label': r"$\log(\epsilon_{\star})$"},
        'dlogedz': {'fit': True, 'value': 0,  'lower': -0.5, 'upper': 0.5, 'label': r'$d\log(\epsilon_{\star})/dz$'},
        'sig':     {'fit': True, 'value': 0, 'lower': 0, 'upper': 6, 'label': r'$\sigma_{\rm{UV}, M_c}$'},
        'dsigdz':  {'fit': True, 'value': 0,  'lower': -1, 'upper': 1, 'label': r'$d\sigma_{\rm{UV}}/dz$'},
        'dsigdlogM':  {'fit': True, 'value': 0,  'lower': -2, 'upper': 3, 'label': r'$d\sigma_{\rm{UV}}/d\log(M_{h})$'}
    }

    return pd.DataFrame(default_values).T