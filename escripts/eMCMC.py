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
    def __init__(self, sorted_data, param_data, Mhpivot = 11, backend_filename = None, accretion_model = 'RP16', precisionboost = 1.95, min_t = 5e6, burn_in = 8000):
        self.sorted_data = sorted_data # This isn't used in the MCMC at all but it's useful to have for plotting results since it divides data
                                        # by redshift & author. There can also be datasets in here that aren't used in the likelihoods-- see 
                                        # edata.sorted_data
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
        self.backend_filename = backend_filename
        self.min_t = min_t # Minimum time that a halo could convert all of its gas into stars. Used to set min(MUV)
        if self.backend_filename is not None:
            self.backend = emcee.backends.HDFBackend(self.backend_filename)
            self.backend.reset(self.nwalkers, self.ndim)
        else:
            self.backend = None

        # Get cosmological parameters, construct HMF from Zeus
        CosmoParams_input = zeus21.Cosmo_Parameters_Input(zmin_CLASS=0.0)
        self.CosmoParams,ClassyCosmo, CorrFclass, self.HMFintclass =  zeus21.cosmo_wrapper(self.UserParams, CosmoParams_input) 
        #HMFintclass.HMF_int gives the halo mass function in units M_odot^-1*Mpc^-3


        # Get baseline astronomical parameters
        self.accretion_model = accretion_model
        self.Astro_Parameters = zeus21.Astro_Parameters(self.UserParams, self.CosmoParams, accretion_model = self.accretion_model)

        # Create MCMC sampler
        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_prob, backend = self.backend) 

        self.burn_in = burn_in

    def calc_min_MUV(self, Mhtab):
        """
        Calculate the brightest galaxies of a given halo mass can be
        """
        if self.min_t is None:
            return None
        
        fb = self.CosmoParams.OmegaB / self.CosmoParams.OmegaM
        max_SFR = Mhtab*fb/self.min_t
        MUV = zeus21.UVLFs.MUV_of_SFR(max_SFR, self.Astro_Parameters._kappaUV)
        return MUV

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

        val = lprior + lpost
        if not np.isscalar(val):
            print("NON-SCALAR RETURN:", type(val), np.shape(val))
            raise ValueError
        
        return val
 
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
        
    def log_like(self, paramvector, alt_data = None, return_by_z = False):
        """
        Calculate log likelihood
        Inputs:
            paramvector [1darray]: values of parameters
            alt_data [Ndarray]: alternative data to input if you don't want to use self.data (in case you want to calculate log_like over a diff.
                                redshift range than you originally fit to)
            return_by_z [bool]: whether or not to also return a list of log likelihoods, broken down by each redshift
        Returns:
            loglike_curr [float]: log likelihood of the data given the model parameters
        """
        loglike_curr = 0.0
        log_like_z = np.zeros_like(self.zs)

        # self.data has all the z & UVLF data
        if alt_data is not None:
            data = alt_data
        else:
            data = self.data
        
        for i, dataarrayz in enumerate(data): # Add the log likelihoods together for each redshift. The log likelihood is just a sum over all the points
            # anyways, so this makes sense
            
            zdat, zerr, xdat, ydat, yerr_upper, yerr_lower, xerr = edata.decompose(dataarrayz)
    
            uvlftheory = self.UVLF_wrapper(zdat,zerr,xdat,xerr, paramvector)
            
            # This applies the correct sigma to the Gaussian distribution of the likelihood, based on whether the point returned is above or below
            # the mean, for when the datapoints have asymmetrical errorbars. 
            yerr_asymmetrical = np.array([yerr_upper[i] if uvlftheory[i] > ydat[i] else yerr_lower[i] for i in range(len(ydat))])

            # this formula is just proportional to the natural log of a Gaussian
            log_like_z[i] = -np.sum((ydat - uvlftheory)**2/(2.0 * yerr_asymmetrical**2))

        if return_by_z:
            return log_like_z
        
        return np.sum(log_like_z)

    def UVLF_wrapper(self, zcenter, zwidth, MUVcenters, MUVwidths, paramvector, return_weights = False, get_bias = False):
        """
        Computes and returns the UVLF at z=zcenters, with width zwidths, in bins centered at MUVcenters with width MUVwidths
        Inputs:
            zcenters [float]: center redshift value for binned UVLF data
            zwdith [float]: width of the redshift bin
            MUVcenters [1darray]: the central UV magnitude in each bin
            MUVwidths [1darray]: the width of the UV magnitude bins
            paramvector [1darray]: parameters
            minMUV [1darray]: alternative minimum MUV
            return_weights [bool]: whether to return the grid of P(MUV|Mh) instead of the UVLF
            get_bias [bool]: whether or not to calculate the bias-weighted UVLF from Zeus & then divide by the ULVF to get back the bias
        Outputs:
            PhiUV [1darray]: In units of mag^-1 Mpc^-3
            bias [1darray]: b(MUV) (unitless), returned if get_bias is True
        """
        minMUV = self.calc_min_MUV(self.HMFintclass.Mhtab)
        params = self.time_evolution(paramvector, zcenter)
        astroparams = self.param_wrapper(params)

        if return_weights:
            return zeus21.UVLFs.UVLF_binned(astroparams,self.CosmoParams,self.HMFintclass,zcenter,zwidth,MUVcenters,MUVwidths,minMUV,
                                            RETURNWEIGHTS = True)
        else:
            UVLFs_std = zeus21.UVLFs.UVLF_binned(astroparams,self.CosmoParams,self.HMFintclass,zcenter,zwidth,MUVcenters,MUVwidths, minMUV)
            if get_bias:
                bias_weighted_UVLF = zeus21.UVLFs.UVLF_binned(astroparams,self.CosmoParams,self.HMFintclass,zcenter,zwidth,MUVcenters,MUVwidths, minMUV,
                                                     RETURNBIAS = True)
                return UVLFs_std, bias_weighted_UVLF/UVLFs_std
            else:
                return UVLFs_std

    def time_evolution(self, paramvector, zcenter):
        """
        Applies the time evolution of each parameter so that we feed the evolved value, matching the given redshift, to the UVLF wrapper
        Applies the halo mass evolution and minimum of sigmaUV.
        Inputs:
            paramvector [1darray]: parameters
            zcenter [float]: center redshift value for binned UVLF data
        Returns:
            [alphastar, betastar, log10epsstar, log10Mcstar, sigmaUV, C0, C1]: values of these parameters that match the given redshift
        """

        # Deal with piecewise parameters
        insert_default_vals = self.assign_piecewise(self.param_data['value'].iloc[self.notfit], zcenter)
        insert_input_vals = self.assign_piecewise(paramvector, zcenter)

        # Insert all parameter values, prepare for time / halo mass evolution
        full_paramvector =  np.zeros(len(self.param_data['label'])) # All parameters accounted for
        full_paramvector[self.notfit] = insert_default_vals
        full_paramvector[self.fit] = insert_input_vals

        # Apply time evolution
        base_idx = np.arange(0, 10, 2) # The order of the parameters are alternating base & time derivative
        param_base = full_paramvector[base_idx]
        time_derivs = full_paramvector[base_idx + 1]
        param_values = param_base + (time_derivs*(zcenter-self.Astro_Parameters._zpivot))
        # Limit alpha to be positive and beta to be negative regardless of redshift evolution to preserve the double power law structure of SFE
        param_values[2] = max(param_values[2], 0) # alpha should be positive
        param_values[3] = min(param_values[3], 0) # beta should be negative

        # Apply mass dependence of sigma
        sig = param_values[4]
        dsigdM = full_paramvector[10]
        min_sig = full_paramvector[11]
        param_values = list(param_values) # Need to convert to a list such that the last element can be an array
        sig_array = sig + (dsigdM*(np.log10(self.HMFintclass.Mhtab)-self.Mhpivot))
        sig_array = sig_array.clip(min=min_sig)
        param_values[4] = sig_array 

        # Add back in C0 & C1 dust parameters
        param_values.extend(full_paramvector[-2:])

        return param_values
    
    def assign_piecewise(self, values, zcenter):
        """
        Helper function for time_evolution. If any parameters are given as piecewise instead of single values, assign the single value that
        corresponds to the redshfit zcenter.
        Inputs:
            value [list]: list of parameters. May contain lists embedded in it if there are piecewise parameters.
            zcenter [float]: center redshift value for binned UVLF data
        Returns:
            insert_vals [list]: list of parameters with a single value corresponding to each slot. The single value is chosen to correspond 
                                to the given zcenter.
        """
        insert_vals = np.zeros(len(values))
        for i, val in enumerate(values):
            if np.isscalar(val):
                insert_vals[i] = val
            else: # Assign the correct value of the parameter if the input is piecewise (find which redshift it matches to)
                zindex = np.where(self.zs == zcenter)[0][0]
                insert_vals[i] = val[zindex]
        return insert_vals
                    

    def param_wrapper(self, params):
        """
        Puts paramvector into a format that Zeus can read
        Inputs:
            params [1darray]: parameters
        Outputs:
            astroparams [zeus Astro_Parameters object]: parameters for the UVLF, wrapped so that Zeus can read them
        """
        alphastar, betastar, log10Mcstar, log10epsstar, sigmaUV, C0, C1 = params
        astroparams = zeus21.Astro_Parameters(self.UserParams, self.CosmoParams, epsstar=10**log10epsstar, 
                                              Mc=10**log10Mcstar,alphastar=alphastar, betastar=betastar, sigmaUV = sigmaUV, C0dust = C0, C1dust = C1,
                                              accretion_model = self.accretion_model)
        return astroparams

    def get_fit(self, backend_file = None, exclude_unfit = True, include_params = None, return_log_prob = False, burn_in = None):
        """
        Get samples and best fit values from MCMC samples (or, if the parameter is not fit, return the default value).
        Inputs:
            backend_file [str]: .h5 file containing previously saved chain
            exclude_unfit [bool]: whether or not to exclude parameters that are set by default (and not by the MCMC chain) 
            include_params [list]: use if you only want to return certain parameters. Anything not listed will not be returned. If None, all parameters 
                                    will be returned
            return_log_prob [bool]: whether or not to also return the log probability for each sample (for finding the best fit or the best fit in a certain 
                                    region of parameter space.)
            burn_in [int]: burn in if different from default
        Outputs:
            samples [Ndarray]: MCMC chain samples
            best_fit [list]: ordered list of best fit parameters (or default parameters where applicable)
            bounds [Nx2 array]: Upper and lower bounds on parameter values that correspond to the 16th & 84th percentile 
            all_labels [list]: TeX representation of parameters, used with make_table()
            log_prob [list]: Probability of each sample, if return_log_prob
        """
        
        if burn_in is None:
            burn_in = self.burn_in
        if include_params is None:
            include_params = self.param_data.T.keys()
        
        if backend_file is None:
            samples = self.sampler.get_chain(discard = burn_in, flat=True)
            log_prob = self.sampler.get_log_prob(discard=burn_in, flat=True)
        else:
            reader = emcee.backends.HDFBackend(backend_file)
            samples = reader.get_chain(discard = self.burn_in, flat=True)
            log_prob = reader.get_log_prob(discard=self.burn_in, flat = True)

        # Get highest probability sample
        best_fit_data = samples[np.argmax(log_prob)]
        i = 0
        best_fit = []
        all_labels = []
        exclude_indices = []
        bounds_insert = []
    
        for index, key in enumerate(self.param_data.T.keys()):
            if key not in include_params:
                if index not in self.notfit: # Make sure you don't double count if the parameter wasn't fit anyways
                    exclude_indices.append(i) 
                    i+=1
                continue
            else: 
                if self.param_data.T[key].fit: # Get best fit value if the parameter is fit by MCMC
                    value = best_fit_data[i]
                    bounds_insert.append(index)
                    i += 1
                else: # Otherwise, take the default value
                    if exclude_unfit:
                        continue
                    else:
                        value = self.param_data.T[key].value 
            best_fit.append(value)
            all_labels.append(self.param_data.T[key].label)

        # Get the bounds on best fit data
        if exclude_unfit:
            bounds= np.percentile(samples, [16, 84], axis = 0)
        else:
            bounds = np.full((2, len(best_fit)), np.nan)
            sampled_bounds = np.percentile(samples, [16, 84], axis = 0)
            bounds[:, bounds_insert] = sampled_bounds
        if return_log_prob:
            return np.delete(samples, exclude_indices, axis=1), np.array(best_fit), bounds, all_labels, log_prob
        else:
            return np.delete(samples, exclude_indices, axis=1), np.array(best_fit), bounds, all_labels

    def run_MCMC(self, Nsteps = None, ICs = None):
        """
        Run MCMC, store the chain that's created
        Inputs:
            Nsteps [int]: number of steps in the chain. This will be multiplied by self.ndim for the total number of samples. If None will default to 25
                            times the burn in
        Outputs:
            None
        """
        self.sampler.reset()

        if Nsteps is None:
            Nsteps = 25 * self.burn_in # self.burn_in should be set to twice the autocorrelation length such that this is 50 times the autocorrelation length
        
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
    
    def get_all_MUV_bins(self):
        """
        Get all unique MUV bins & errors present in the data. Helpful for plotting over the largest set of x values that you can.
        Returns:
            MUVcenters [1darray]: center of all unique MUV bins
            MUVwidths [1darray]: widths of all unique MUV bins
        """

        # Get the biggest possible grid of MUV data to plot over
        MUVcenters, inds = np.unique(np.concatenate([data[2] for data in self.data]), return_index = True) 
        # Get the corresponding bin width
        MUVwidths = np.concatenate([data[6] for data in self.data])[inds]

        return MUVcenters, MUVwidths
        

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
        'alpha':   {'fit': True, 'value': 0.6, 'lower': 0, 'upper': 2.5, 'label': r"$\alpha$"},
        'dalphadz':{'fit': False, 'value': 0,  'lower': -0.5, 'upper': 0.5, 'label': r"$d\alpha/dz$"},
        'beta':    {'fit': True, 'value': -0.5,  'lower': -2, 'upper': 0, 'label': r"$\beta$"},
        'dbetadz': {'fit': False, 'value': 0,  'lower': -0.5, 'upper': 0.5, 'label': r"$d\beta/dz$"},
        'logMc':   {'fit': True, 'value': 12, 'lower': 10, 'upper': 14, 'label': r'$\log(M_c)$'},
        'dlogMcdz':{'fit': False, 'value': 0,  'lower': -0.5, 'upper': 0.5, 'label': r'$d\log(M_c)/dz$'},
        'loge':    {'fit': True, 'value': -0.5, 'lower': -3, 'upper': 0, 'label': r"$\log(\epsilon_{\star})$"},
        'dlogedz': {'fit': False, 'value': 0,  'lower': -0.5, 'upper': 0.5, 'label': r'$d\log(\epsilon_{\star})/dz$'},
        'sig':     {'fit': True, 'value': 0, 'lower': 0, 'upper': 5, 'label': r'$\sigma_{\rm{UV}, M_c}$'},
        'dsigdz':  {'fit': False, 'value': 0,  'lower': -0.5, 'upper': 0.5, 'label': r'$d\sigma_{\rm{UV}}/dz$'},
        'dsigdlogM':  {'fit': False, 'value': 0,  'lower': -1.1, 'upper': 1.1, 'label': r'$d\sigma_{\rm{UV}}/d\log(M_{h})$'},
        'min_sig': {'fit': False, 'value': 0.3, 'lower': 0.2, 'upper': 2.5, 'label': r'$\min(\sigma_{\rm{UV}}$)'},
        'C0': {'fit':False, 'value':4.43, 'lower': 2.5, 'upper':4.5, 'label': r'$C_{0}$'},
        'C1': {'fit': False, 'value': 1.99, 'lower': 1.1, 'upper': 2.1, 'label': r'$C_{1}$'}
    }

    return pd.DataFrame(default_values).T

def get_narrowed_ICs(my_UVLF, best_fit = None, bounds = None, labels = None, backend_file = None, include_params = None):
    """
    Get bounds for re-running a chain around the best fit (narrowing the initial conditions)
    Inputs:
        my_UVLF [UVLF object]
        best_fit [list of floats]: best fit values
        bounds [2darray of floats]: lower & upper bounds on best fit
        labels [list of strings]: parameter names
        backend_file [str]: name of file where walkers are stored
        include_params [list]: list of parameter keys to include
    Returns:
        lowers, uppers for ICs in narrowed run, considering phyiscal bounds and the symmetry of the ICs
    """
    if backend_file is not None:
        _, best_fit, bounds, labels = my_UVLF.get_fit(backend_file = backend_file)
    
    max_bounds = np.maximum(np.abs(best_fit-bounds[0]), np.abs(bounds[1]-best_fit)) # Largest deviation from the best fit
    lowers = np.zeros_like(best_fit)
    uppers = np.zeros_like(best_fit)

    if include_params is None:
        include_idxs = my_UVLF.fit
    else:
        include_idxs = my_UVLF.param_data.T.keys().isin(include_params)

    for i, (bf, label, lower, upper, default_lower, default_upper, max_bound) in enumerate(zip(best_fit, labels, bounds[0], 
                                                                                    bounds[1], my_UVLF.lowers[include_idxs],
                                                                             my_UVLF.uppers[include_idxs], max_bounds)):
        if bf > lower:
            lowers[i] = lower
        else:
            symm = bf - max_bound
            if symm > default_lower:
                lowers[i] = symm
            else:
                lowers[i] = default_lower
                print(f'{label} lower, {symm}, out of bounds')
        if bf < upper:
            uppers[i] = upper
        else:
            symm = bf + max_bound
            if symm < default_upper:
                uppers[i] = symm
            else:
                uppers[i] = default_upper
                print(f'{label} upper, {symm}, out of bounds')

    return lowers, uppers
