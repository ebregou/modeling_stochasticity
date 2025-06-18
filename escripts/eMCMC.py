# Purpose: interface with Zeus to run MCMC on models of the UVLF with time-evolving sigma_UV
# Author: Emily Bregou, Julian Muñoz
# Depends on: numpy, zeus21, emcee

# Standard packages
import numpy as np
import emcee

# Local packages
import zeus21

# MCMC class
class UVLF():
    def __init__(self, data, param_data, MINRELERROR = 0.2):
        self.data = data
        if type(self.data[0]) is not list: # Make sure the format is correct for the rest of the script if only one redshift is input
            self.data = [self.data] 
        self.zs = [dat[0] for dat in self.data]
        self.param_data = list(param_data.values()) # Turn a dictionary of dictionaries into a list of dictionaries containing relevant parameters
        self.lowers = [p['lower'] * p['scale'] for p in self.param_data if p.get('fit', True)]
        self.uppers = [p['upper'] * p['scale'] for p in self.param_data if p.get('fit', True)]
        self.ndim = len(self.lowers)
        self.nwalkers = 2*self.ndim
        self.MINRELERROR = MINRELERROR

        # Get cosmological parameters, construct HMF from Zeus
        CosmoParams_input = zeus21.Cosmo_Parameters_Input(zmin_CLASS=0.0)
        self.CosmoParams,ClassyCosmo, CorrFclass, self.HMFintclass =  zeus21.cosmo_wrapper(CosmoParams_input)

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
        params1 = [p['start'] * p['scale'] for p in self.param_data if p.get('fit', True)]
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
        
    def log_like(self, paramvector):
        """
        Calculate log likelihood
        Inputs:
            paramvector [1darray]: values of parameters
        Returns:
            loglike_curr [float]: log likelihood of the data given the model parameters
        """
        
        #now bin it appropriately at each z -- data part
        loglike_curr = 0.0

        # self.data has all the z & UVLF data
        for dataarrayz in self.data: # Add the log likelihoods together for each redshift. The log likelihood is just a sum over all the points
            # anyways, so this makes sense
            #datHSTz4=[3.8, mags_z4,phi_z4,err_z4,errx_z4]
            zdat = dataarrayz[0]
            zerr = dataarrayz[1]
            xdat = dataarrayz[2]
            ydat = dataarrayz[3]
            yerr = dataarrayz[4] 
            xerr = dataarrayz[5]                
            #izus = np.argmin(np.abs(z0list - zdat))      
            
            yerr = np.fmax(yerr, ydat*self.MINRELERROR) # Make sure error bars aren't any smaller than the relative error set above
    
            uvlftheory = self.UVLF_wrapper(zdat,zerr,xdat,xerr, paramvector)
            
            loglike_curr += -np.sum( (ydat - uvlftheory)**2/(2.0 * yerr**2) ) #assumed Gaussian, to be revisited.
        
        return loglike_curr

    def UVLF_wrapper(self, zcenter, zwidth, MUVcenters, MUVwidths, paramvector):
        """
        Computes and returns the UVLF at z=zcenters, with width zwidths, in bins centered at MUVcenters with width MUVwidths
        Inputs:
            zcenters [float]: center redshift value for binned UVLF data
            zwdith [float]: width of the redshift bin
            MUVcenters [1darray]: the central UV magnitude in each bin
            MUVwidths [1darray]: the width of the UV magnitude bins
            paramvector [1darray]: scaled values of parameters
        Outputs:
            PhiUV [1darray]: In units of mag^-1 Mpc^-3
        """

        unscaled_params = self.time_evolution(paramvector, zcenter)
        astroparams = self.param_wrapper(unscaled_params, zcenter)
        UVLFs_std = zeus21.UVLFs.UVLF_binned(astroparams,self.CosmoParams,self.HMFintclass,zcenter,zwidth,MUVcenters,MUVwidths)
        
        return UVLFs_std

    def time_evolution(self, paramvector, zcenter):
        """
        Applies the time evolution of each parameter so that we feed the evolved value, matching the given redshift, to the UVLF wrapper
        Inputs:
            paramvector [1darray]: scaled values of parameters
            zcenter [float]: center redshift value for binned UVLF data
        Returns:
            [log10epsstar, log10Mcstar, alphastar, betastar, sigmaUV]: values of these parameters that match the given redshift
        """
        assert len(paramvector) == len([p for p in self.param_data if p['fit']]), 'The length of paramvector does not match the number of parameters you want to fit'
        
        # Rescale, deal with constant parameters, deal with piecewise
        unscaled_params = np.zeros(len(self.param_data))
        j = 0 # This keeps track of how far we are into paramvector
        for i, param in enumerate(self.param_data): 
            if param['fit']: # If this is a fit parameter
                value = paramvector[j] 
                j+=1
            else: # If this parameter is held constant
                value = param['value']

            if not np.isscalar(value): # Piecewise, get the value that corresponds to that redshift
                index = np.where(self.zs == zcenter)[0][0]
                value = value[index]
                
            unscaled_params[i] = value / param['scale'] # Unscale (get back the true value of the parameter)

        # Apply time & mass evolution
        final_values = []
        for i in range(5): # This is alpha*, beta*, M_h, eps*, sigmaUV, the 5 base parameters
            base_idx = 3 * i
            value = unscaled_params[base_idx]
            time_deriv = unscaled_params[base_idx+1]
            mass_deriv = unscaled_params[base_idx+2]

            value = value + (time_deriv*(zcenter-8))
            if mass_deriv != 0:
               value = (value + (mass_deriv*(np.log10(self.HMFintclass.Mhtab))))
            final_values.append(value)

        return final_values

    def param_wrapper(self, unscaled_params, zcenter):
        """
        Puts paramvector into a format that Zeus can read
        Inputs:
            unscaled_params [1darray]: unscaled values of parameters
            zcenter [float]: center redshift value for binned UVLF data
        Outputs:
            astroparams [zeus Astro_Parameters object]: parameters for the UVLF, wrapped so that Zeus can read them
        """
        alphastar, betastar, log10Mcstar, log10epsstar, sigmaUV = unscaled_params
        astroparams = zeus21.Astro_Parameters(self.CosmoParams,epsstar=10**log10epsstar, Mc=10**log10Mcstar,alphastar=alphastar, 
                                              betastar=betastar, sigmaUV = sigmaUV) 
        
        return astroparams

def build_param_data(custom_params):
    """
    Builds the metadata around each parameter
    Inputs:
        custom_params [dict of dicts]: dictionary of dictionaries that describe how you want to modify parameters from their default values. Any
                                       labels that aren't passed will assume their default values
    Returns:
        default_values [dict of dicts]: dictionary with updated data set by custom_params and default data otherwise                        
    """
    
    # Master dictionary with defaults for each parameter label
    default_values = {
        'alpha':   {'fit': True, 'value': 0.6, 'start': 0.6, 'scale': 1, 'lower': 0, 'upper': 4, 'label': r"$\alpha$"},
        'dalphadz':{'fit': False, 'value': 0, 'start': 0.01, 'scale': 1, 'lower': -0.5, 'upper': 0.5, 'label': r"$d\alpha/dz$"},
        'dalphadM':{'fit': False, 'value': 0, 'start': 0.01, 'scale': 1, 'lower': -0.5, 'upper': 0.5, 'label': r"$d\alpha/dM_{h}$"},
        'beta':    {'fit': True, 'value': -0.5, 'start': -0.5, 'scale': 1, 'lower': -1, 'upper': 0, 'label': r"$\beta$"},
        'dbetadz': {'fit': False, 'value': 0, 'start': 0.01, 'scale': 1, 'lower': -0.5, 'upper': 0.5, 'label': r"$d\beta/dz$"},
        'dbetadM': {'fit': False, 'value': 0, 'start': 0.01, 'scale': 1, 'lower': -0.5, 'upper': 0.5, 'label': r"$d\beta/dM_{h}$"},
        'logMc':   {'fit': True, 'value': 12, 'start': 12, 'scale': 1, 'lower': 9, 'upper': 16, 'label': r'$\log(M_c)$'},
        'dlogMcdz':{'fit': False, 'value': 0, 'start': 0.01, 'scale': 1, 'lower': -0.5, 'upper': 0.5, 'label': r'$d\log(M_c)/dz$'},
        'dlogMcdM':{'fit': False, 'value': 0, 'start': 0.01, 'scale': 1, 'lower': -0.5, 'upper': 0.5, 'label': r'$d\log(M_c)/dM_{h}$'},
        'loge':    {'fit': True, 'value': -0.5, 'start': -1, 'scale': 1, 'lower': -1, 'upper': 1, 'label': r"$\log(\epsilon_0)$"},
        'dlogedz': {'fit': False, 'value': 0, 'start': 0.01, 'scale': 1, 'lower': -0.5, 'upper': 0.5, 'label': r'$d\log(\epsilon_0)/dz$'},
        'dlogedM': {'fit': False, 'value': 0, 'start': 0.01, 'scale': 1, 'lower': -0.5, 'upper': 0.5, 
                   'label': r'$d\log(\epsilon_0)/dM_{h}$'},
        'sig':     {'fit': True, 'value': 0, 'start': 0.5, 'scale': 1, 'lower': 0, 'upper': 3, 'label': r'$\sigma_{\rm{UV}, 0}$'},
        'dsigdz':  {'fit': False, 'value': 0, 'start': 0.01, 'scale': 1, 'lower': -0.5, 'upper': 0.5, 'label': r'$d\sigma_{\rm{UV}}/dz$'},
        'dsigdM':  {'fit': False, 'value': 0, 'start': 0.01, 'scale': 1, 'lower': -0.5, 'upper': 0.5, 
                    'label': r'$d\sigma_{\rm{UV}}/dM_{h}$'}
    }

    if custom_params is None:
        return default_values
        
    for label in custom_params:
        if label not in default_values:
            raise ValueError(f"No default values found for label: {label}")

        default_values[label].update(custom_params[label])

    return default_values