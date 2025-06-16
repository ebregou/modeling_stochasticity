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
    def __init__(self, data, params, MINRELERROR = 0.2):
        """
        Scale is an array that describes the factor to scale each parameter has been multiplied by. The reason for multiplying some parameters by 
        a scale factor is because some have best fit values that are very small (e.g. 0.01), especially the time-evolution parameters. Scaling
        them up (by multiplying by 100 in this case) allows steps in each parameter to have the same order of magnitude (e.g. 1)
        """
        print("I\'ve hardcoded the Gelli+24 sigma(Mh). Change this if it's not what you want.")
        self.data = data
        if type(self.data[0]) is not list:
            self.data = [self.data] # Make sure the format is correct for the rest of the script
        self.zs = [dat[0] for dat in self.data]
        self.parameters = params
        self.lowers = self.parameters.param_dict['lowers']
        self.uppers = self.parameters.param_dict['uppers']
        self.ndim = len(self.parameters.param_dict['base_values'])
        self.nwalkers = 2*self.ndim
        if scale is None:
            self.scale = np.ones_like(self.params1) # No scaling
        else:
            assert len(scale) == len(self.params1), 'You must provide the scale factors for every parameter'
            self.scale = scale
            
        self.MINRELERROR = 0.2
        self.piecewise_eps = piecewise_eps
        self.piecewise_sig = piecewise_sig
        self.time_dependence = time_dependence
        self.mass_dependence = mass_dependence

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
        p0 = self.parameters.base_vals + step_size * (np.random.randn(self.nwalkers, self.ndim))

        # Run a short MCMC to spread the walkers out a bit
        ICs = self.sampler.run_mcmc(p0, 100)
        
        return ICs

    def log_prob(self): 
        """
        Calculate the log probability, taking into account the likelihood & the prior
        Inputs:
            paramvector [1darray]: values of parameters
        Outputs:
            log probability [float]
        """
        paramvector = self.parameters.param_dict['base_vals']
        lprior=self.log_prior()
        if (lprior > -np.inf): #only run if in prior range. avoid weird behavior for negative logs etc
            lpost=self.log_like()
        else:
            lpost = 0.0 #doesn't matter, added to -inf
        #print(paramvector)
        return lprior + lpost

    
    def log_prior(self):
        """
        Calculate the log prior for a flat prior: 0 if within range; negative infinity if outside
        Inputs:
            paramvector [1darray]: values of parameters
        Returns:
            prior [float]: (0 if within range; negative infinity if outside)
        """
        paramvector = self.parameters.param_dict['base_vals']
        if all(lower < t < upper for t, lower, upper in zip(paramvector, self.lowers, self.uppers)):
            return 0.0  # log(1)
        return -np.inf  # log(0)
        
    def log_like(self):
        """
        Calculate log likelihood
        Inputs:
            paramvector [1darray]: values of parameters
            piecewise_eps [1darray]: piecewise value of epsilon at each redshift, if applicable
            piecewise_sig [1darray]: piecewise value of sigma at each redshift, if applicable
        Returns:
            loglike_curr [float]: log likelihood of the data given the model parameters
        """
        paramvector = self.parameters.param_dict['base_vals']
        
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
    
            uvlftheory = self.UVLF_wrapper(zdat,zerr,xdat,xerr, paramvector, piecewise_eps, piecewise_sig)
            
            loglike_curr += -np.sum( (ydat - uvlftheory)**2/(2.0 * yerr**2) ) #assumed Gaussian, to be revisited.
        
        return loglike_curr

    def UVLF_wrapper(self, zcenter, zwidth, MUVcenters, MUVwidths):
        """
        Computes and returns the UVLF at z=zcenters, with width zwidths, in bins centered at MUVcenters with width MUVwidths
        Inputs:
            zcenters [float]: center redshift value for binned UVLF data
            zwdith [float]: width of the redshift bin
            MUVcenters [1darray]: the central UV magnitude in each bin
            MUVwidths [1darray]: the width of the UV magnitude bins
            paramvector [1darray]: values of parameters
            piecewise_eps [1darray]: piecewise value of epsilon at each redshift, if applicable
            piecewise_sig [1darray]: piecewise value of sigma at each redshift, if applicable
        Outputs:
            PhiUV [1darray]: In units of mag^-1 Mpc^-3
        """
        
        astroparams = self.param_wrapper(paramvector, zcenter)
        UVLFs_std = zeus21.UVLFs.UVLF_binned(astroparams,self.CosmoParams,self.HMFintclass,zcenter,zwidth,MUVcenters,MUVwidths)
        
        return UVLFs_std

    def param_wrapper(self, zcenter):
        """
        Puts paramvector into a format that Zeus can read & deals with the time evolution
        Inputs:
            paramvector [1darray]: values of parameters
            zcenter [float]: center redshift value for binned UVLF data
            piecewise_eps [1darray]: piecewise value of epsilon at each redshift, if applicable
            piecewise_sig [1darray]: piecewise value of sigma at each redshift, if applicable
        Outputs:
            astroparams [zeus Astro_Parameters object]: parameters for the UVLF, wrapped so that Zeus can read them
        """
        alphastar, betastar, log10Mcstar, log10epsstar, sigmaUV = [self.parameters.evolve_param(name, zcenter, self.zs, self.HMFintclass.Mhtab) 
                                                                   for name in 
                                                                   ['alphastar', 'betastar', 'log10Mcstar', 'log10epsstar', 'sigmaUV']]
        astroparams = zeus21.Astro_Parameters(self.CosmoParams,epsstar=10**log10epsstar, Mc=10**log10Mcstar,alphastar=alphastar, 
                                              betastar=betastar, sigmaUV = sigmaUV) 
        
        return astroparams

#-------------------------------------------------------------------------------------------------------------------------------------------------

class parameters():
    def __init__(labels, const_val, base_vals, time_derivs, mass_derivs, scales, lowers, uppers):
        self.param_dict = {'labels': labels,
                       'const_val': const_val,
                       'base_vals': [val / scale for val,scale in zip(base_vals, scales)],
                       'time_derivs': tim_derivs,
                       'mass_derivs': mass_derivs,
                       'lowers': lowers,
                       'uppers': uppers
                      }
    
    def evolve_param(self, match_label, z, ztab, Mhtab):
        i = self.param_dict['labels'].index(match_label)

        if self.param_dict['const_val'][i] is not None: # Use this if you want to hold the parameter constant & not MCMC over it.
            return self.param_dict['const_val'][i] 
        
        base_val = self.param_dict['base_vals'][i]

        if np.size(base_val) > 1: # This means that the parameter is defined piecewise
            zindex = np.where(ztab == z)[0]
            return base_val[zindex] # Return the value of the parameter that corresponds to that redshift.

        base_val = base_val + (z-8)*self.param_dict['time_derivs'][i]

        if self.param_dict['mass_derivs'][i] is not None: # Make the parameter an array w/ a value for each halo mass 
            base_val = base_val + (self.param_dict['mass_derivs'][i]*np.log10(Mhtab))

        return base_val
        