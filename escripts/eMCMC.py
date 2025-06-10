# Purpose: interface with Zeus to run MCMC on models of the UVLF with time-evolving sigma_UV
# Author: Emily Bregou, Julian Muñoz
# Depends on: numpy, zeus21, emcee

# Standard packages
import numpy as np
import zeus21
import emcee

# MCMC class
class UVLF():
    def __init__(self, data, params1, lowers, uppers, time_dependence = False, piecewise_eps = False, piecewise_sig = False, scale = 100, 
                 MINRELERROR = 0.2):
        self.data = data
        self.zs = [dat[0] for dat in self.data]
        self.lowers = lowers
        self.uppers = uppers
        self.params1 = params1
        self.ndim = len(params1)
        self.nwalkers = 2*self.ndim
        self.scale = scale
        self.MINRELERROR = 0.2
        self.piecewise_eps = piecewise_eps
        self.piecewise_sig = piecewise_sig
        self.time_dependence = time_dependence

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
        p0 = self.params1 + step_size * (np.random.randn(self.nwalkers, self.ndim))

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
        
    def log_like(self, paramvector, piecewise_eps = None, piecewise_sig = None):
        """
        Calculate log likelihood
        Inputs:
            paramvector [1darray]: values of parameters
            piecewise_eps [1darray]: piecewise value of epsilon at each redshift, if applicable
            piecewise_sig [1darray]: piecewise value of sigma at each redshift, if applicable
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
    
            uvlftheory = self.UVLF_wrapper(zdat,zerr,xdat,xerr, paramvector, piecewise_eps, piecewise_sig)
            
            loglike_curr += -np.sum( (ydat - uvlftheory)**2/(2.0 * yerr**2) ) #assumed Gaussian, to be revisited.
        
        return loglike_curr

    def UVLF_wrapper(self, zcenter, zwidth, MUVcenters, MUVwidths, paramvector, piecewise_eps = None, piecewise_sig = None):
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
        # Set piecewise flags
        piecewise_eps = self.piecewise_eps if piecewise_eps is None else piecewise_eps
        piecewise_sig = self.piecewise_sig if piecewise_sig is None else piecewise_sig
        
        astroparams = self.param_wrapper(paramvector, zcenter, MUVcenters, piecewise_eps, piecewise_sig)
        UVLFs_std = zeus21.UVLFs.UVLF_binned(astroparams,self.CosmoParams,self.HMFintclass,zcenter,zwidth,MUVcenters,MUVwidths)
        
        return UVLFs_std

    def param_wrapper(self, paramvector, zcenter, MUVcenters, piecewise_eps, piecewise_sig):
        """
        Puts paramvector into a format that Zeus can read & deals with the time evolution
        Inputs:
            paramvector [1darray]: values of parameters
            zcenter [float]: center redshift value for binned UVLF data
            MUVcenters [1darray]: the central UV magnitude in each bin
            piecewise_eps [1darray]: piecewise value of epsilon at each redshift, if applicable
            piecewise_sig [1darray]: piecewise value of sigma at each redshift, if applicable
        Outputs:
            astroparams [zeus Astro_Parameters object]: parameters for the UVLF, wrapped so that Zeus can read them
        """
        log10epsstar, log10Mcstar, alphastar, betastar, sigmaUV = self.time_evolution(paramvector, zcenter, MUVcenters, piecewise_eps, 
                                                                                      piecewise_sig)
        astroparams = zeus21.Astro_Parameters(self.CosmoParams,epsstar=10**log10epsstar, Mc=10**log10Mcstar,alphastar=alphastar, 
                                              betastar=betastar, sigmaUV = sigmaUV) 
        
        return astroparams

    def time_evolution(self, paramvector, zcenter, MUVcenters, piecewise_eps, piecewise_sig):
        """
        Applies the time evolution of each parameter so that we feed the evolved value, matching the given redshift, to the UVLF wrapper
        Inputs:
            paramvector [1darray]: values of parameters
            zcenter [float]: center redshift value for binned UVLF data
            MUVcenters [1darray]: the central UV magnitude in each bin
            piecewise_eps [1darray]: piecewise value of epsilon at each redshift, if applicable
            piecewise_sig [1darray]: piecewise value of sigma at each redshift, if applicable
        Returns:
            [log10epsstar, log10Mcstar, alphastar, betastar, sigmaUV]: values of these parameters that match the given redshift
        """
        if self.time_dependence:
            i = 6 # To keep track of the indices of the input vector when we don't know in advance how many parameters were passed
            # Depends on whether piecewise was used, and if so, how many redshift bins there are
            log10Mcstar_base, dMdz, alphastar_base, dadz, betastar_base, dbdz = paramvector[:i]

            # Apply time evolution
            # Divide by scale so that the d/dz parameters can scale similarly to other parameters
            log10Mcstar = log10Mcstar_base + (dMdz*(zcenter-8)/self.scale) # Center at z = 8 like Muñoz+23
            alphastar = alphastar_base + (dadz*(zcenter-8)/self.scale)
            betastar = betastar_base + (dbdz*(zcenter-8)/self.scale)

        else:
            i = 3
            log10Mcstar, alphastar, betastar = paramvector[:i]

        # Assign values to epsilon & sigma based off of whether we're doing linear or piecewise evolution
        if piecewise_eps:
            log10epsstar_base = np.array(paramvector[i:i+len(self.zs)])
            i += len(self.zs)
            index = np.where(self.zs == zcenter)[0]
            log10epsstar =  log10epsstar_base[index]
        else:
            if self.time_dependence:
                log10epsstar_base = paramvector[i]
                i+=1
                dedz = paramvector[i]
                i+=1
                log10epsstar = log10epsstar_base + (dedz*(zcenter-8)/self.scale)
            else:
                log10epsstar = paramvector[i]
                i+=1

        if piecewise_sig:
            sigmaUV_base = np.array(paramvector[i:i+len(self.zs)])
            i += len(self.zs)
            index = np.where(self.zs == zcenter)[0]
            sigmaUV =  sigmaUV_base[index]
        else: 
            if self.time_dependence:
                sigmaUV_base = paramvector[i]
                i += 1
                dsdz = paramvector[i]
                i += 1
                sigmaUV = sigmaUV_base + (dsdz*(zcenter-8)/self.scale)
            else:
                sigmaUV = paramvector[i]
                i+=1
            dsdMUV = paramvector[i]
            i +=1
            sigmaUV_vec = (sigmaUV + (dsdMUV*(MUVcenters+20)/self.scale)).reshape(-1,1)
        
        return [log10epsstar, log10Mcstar, alphastar, betastar, sigmaUV_vec]