def UVLF_binned(Astro_Parameters,Cosmo_Parameters,HMF_interpolator, zcenter, zwidth, MUVcenters, MUVwidths, DUST_FLAG=True, RETURNBIAS = False):
    'Binned UVLF in units of 1/Mpc^3/mag, for bins at <zcenter> with a Gaussian width zwidth, centered at MUV centers with tophat width MUVwidths. z width only in HMF since that varies the most rapidly. If flag RETURNBIAS set to true it returns number-avgd bias instead of UVLF, still have to divide by UVLF'
    print('working')


    if(constants.NZ_TOINT>1):
        DZ_TOINT = np.linspace(-np.sqrt(constants.NZ_TOINT/3.),np.sqrt(constants.NZ_TOINT/3.),constants.NZ_TOINT) #in sigmas around zcenter
    else:
        DZ_TOINT = np.array([0.0])
    WEIGHTS_TOINT = np.exp(-DZ_TOINT**2/2.)/np.sum(np.exp(-DZ_TOINT**2/2.)) #assumed Gaussian in z, fair



    
    SFRlist = SFR_II(Astro_Parameters,Cosmo_Parameters,HMF_interpolator, HMF_interpolator.Mhtab, zcenter, zcenter)
    sigmaUV = Astro_Parameters.sigmaUV
    
    if (constants.FLAG_RENORMALIZE_LUV == True): #lower the LUV (or SFR) to recover the true avg, not log-avg
        SFRlist/= np.exp((np.log(10)/2.5*sigmaUV)**2/2.0)
        
    MUVbarlist = MUV_of_SFR(SFRlist, Astro_Parameters._kappaUV) #avg for each Mh
    MUVbarlist = np.fmin(MUVbarlist,constants._MAGMAX)
    

    if(RETURNBIAS==True): # weight by bias
        biasM = np.array([bias_Tinker(Cosmo_Parameters, HMF_interpolator.sigma_int(HMF_interpolator.Mhtab,zcenter+dz*zwidth)) for dz in DZ_TOINT])
    else: # do not weight by bias
        biasM = np.ones_like(WEIGHTS_TOINT)
 
        
    HMFtab = np.array([HMF_interpolator.HMF_int(HMF_interpolator.Mhtab,zcenter+dz*zwidth) for dz in DZ_TOINT])
    HMFcurr = np.sum(WEIGHTS_TOINT * HMFtab.T * biasM.T,axis=1)

    #cannot directly 'dust' the theory since the properties of the IRX-beta relation are calibrated on observed MUV. Recursion instead:
    currMUV = MUVbarlist
    if(DUST_FLAG==True):
        currMUV2 = np.ones_like(currMUV)
        while(np.sum(np.abs((currMUV2-currMUV)/currMUV)) > 0.02):
            currMUV2 = currMUV
            currMUV = MUVbarlist + AUV(Astro_Parameters,zcenter,currMUV)
           
    
    MUVcuthi = MUVcenters +  MUVwidths/2.
    MUVcutlo = MUVcenters -  MUVwidths/2.
    
    xhi = np.subtract.outer(MUVcuthi, currMUV)/(np.sqrt(2) * sigmaUV)
    xlo = np.subtract.outer(MUVcutlo, currMUV )/(np.sqrt(2) * sigmaUV)
    weights = (erf(xhi) - erf(xlo)).T/(2.0 * MUVwidths)
    
    UVLF_filtered = np.trapz(weights.T * HMFcurr, HMF_interpolator.Mhtab, axis=-1)

    if(Astro_Parameters.USE_POPIII==False):
        return UVLF_filtered
    else:
        _J21interptemp = interp1d(np.linspace(0,100,3), np.zeros(3), kind = 'linear', bounds_error = False, fill_value = 0,) #TODO: how to deal with J21, requires running get_21_coefficients
        SFRlist_III = SFR_III(Astro_Parameters, Cosmo_Parameters, HMF_interpolator, HMF_interpolator.Mhtab, _J21interptemp, zcenter, zcenter, Cosmo_Parameters.vcb_avg)
    
        MUVbarlist_III = MUV_of_SFR(SFRlist_III, Astro_Parameters._kappaUV_III) #avg for each Mh
        MUVbarlist_III = np.fmin(MUVbarlist_III,constants._MAGMAX)
          
        #and the same for popIII, TODO: ignore dust for pop3 for now
        xhi = np.subtract.outer(MUVcuthi, MUVbarlist_III)/(np.sqrt(2) * sigmaUV)
        xlo = np.subtract.outer(MUVcutlo, MUVbarlist_III)/(np.sqrt(2) * sigmaUV)
        weights = (erf(xhi) - erf(xlo)).T/(2.0 * MUVwidths)

        UVLF_filtered_III = np.trapz(weights.T * HMFcurr, HMF_interpolator.Mhtab, axis=-1)
    
        return UVLF_filtered, UVLF_filtered_III
