# Purpose: cosmology calculations
# Author: Emily Bregou

from classy import Class
import zeus21


def classy_cosmology(Omega_b = 0.04927, Omega_m = 0.3156, h = 0.6727, k_max = 1e3, z_max = 10):
    """
    Calculate cosmology with Class
    Parameters:
        Omega_b [float]: Omega baryons
        Omega_m [float]: Omega matter (Omega matter >= Omega baryons)
        h [float]: dimensionless Hubble parameter
        k_max [float]: maximum wavenumber (1/Mpc)
        z_max [float]: maximum redshift
    Returns:
        cosmo [Class object]: computed cosmology
    """
    #Initialize the cosmology
    params = {'output': 'mPk',
          'non linear': 'halofit',
          'Omega_b': Omega_b,
          'Omega_cdm': Omega_m - Omega_b,
          'h': h,
          'P_k_max_1/Mpc': k_max,
          'z_max_pk': z_max
    }
    cosmo = Class()
    cosmo.set(params)

    # Compute everything, store it in cosmo
    cosmo.compute()

    return cosmo

def hmf_zeus(params = None, z_min = 0, HMF_CHOICE = 'ST'):
    """
    Initiate Zeus21 & calculate the relevant parameters
    Inputs:
        params [dict]: cosmological parameters with keys including 'Omega_b', 'Omega_m', 'h'
        z_min [float]: minimum redshift
        HMF_CHOICE [str]: choice of 'ST' for Sheth-Tormen or 'Yung' for the Tinker08 form of the HMF
    Returns:
        params[dict]: cosmological parameters with keys including 'Omega_b', 'Omega_m', 'h'
        HMFintclass: class that contains anything you want to do with the HMF, including HMFintclass.HMF_int() which gives dn/dM with units 
                     M_sun^-1*Mpc^-3
    """
    ClassyCosmo, CosmoParams = get_CosmoParams(params, z_min, HMF_CHOICE)
    HMFintclass = zeus21.HMF_interpolator(CosmoParams, ClassyCosmo)
    
    return params, HMFintclass

def get_params():
    """
    Get standard cosmology parameters from Class
    Returns:
        params [dict]: dictionary with keys for Omega_b, Omega_m, Omega_L, and the Hubble parameter, h and the corresponding standard values from
                       Class
    """
    ClassCosmo = Class()
    ClassCosmo.compute()
    params = {'Omega_b': ClassCosmo.Omega_b(),
          'Omega_m': ClassCosmo.Omega_m(),
          'Omega_L': ClassCosmo.Omega_Lambda(),
          'h': 0.6727} # h -> H0/100
    return params

def get_CosmoParams(params = None, z_min = 0, HMF_CHOICE = 'ST'):
    """
    Get CosmoParams in a format Zeus21 can read
    Inputs:
        params [dict]: dictionary with keys for Omega_b, Omega_m, Omega_L, and the Hubble parameter, h and the corresponding standard values from
                       Class
        z_min [float]: minimum redshift
    Outputs:
        ClassyCosmo: ouptut of running Class with the given input parameters
        CosmoParams: cosmological parameters in a format that Zeus21 can read
    """
    
    if params is None:
        params = get_params()
    
    # Zeus takes their parameters scaled by the hubble constant
    h = params['h']
    omegab =  params['Omega_b'] * h**2
    omegac = (params['Omega_m'] - params['Omega_b']) * h**2 # omega_CDM
    CosmoParams_input = zeus21.Cosmo_Parameters_Input(omegab, omegac, h, zmin_CLASS = z_min, HMF_CHOICE = HMF_CHOICE) 

    # Run the cosmology & get parameters out
    ClassyCosmo = zeus21.runclass(CosmoParams_input)
    UserParams = zeus21.User_Parameters()
    CosmoParams = zeus21.Cosmo_Parameters(UserParams, CosmoParams_input, ClassyCosmo)

    return ClassyCosmo, CosmoParams