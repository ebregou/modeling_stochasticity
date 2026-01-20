# Purpose: cosmology calculations
# Author: Emily Bregou

from classy import Class
import zeus21
import numpy as np
import astropy.units as u


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

def calc_Mz(zs, M0, my_UVLF):
    """
    Calculate the mass of a halo at redshift z given its mass at z = 0.
    Currently this takes a UVLF object as an input because it saves time when you already have cosmological parameters stored. I could make this
    method more flexible in the future.
    Inputs:
        zs [1darray]: list of redshifts to calculate the halo mass for
        M0 [float]: log(M_h/M_odot) at z = 0
        my_UVLF [UVLF object]: UVLF object used for efficiency since it has cosmological parameters stored
    """
    # Initiate
    steps = len(zs)
    Ms = np.zeros(steps+1)
    Ms[0] = 10**M0
    dz = (zs.max() - zs.min())/(steps+1)
    
    for i in range(steps): # Integrate dM/dz to get M(z)
        # Multiply dM/dt by dt/dz to get M_dot
        M_dot = calc_M_dot(Ms[i], zs[i], my_UVLF) /  ((1+zs[i]) * 
                                                      (zeus21.Hub(my_UVLF.CosmoParams, zs[i])*u.km/(u.s*u.Mpc)).to(u.year**-1).value) 
        # Had to take out a negative here to make it work; not sure why. Normally dt/dz would include a -(1+z)
        Ms[i+1] = Ms[i] - (M_dot * dz)

    return np.log10(Ms)

def calc_M_dot(M_h, z, my_UVLF):
    """
    Calculate the mass accretion rate of a halo given redshift & the mass at that redshift
    Inputs:
        M_h [float]: Mass of the halo at redshift z [solar masses]
        z [float]: redshift
        my_UVLF [UVLF object]: UVLF object used for efficiency since it has cosmological parameters stored
    Returns:
        Mh_dot [solar masses per year]
    """
    return 25.3*(M_h/1e12)**(1.1)*(1+(1.65*z))*np.sqrt((my_UVLF.CosmoParams.OmegaM*(1+z)**3)+my_UVLF.CosmoParams.OmegaL)