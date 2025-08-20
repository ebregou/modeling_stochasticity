# Purpose: handle UVLF data before running MCMCs on it
# Author: Emily Bregou

# Standard packages
import numpy as np

def decompose(data, MINRELERROR = 0.2):
    """
    Decompose data corresponding to a single redshift into its components, ensure that the error bars are reasonable
    Inputs:
        data [Nx6 array]: data corresponding to a single redshift
        MINRELERROR [float]: minimum relative error (prevents the error bars from being unrealistically small)
    Outputs:
        zdat, zerr, xdat, ydat, modified yerr_upper, yerr_lower, xerr
    """
    zdat = data[0] # redshift
    zerr = data[1] # delta redshift (redshift bin)
    xdat = data[2] # MUV
    ydat = data[3]# phiUV 
    yerr_upper = data[4]
    yerr_lower = data[5]
    xerr = data[6] # error on MUV (MUV bins)

    yerr_upper, yerr_lower = np.fmax(yerr_upper, ydat * MINRELERROR), np.fmax(yerr_lower, ydat*MINRELERROR) # Make sure error bars aren't any
                                                                                                            # smaller than the relative error

    return zdat, zerr, xdat, ydat, yerr_upper, yerr_lower, xerr

def get_sorted(file_names, data_labels = None, include_zs = None): 
    """
    Read in data and organize it by dataset and redshift. We want the data sorted by dataset for plotting purposes (so we can assign credit to the
    studies that observed different things). 
    Inputs:
        file_names [list of strs]: file names to read data from
        data_labels [list of strs]: names of datasets, for plotting purposes
        include_zs [list of floats]: optional, which redshifts to include in the output data
    Returns:
        data, organized by dataset and redshift
    """
    if data_labels is None:
        data_labels = np.full('', len(file_names))
        
    all_data = []
    for fn in file_names:
        all_data.append(np.loadtxt(fn, unpack = True))
                        
    redshifts = np.unique(np.concatenate([data[0] for data in all_data])) # Get a list of all redshifts that exist within the files
    if include_zs is None: # Include all redshifts if none are specified
        include_zs = redshifts
    dredshifts = np.ones_like(redshifts)/2. #approximate, there are true window functions to use

    # Organize data by redshift
    sorted_data = []
    #format is     zdat = data[0] zerr = data[1] xdat = data[2],  ydat = data[3]  yerr_upper = data[4]  yerr_lower = data [5] xerr = data[6] 
    
    for iz,z in enumerate(redshifts):
        if z not in include_zs:
            print(f'Skipping $z = {z}')
            continue
        z_separated = []
        for data, label in zip(all_data, data_labels):
            zlistindex = data[0] == 1.0*z
            datarr = [z,dredshifts[iz], data[1][zlistindex], data[3][zlistindex], data[4][zlistindex], np.abs(data[5][zlistindex]), 
                      #Use absolute value because the lower error bar is given as negative (centered on ydat)
                      data[2][zlistindex], label]
            z_separated.append(datarr)
        sorted_data.append(z_separated)

    return sorted_data

def reduce(sorted_data): 
    """
    Remove the sorting by dataset and just sort by redshift. Make sure the data go in order of MUV. The MCMC doesn't care where the data comes
    from, so this is for that.
    Inputs:
        sorted_data [list]: from get_sorted, data sorted by dataset and redshift.
    Returns:
        reduced_data [list]: data sorted by just redshift, ordered by MUV
    """
    reduced_data = []
    for zbin in sorted_data:
        z = zbin[0][0]
        dz = zbin[0][1]
        MUVs = np.concatenate([dat[2]for dat in zbin])
        sorting = np.argsort(MUVs)
        MUVs = MUVs[sorting]
        phis = np.concatenate([dat[3] for dat in zbin])[sorting]
        yerr_upper = np.concatenate([dat[4] for dat in zbin])[sorting]
        yerr_lower = np.concatenate([dat[5] for dat in zbin])[sorting]
        dMUV = np.concatenate([dat[6] for dat in zbin])[sorting]
        reduced_data.append([z, dz, MUVs, phis, yerr_upper, yerr_lower, dMUV])


    return reduced_data

def save(redshifts, magnitudes, dmag, phi, plus_sig, minus_sig, file_name):
    """
    Save a file in the correct format for reading
    Inputs:
        redshifts [1darray]
        magnitudes [1darray]
        dmag [1darray]: width of magnitude bin
        phi [1darray]: phi_UV in units of mag^-1 Mpc^-3
        plus_sig [1darray]: uppper error bar on phi_UV, same units
        minus_sig [1darray]: lower error bar on phi_UV, same units
        file_name [str]: complete path to where you want the file saved
    Outputs:
        Saved file under the given directory
    """
    data = np.hstack((np.array(redshifts).reshape(-1,1), np.array(magnitudes).reshape(-1,1), np.array(dmag).reshape(-1,1),
                      np.array(phi).reshape(-1,1), np.array(plus_sig).reshape(-1,1), np.array(minus_sig).reshape(-1,1)))
    np.savetxt(file_name, data, fmt = '%.4e', delimeter = '     ')

    return