# Purpose: useful statistical calculations
# Author: Emily Bregou

import numpy as np

def calc_log_error(dat, err):
    """
    Propagating error into logspace (first order expansion). See notes from 5/8 for the derivation
    Parameters:
        dat [1darray]: data in linear space
        err [1darray of tuples]: error in linear space
    """
    return err/(np.log(10)*dat)