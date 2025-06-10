# Purpose: useful statistical calculations
# Author: Emily Bregou
# Depends on: numpy

import numpy as np

def calc_log_error(dat, err):
    """
    Propagating error into logspace. See notes from 5/8 for the derivation
    Parameters:
        dat [1darray]: data in linear space
        err [1darray]: error in linear space
    """
    return err/(np.log(10)*dat)