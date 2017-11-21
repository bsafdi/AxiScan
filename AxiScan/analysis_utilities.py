import numpy as np
import scipy
import scipy.stats
import scipy.special

c= 299798.452

def getSigma_A(mass, num_stacked, collectionTime, v0, vObs, PSDback):
    """ Calculate the uncertainty on A associated with a given set of parameters
        assuming the Asimov dataset and a standard halo model. Parameters are
        as for solveForA
    """

    # Convert velocities to natural units
    v0 /= c
    vObs /= c

    # Break the calculation into three factors
    factor1 = num_stacked * collectionTime * np.pi / (2. * mass * PSDback**2.)
    factor2 = scipy.special.erf(np.sqrt(2.) * vObs / v0)
    factor3 = 1. / (np.sqrt(2.*np.pi) * v0 * vObs)

    return 1. / np.sqrt(factor1 * factor2 * factor3)


def zScore(N):
    """ Appropriate factor for the N-sigma confidence limit derived using the
        Asimov dataset
    """

    alpha = .05
    return scipy.stats.norm.ppf(1-alpha)+N

def find_nearest_index(array,value):
    idx = np.nanargmin(np.abs(array-value))
    return idx

