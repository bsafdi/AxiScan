###############################################################################
# analysis_utilities.py
###############################################################################
#
# Two useful functions used in the example notebook
#
###############################################################################


# Import basic functions
import numpy as np
import scipy.stats
import scipy.special


def getSigma_A(mass, num_stacked, collectionTime, v0, vObs, lambdaB):
    """ Calculate the uncertainty on A associated with a given set of parameters
        assuming the Asimov dataset and a standard halo model

    :param mass: axion mass [angular frequency Hz]
    :param num_stacked: number of stackings
    :param collectionTime: time of individual scan [s]
    :param v0: velocity dispersion of SHM [km/s]
    :param vObs: lab/observer/Earth speed w.r.t. the galactic frame [km/s]
    :param lambdaB: mean background noise [Wb^2/Hz]

    :returns: standard deviation of A
    """

    # Convert velocities to natural units
    c = 299792.458 # speed of light [km/s]
    v0 /= c
    vObs /= c

    # Break the calculation into three factors
    factor1 = num_stacked * collectionTime * np.pi / (2. * mass * lambdaB**2.)
    factor2 = scipy.special.erf(np.sqrt(2.) * vObs / v0)
    factor3 = 1. / (np.sqrt(2.*np.pi) * v0 * vObs)

    return 1. / np.sqrt(factor1 * factor2 * factor3)


def zScore(N):
    """ Appropriate factor for the N-sigma confidence limit derived using the
        Asimov dataset
    """

    return scipy.stats.norm.ppf(0.95)+N
