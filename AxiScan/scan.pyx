###############################################################################
# scan.pyx
###############################################################################
#
# Evaluate the Test Statistic for ABRACADABRA at a series of input values
#
###############################################################################


# Import basic functions
import numpy as np
cimport numpy as np
cimport cython
from . cimport axion_ll as ll


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef TS_Scan(double[::1] PSD, double[::1] freqs, double[::1] mass_TestSet,
              double[::1] A_TestSet, double lambdaB, double v0, double vObs,
              double num_stacked):
    """ Evalute  the Test Statistic (TS), or more accurately Theta, for a given
        dataset and set of model parameters. From this can determine TS of
        discovery or TS for 95% limits.

    :param PSD: power spectral density data at those frequencies [Wb^2/Hz]
    :param freqs: frequencies scanned over [Hz]
    :param mass_TestSet: array of masses to evaluate Theta at
    :param A_TestSet: array of signal strength values to evaluate Theta at
    :param lambdaB: mean background noise [Wb^2/Hz]
    :param v0: velocity dispersion of SHM [km/s]
    :param vObs: lab/observer/Earth speed w.r.t. the galactic frame [km/s]
    :param num_stacked: number of stackings

    :returns: Theta
    """

    cdef int N_masses = len(mass_TestSet)
    cdef int N_AVals = len(A_TestSet)
    cdef double[:, ::1] Theta_Array = np.zeros((N_masses, N_AVals))
    cdef Py_ssize_t iM, iA

    # Evaluate Theta for every mass and A value
    for iM in range(N_masses):
        for iA in range(N_AVals):
   
            Theta_Array[iM, iA] += ll.stacked_ll(freqs, PSD, mass_TestSet[iM],  
                                                 A_TestSet[iA], v0, vObs, 
                                                 lambdaB, num_stacked)

            Theta_Array[iM, iA] -= ll.stacked_ll(freqs, PSD, mass_TestSet[iM], 
                                                 0, v0, vObs, lambdaB, 
                                                 num_stacked)

            Theta_Array[iM, iA] *= 2. # Theta is 2x the ll

    return Theta_Array


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef double SHM_AnnualMod_ll(double[::1] freqs, double[:, ::1] PSD, 
                              double mass, double A, double v0, double vDotMag, 
                              double alpha, double tbar, double lambdaB, 
                              double num_stacked) nogil:
    """ python wrapper for the annual modulation log likelihood

    :param freqs: frequencies scanned over [Hz]
    :param PSD: power spectral density data at those frequencies [Wb^2/Hz]
    :param mass: axion mass [angular frequency Hz]
    :param A: signal strength parameters [Wb^2]
    :param v0: velocity dispersion of SHM [km/s]
    :param vDotMag: velocity of the sun w.r.t. the galactic frame [km/s]
    :param alpha/tbar: scalar quantities defining direction of vDot
    :param lambdaB: mean background noise [Wb^2/Hz]
    :param num_stacked: number of stackings

    :returns: log likelihood (ll)
    """

    return ll.SHM_AnnualMod_ll(freqs, PSD, mass, A, v0, vDotMag, alpha, tbar, 
                               lambdaB, num_stacked)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef double Sub_AnnualMod_ll(double[::1] freqs, double[:, ::1] PSD, 
                              double mass, double A, double v0_Halo, 
                              double vDotMag_Halo, double alpha_Halo, 
                              double tbar_Halo, double v0_Sub, 
                              double vDotMag_Sub, double alpha_Sub, 
                              double tbar_Sub, double frac_Sub, double lambdaB, 
                              double num_stacked) nogil:
    """ python wrapper for the annual modulation + substructure log likelihood
    
    :param freqs: frequencies scanned over [Hz]
    :param PSD: power spectral density data at those frequencies [Wb^2/Hz]
    :param mass: axion mass [angular frequency Hz]
    :param A: signal strength parameters [Wb^2]
    Following 4 parameters defined for the Halo (_Halo) and substructure (_Sub)
    :param v0: velocity dispersion of SHM [km/s]
    :param vDotMag: velocity of the sun w.r.t. the galactic frame [km/s]
    :param alpha/tbar: scalar quantities defining direction of vDot
    :param frac_Sub: fraction of local DM in the substructure
    :param lambdaB: mean background noise [Wb^2/Hz]
    :param num_stacked: number of stackings

    :returns: log likelihood (ll)
    """

    return ll.Sub_AnnualMod_ll(freqs, PSD, mass, A, v0_Halo, vDotMag_Halo, 
                               alpha_Halo, tbar_Halo, v0_Sub, vDotMag_Sub, 
                               alpha_Sub, tbar_Sub, frac_Sub, lambdaB, 
                               num_stacked)
