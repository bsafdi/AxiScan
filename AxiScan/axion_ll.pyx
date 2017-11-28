###############################################################################
# axion_ll.pyx
###############################################################################
#
# Class to calculate the log-likelihood of an observed data set given model 
# parameters
#
###############################################################################


# Import basic functions
import numpy as np
cimport numpy as np
cimport cython
from .speed_dist cimport get_vObs
from .speed_dist cimport f_SHM

# C math functions
cdef extern from "math.h":
    double pow(double x, double y) nogil
    double log(double x) nogil
    double sqrt(double x) nogil

# Physical Constants
cdef double pi = np.pi 
cdef double c = 299792.458 # speed of light [km/s]


######################
# External functions #
######################

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double stacked_ll(double[::1] freqs, double[::1] PSD, double mass,
                       double A, double v0, double vObs, double lambdaB,
                       double num_stacked) nogil:
    """ Stacked log likelihood for a given PSD dataset and model params

    :param freqs: frequencies scanned over [Hz]
    :param PSD: power spectral density data at those frequencies [Wb^2/Hz]
    :param mass: axion mass [angular frequency Hz]
    :param A: signal strength parameters [Wb^2]
    :param v0: velocity dispersion of SHM [km/s]
    :param vObs: lab/observer/Earth speed w.r.t. the galactic frame [km/s]
    :param lambdaB: mean background noise [Wb^2/Hz]
    :param num_stacked: number of stackings

    :returns: log likelihood (ll)
    """

    # Set up length of input data and output variable
    cdef int N_freqs = freqs.shape[0]
    cdef double ll = 0.0

    # Set up loop variables
    cdef double v, lambdaK
    # Scan from the frequency of mass up to some value well above the peak
    # of the velocity distribution
    cdef double fmin = mass / 2. / pi
    cdef double fmax = fmin * (1+3*(vObs + v0)**2 / c**2)

    cdef int fmin_Index = getIndex(freqs, fmin)
    cdef int fmax_Index = getIndex(freqs, fmax)

    cdef Py_ssize_t ifrq
    for ifrq in range(fmin_Index, fmax_Index):
        v = sqrt(2. * (2.*pi*freqs[ifrq]-mass) / mass) 
        lambdaK = A * pi * c*f_SHM(v*c, v0, vObs) / mass / v + lambdaB

        ll += -PSD[ifrq] / lambdaK - log(lambdaK)

    return ll * num_stacked


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double SHM_AnnualMod_ll(double[::1] freqs, double[:, ::1] PSD, double mass,
                             double A, double v0, double vDotMag, double alpha,
                             double tbar, double lambdaB,
                             double num_stacked) nogil:
    """ log likelihood with annual modulation for a given PSD dataset and 
        model params

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

    # Set up length of input data and output variable
    cdef int N_freqs = freqs.shape[0]
    cdef int N_days = PSD.shape[0]
    cdef double ll = 0.0

    # Set up loop variables
    cdef double v, vObs, lambdaK
    # Scan from the frequency of mass up to some value well above the peak
    # of the velocity distribution
    cdef double fmin = mass / 2. / pi
    cdef double fmax = fmin * (1+3*(vDotMag + v0)**2 / c**2)

    cdef int fmin_Index = getIndex(freqs, fmin)
    cdef int fmax_Index = getIndex(freqs, fmax)

    cdef Py_ssize_t ifrq, iDay

    for iDay in range(N_days):
        vObs = get_vObs(vDotMag, alpha, tbar, iDay)
       
        for ifrq in range(fmin_Index, fmax_Index):
            v = sqrt(2. * (2.*pi*freqs[ifrq]-mass) / mass)    
            lambdaK  = A * pi*c * f_SHM(v*c, v0, vObs) / mass / v + lambdaB

            ll += -PSD[iDay, ifrq] / lambdaK - log(lambdaK)

    return ll * num_stacked


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double Sub_AnnualMod_ll(double[::1] freqs, double[:, ::1] PSD, double mass,
                             double A, double v0_Halo, double vDotMag_Halo,
                             double alpha_Halo, double tbar_Halo, double v0_Sub,
                             double vDotMag_Sub, double alpha_Sub,
                             double tbar_Sub, double frac_Sub, double lambdaB,
                             double num_stacked) nogil:
    """ log likelihood with annual modulation and substructure for a given PSD 
        dataset and model params

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

    # Set up length of input data and output variable
    cdef int N_freqs = freqs.shape[0]
    cdef int N_days = PSD.shape[0]
    cdef double ll = 0.0

    # Set up loop variables
    cdef double v, vObs_Halo, vObs_Sub, lambdaK
    # Scan from the frequency of mass up to some value well above the peak
    # of the velocity distribution
    cdef double fmin = mass / 2. / pi
    cdef double fmax = fmin * (1+3*(vDotMag_Halo + v0_Halo)**2 / c**2)

    cdef int fmin_Index = getIndex(freqs, fmin)
    cdef int fmax_Index = getIndex(freqs, fmax)

    cdef Py_ssize_t ifrq, iDay

    for iDay in range(N_days):
        vObs_Halo = get_vObs(vDotMag_Halo, alpha_Halo, tbar_Halo, iDay)
        vObs_Sub = get_vObs(vDotMag_Sub, alpha_Sub, tbar_Sub, iDay)
       
        for ifrq in range(fmin_Index, fmax_Index):
            v = sqrt(2.0*(2.0*pi*freqs[ifrq]-mass)/ mass)    
            lambdaK = (1-frac_Sub) * A * pi * c * \
                      f_SHM(c*v, v0_Halo, vObs_Halo) / mass / v 
            lambdaK += frac_Sub * A * pi * c *  \
                       f_SHM(c*v, v0_Sub, vObs_Sub) / mass / v
            lambdaK += lambdaB

            ll += -PSD[iDay, ifrq] / lambdaK - log(lambdaK)

    return ll * num_stacked


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef int getIndex(double[::1] freqs, double target) nogil:
    """ Sort through an ordered array of frequencies (freqs) to find the
        nearest value to a specefic f (target)
    """
    cdef int N_freqs = freqs.shape[0]
    cdef Py_ssize_t i

    if freqs[0] > target:
        return 0

    for i in range(N_freqs-1):
        if freqs[i] <= target and freqs[i+1] > target:
            return i+1


    return N_freqs-1
