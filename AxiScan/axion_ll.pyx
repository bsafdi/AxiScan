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
cimport speed_dist as sd

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
                       double A, double v0, double vObs, double PSDback,
                       double num_stacked) nogil:

    # Set up length of input data and output variable
    cdef int N_freqs = freqs.shape[0]
    cdef double ll = 0.0

    # Set up loop variables
    cdef double v, lambdaK
    # Scan from the frequency of mass up to some value well above the peak
    # of the velocity distribution
    cdef double fmin = mass / 2. / pi
    cdef double fmax = fmin * (3.*(pow(vObs,2.) + pow(v0,2.))/pow(c,2.))

    cdef int fmin_Index = getIndex(freqs, fmin)
    cdef int fmax_Index = getIndex(freqs, fmax)

    cdef Py_ssize_t ifrq

    for ifrq in range(fmin_Index, fmax_Index):
        v = sqrt(2.0*(2.0*pi*freqs[ifrq]-mass)/ mass)        
        lambdaK = A * pi * sd.f_SHM(v, v0, vObs) / mass / v + PSDback

        ll += -PSD[ifrq] / lambdaK - log(lambdaK)

    return ll * num_stacked

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double SHM_AnnualMod_ll(double[::1] freqs, double[:, ::1] PSD, double mass,
                             double A, double v0, double vDotMag, double alpha,
                             double tbar, double PSDback,
                             double num_stacked) nogil:

    # Set up length of input data and output variable
    cdef int N_freqs = freqs.shape[0]
    cdef int N_days = PSD.shape[0]
    cdef double ll = 0.0

    # Set up loop variables
    cdef double v, vObs, lambdaK
    cdef double fmin = mass / 2.0 / pi
    cdef double fmax = fmin * (1.0 * 3.0 * (vDotMag**2 + v0**2)/c**2)

    cdef int fmin_Index = getIndex(freqs, fmin)
    cdef int fmax_Index = getIndex(freqs, fmax)

    cdef Py_ssize_t ifrq, iDay

    for iDay in range(N_days):
        vObs = sd.get_vObs(vDotMag, alpha, tbar, iDay)
       
        for ifrq in range(fmin_Index, fmax_Index):
            v = sqrt(2.0*(2.0*pi*freqs[ifrq]-mass)/ mass)    
            lambdaK  = A * pi * sd.f_SHM(v, v0, vObs) / mass / v + PSDback

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
                             double tbar_Sub, double frac_Sub, double PSDback,
                             double num_stacked) nogil:

    # Set up length of input data and output variable
    cdef int N_freqs = freqs.shape[0]
    cdef int N_days = PSD.shape[0]
    cdef double ll = 0.0

    # Set up loop variables
    cdef double v, vObs_Halo, vObs_Sub, lambdaK
    cdef double fmin = mass / 2.0 / pi
    cdef double fmax = fmin * (1.0 * 3.0 * (vDotMag_Halo**2 + v0_Halo**2)/c**2)

    cdef int fmin_Index = getIndex(freqs, fmin)
    cdef int fmax_Index = getIndex(freqs, fmax)

    cdef Py_ssize_t ifrq, iDay

    for iDay in range(N_days):
        vObs_Halo = sd.get_vObs(vDotMag_Halo, alpha_Halo, tbar_Halo, iDay)
        vObs_Sub = sd.get_vObs(vDotMag_Sub, alpha_Sub, tbar_Sub, iDay)
       
        for ifrq in range(fmin_Index, fmax_Index):
            v = sqrt(2.0*(2.0*pi*freqs[ifrq]-mass)/ mass)    
            lambdaK = (1-frac_Sub)*A * pi * sd.f_SHM(v, v0_Halo, vObs_Halo) / mass / v 
            lambdaK += frac_Sub*A * pi * sd.f_SHM(v, v0_Sub, vObs_Sub) / mass / v
            lambdaK += PSDback

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

    cdef int start = 0
    cdef int end = freqs.shape[0]
    cdef int trial

    while end-start > 1:
        trial = (end-start) / 2 + start
        if target > freqs[trial]:
            start = trial
        else:
            end = trial

    if end == freqs.shape[0]:
        return end -1
    else:
        return end
