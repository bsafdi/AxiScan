###############################################################################
# likelihoods.pxd
###############################################################################
#
# Class to calculate the log-likelihood of an observed
# data set given model parameters
#
###############################################################################

# Import basic functions
import numpy as np
cimport numpy as np
cimport cython
cimport phase_space_distribution


# C math functions
cdef extern from "math.h":
    double log(double x) nogil
    double sqrt(double x) nogil

# Useful constants
cdef double pi = 3.1415927
cdef double c = 299792.458 # speed of light [km/s]

cdef int getIndex(double[::1] freqs, double target) nogil

cdef double stacked_likelihood(double[::1] freqs, double[::1] PSD, double mass, double A, double v0,
                    double vObs, double PSDback, double num_stacked) nogil

cdef double SHM_AnnualMod_Likelihood(double[::1] freqs, double[:, ::1] PSD, double mass, double A,
                                     double v0, double vDotMag, double alpha, double tbar,
                                     double PSDback, double num_stacked) nogil

cdef double Substructure_AnnualMod_Likelihood(double[::1] freqs, double[:, ::1] PSD, double mass, double A,
                                              double v0_Halo, double vDotMag_Halo, double alpha_Halo, double tbar_Halo,
                                              double v0_Sub, double vDotMag_Sub, double alpha_Sub, double tbar_Sub,
                                              double frac_Sub, double PSDback, double num_stacked) nogil
