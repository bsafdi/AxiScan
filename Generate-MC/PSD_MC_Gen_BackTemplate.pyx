###############################################################################
# PSD_MC_Gen_BackTemplate.pyx
###############################################################################
#
# Generate ABRACADABRA Monte Carlo for some signal and background
# Output is the power spectral density or PSD
# This version assumes we have a template for the background
#
###############################################################################


# Import basic functions
import numpy as np
cimport numpy as np
cimport cython 

# Useful constants
cdef double pi = np.pi
cdef double c = 299792.458 # speed of light [km/s]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef PSD_gen(double[::1] freqs, double ma, double A, double[::1] PSDback, 
              double v0, double vObs, int num_stacked, int seed):
    """ Generate ABRACADABRA Monte Carlo in the form PSDs, as a function of the
        following parameters:
          - freqs: array of frequencies to calculate the PSD at [Hz]
          - ma: ma/2pi is the frequency associated with the axion mass [Hz]
          - A: shorthand for the strength of the axion PSD;
            A = (gagg * Bmax * VB * alpha)^2 * rhoDM * (L/Lp) / 4
          - PSDback: the mean expected background PSD in each freq bin
          - v0: velocity dispersion (=sqrt(2) sigma_v), nominally 220 km/s
          - vObs: velocity of the Sun in the Milky Way frame, nominally 232 km/s
          - num_stacked: the number of subintervals over which the data is stacked
            each of length Delta T = total T / num_stacked
          - seed: an integer to seed the random number generation
    """

    # Seed random number generator
    setSeed(seed)

    cdef int N_freqs = len(freqs)
    cdef double[::1] PSD = np.zeros(N_freqs)
    cdef Py_ssize_t i
    cdef double exp_mean, v, vSq
    cdef float num_stackedf = float(num_stacked)

    with nogil:
        for i in range(N_freqs):
            
            exp_mean = PSDback[i] 
            vSq = 2.0 * (2.0*pi*freqs[i]-ma) / ma
            if vSq > 0:
                v = sqrt(vSq) # [in natural units]
                # Evaluate SHM using velocities in natural units
                exp_mean += A * pi * fSHM(v, v0/c, vObs/c) / ma / v

            PSD[i] = next_gamma_rand(num_stackedf, exp_mean/num_stackedf)
	
    return np.array(PSD)


########################
# Additional Functions #
########################

# C math functions
cdef extern from "math.h":
    double exp(double x) nogil
    double pow(double x, double y) nogil
    double sqrt(double x) nogil


# Random number generation from the Gamma distribution, which contains the
# Erlang distribution as a special case. The Erlang distribution represents
# draws from a sum of independent exponentially distributed random
# variables. When stacking data this is the exact distribution we want to be
# drawing the PSD from

cdef extern from "gsl/gsl_rng.h":
    ctypedef struct gsl_rng_type
    ctypedef struct gsl_rng

    gsl_rng_type *gsl_rng_mt19937
    gsl_rng *gsl_rng_alloc(gsl_rng_type * T) nogil
    void gsl_rng_set(gsl_rng *r, int s) nogil

cdef extern from "gsl/gsl_randist.h":
    double gsl_ran_gamma(gsl_rng *r, double a, double b) nogil

cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)

cpdef inline double next_gamma_rand(double a, double b) nogil:
    return gsl_ran_gamma(r, a, b)

cpdef inline setSeed(int seed):
    gsl_rng_set(r, seed)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double fSHM(double v, double v0, double vObs) nogil:
    """ Standard Halo Model
          - v: velocity to evaluate the SHM at, in natural units
          - v0: velocity dispersion (=sqrt(2) sigma_v), nominally 220 km/s
          - vObs: velocity of the Sun in the Milky Way frame, nominally 232 km/s
        All velocities should be input in natural units
    """

    cdef double norm = 1. / (sqrt(pi) * v0 *vObs)
    cdef double f = norm * v * (exp(- pow(v-vObs, 2.) / pow(v0, 2.))
                              - exp(- pow(v+vObs, 2.) / pow(v0, 2.)) )
    return f
