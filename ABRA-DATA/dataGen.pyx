import numpy as np
import numpy.linalg as LA
cimport numpy as np
cimport cython
from cython.parallel import parallel, prange
import scipy.special as sc
from cython_gsl cimport gsl_sf_dawson

from distributions import f_SHM, f_GF

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef makePSDFast(double[::1] freqs, double PSDback, double A, double ma, double v0_Halo, double vDotMag_Halo,
                  double alpha_Halo, double tbar_Halo, double v0_Stream, double vDotMag_Stream, double alpha_Stream,
                  double tbar_Stream, double fracStream, double day, int includeGF, int seed):

    set_Seed(seed)


    cdef int N_freqs = len(freqs)
    cdef double[::1] PSD = np.zeros((N_freqs))

    cdef double omega = 2.0*np.pi/365.0

    cdef double vObs_Halo = sqrt(vDotMag_Halo**2 + vEarthMag**2 \
                            + 2* vDotMag_Halo * vEarthMag * alpha_Halo*cos(omega*(day - tbar_Halo)))

    cdef double vObs_Stream = sqrt(vDotMag_Stream**2 + vEarthMag**2 \
                              + 2* vDotMag_Stream * vEarthMag * alpha_Stream*cos(omega*(day - tbar_Stream)))

    cdef Py_ssize_t i
    cdef double exp_mean, freq, v, vSq

    cdef double df = freqs[1] - freqs[0]

    for i in range(N_freqs):
        freq = freqs[i]
        vSq = 2.0*(2.0*pi*freq-ma)/ma
            
        if vSq > 0:
            v = sqrt(vSq)

            if includeGF == 1:
                exp_mean = (1.0-fracStream)*(f_SHM(v, v0_Halo/c, vObs_Halo/c)+c*f_GF(v*c,v0_Halo, day)) \
                            + fracStream*(f_SHM(v, v0_Stream/c, vObs_Stream/c)+c*f_GF(v*c,v0_Stream, day))
                exp_mean = exp_mean * A * pi / ma / v + PSDback

            else:
                exp_mean = (1.0-fracStream)*(f_SHM(v, v0_Halo/c, vObs_Halo/c)) \
                            + fracStream*(f_SHM(v, v0_Stream/c, vObs_Stream/c))
                exp_mean = exp_mean * A * pi / ma / v + PSDback

        else:
            exp_mean = PSDback

        PSD[i] = next_exp_rand(exp_mean)

    return PSD


###########################################
###    Definitions, Necessary Methods   ###
###########################################

DTYPE = np.float
ctypedef np.float_t DTYPE_t

cdef extern from "math.h":
    double cos(double x) nogil
    double sqrt(double x) nogil

cdef extern from "gsl/gsl_cdf.h":
    double gsl_cdf_gaussian_Pinv(double P, double sigma) nogil

from libc.stdlib cimport rand, RAND_MAX
cdef double RAND_SCALE = 1.0/RAND_MAX

cdef inline double next_rand() nogil:
    return rand()*RAND_SCALE

cdef extern from "gsl/gsl_rng.h":
    ctypedef struct gsl_rng_type
    ctypedef struct gsl_rng

    gsl_rng_type *gsl_rng_mt19937
    gsl_rng *gsl_rng_alloc(gsl_rng_type * T) nogil
    void gsl_rng_set(gsl_rng *r, int s) nogil

cdef extern from "gsl/gsl_randist.h":
    double gsl_ran_exponential(gsl_rng *r, double) nogil

cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)

cdef inline double next_exp_rand(double mean) nogil:
    return gsl_ran_exponential(r, mean)

cdef inline void set_Seed(int seed) nogil:
    gsl_rng_set(r, seed)

## Physical Constants
cdef double pi = np.pi
cdef double c = 299792.458
cdef double vEarthMag = 29.79 # Earth's speed km/s
cdef double omega = 2. * np.pi / 365. # angular velocity rad/s
