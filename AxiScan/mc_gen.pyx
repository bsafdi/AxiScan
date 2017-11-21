###############################################################################
# mc_gen.pyx
###############################################################################
#
# Class to generate data for analysis
#
###############################################################################


import numpy as np
cimport numpy as np
cimport cython
cimport speed_dist as sd

# Import basic functions
cdef extern from "math.h":
    double log(double x) nogil
    double exp(double x) nogil
    double pow(double x, double y) nogil
    double cos(double x) nogil
    double sin(double x) nogil
    double sqrt(double x) nogil
    double fmax(double x, double y) nogil


cdef extern from "gsl/gsl_cdf.h":
    double gsl_cdf_gaussian_Pinv(double P, double sigma) nogil

from libc.stdlib cimport rand, RAND_MAX
cdef double RAND_SCALE = 1.0/RAND_MAX

cdef inline double next_rand() nogil:
    return rand()*RAND_SCALE

def rand_outer():
    return next_rand()

cdef extern from "gsl/gsl_rng.h":
    ctypedef struct gsl_rng_type
    ctypedef struct gsl_rng

    gsl_rng_type *gsl_rng_mt19937
    gsl_rng *gsl_rng_alloc(gsl_rng_type * T) nogil
    void gsl_rng_set(gsl_rng *r, int s) nogil

cdef extern from "gsl/gsl_randist.h":
    double gsl_ran_exponential(gsl_rng *r, double) nogil
    double gsl_ran_gamma(gsl_rng *r, double a, double b) nogil

cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)

cdef inline double next_exp_rand(double mean) nogil:
    return gsl_ran_exponential(r, mean)


cdef inline double next_gamma_rand(double a, double b) nogil:
    return gsl_ran_gamma(r, a, b)

cdef inline void setSeed(int seed) nogil:
    gsl_rng_set(r, seed)

def setSeed_Outer(seed):
    setSeed(seed)

cdef double pi = np.pi
cdef double c = 299798.452



class Generator:

    def __init__(self, ma, A, PSDback, v0_Halo, vDotMag_Halo, alpha_Halo, tbar_Halo,
                 v0_Stream, vDotMag_Stream, alpha_Stream, tbar_Stream, fracStream,
                 freqs, threads = 1,seed = 0):
            self.ma = ma
            self.A = A
            self.PSDback = PSDback

            self.v0_Halo = v0_Halo
            self.vDotMag_Halo = vDotMag_Halo
            self.alpha_Halo = alpha_Halo
            self.tbar_Halo = tbar_Halo

            self.v0_Stream = v0_Stream
            self.vDotMag_Stream = vDotMag_Stream
            self.alpha_Stream = alpha_Stream
            self.tbar_Stream = tbar_Stream
            self.fracStream = fracStream

            self.freqs = freqs
            self.threads = threads

            self.num_stacked = 86400.0 * (freqs[1] - freqs[0])
            setSeed_Outer(np.random.randint(1e5))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    def makePSDFast(self, double day):

        cdef double num_stacked = self.num_stacked

        cdef double[::1] freqs = self.freqs
        cdef int N_freqs = len(self.freqs)
        cdef double[::1] PSD = np.zeros((N_freqs))

        cdef double A = self.A
        cdef double ma = self.ma
        cdef double v0_Halo = self.v0_Halo
        cdef double vObs_Halo = sd.get_vObs(self.vDotMag_Halo, self.alpha_Halo, self.tbar_Halo, day)
        cdef double v0_Stream = self.v0_Stream
        cdef double vObs_Stream = sd.get_vObs(self.vDotMag_Stream, self.alpha_Stream, self.tbar_Stream, day)
        cdef double fracStream = self.fracStream

        cdef double PSDback = self.PSDback
        cdef Py_ssize_t i
        cdef double exp_mean, freq, v, vSq

        cdef double df = freqs[1] - freqs[0]

        with nogil:
            for i in range(N_freqs):
                freq = freqs[i]
                vSq = 2.0*(2.0*pi*freq-ma)/ma

                if vSq > 0:
                    v = sqrt(vSq)
                    exp_mean = A*pi*(1.0-fracStream)*(sd.f_SHM(v, v0_Halo/c, vObs_Halo/c)) / ma / v \
                               + A*pi*fracStream*(sd.f_SHM(v, v0_Stream/c, vObs_Stream/c)) / ma / v \
                               + PSDback

                else:
                    exp_mean = PSDback

                PSD[i] = next_gamma_rand(num_stacked, exp_mean) / num_stacked

    
        return PSD
