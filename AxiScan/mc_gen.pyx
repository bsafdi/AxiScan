###############################################################################
# mc_gen.pyx
###############################################################################
#
# Class to generate data for analysis
#
###############################################################################


# Import basic functions
import numpy as np
cimport numpy as np
cimport cython
from .speed_dist cimport get_vObs
from .speed_dist cimport f_SHM

# Physical Constants
cdef double pi = np.pi
cdef double c = 299792.458 # speed of light [km/s]

# C math functions
cdef extern from "math.h":
    double sqrt(double x) nogil


# Load C random draw from a gamma distributions function
# Also setup ability to set the random number generation seed
cdef extern from "gsl/gsl_rng.h":
    ctypedef struct gsl_rng_type
    ctypedef struct gsl_rng

    gsl_rng_type *gsl_rng_mt19937
    gsl_rng *gsl_rng_alloc(gsl_rng_type * T) nogil
    void gsl_rng_set(gsl_rng *r, int s) nogil

cdef extern from "gsl/gsl_randist.h":
    double gsl_ran_gamma(gsl_rng *r, double a, double b) nogil

cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)

cdef inline double next_gamma_rand(double a, double b) nogil:
    return gsl_ran_gamma(r, a, b)

cdef inline void setSeed(int seed) nogil:
    gsl_rng_set(r, seed)

def setSeed_Outer(seed):
    setSeed(seed)


class Generator:
    def __init__(self, mass, A, lambdaB, v0_Halo, vDotMag_Halo, alpha_Halo, 
                 tbar_Halo, v0_Sub, vDotMag_Sub, alpha_Sub, 
                 tbar_Sub, frac_Sub, freqs):
        """ Class to generate fake PSD data

        :param mass: axion mass [angular frequency Hz]
        :param A: signal strength parameters [Wb^2]
        :param lambdaB: mean background noise [Wb^2/Hz]
        Following 4 parameters defined for the Halo (_Halo) 
        and substructure (_Sub)
        :param v0: velocity dispersion of SHM [km/s]
        :param vDotMag: velocity of the sun w.r.t. the galactic frame [km/s]
        :param alpha/tbar: scalar quantities defining direction of vDot
        :param frac_Sub: fraction of local DM in the substructure
        :param lambdaB: mean background noise [Wb^2/Hz]
        """

        self.mass = mass
        self.A = A
        self.lambdaB = lambdaB

        self.v0_Halo = v0_Halo
        self.vDotMag_Halo = vDotMag_Halo
        self.alpha_Halo = alpha_Halo
        self.tbar_Halo = tbar_Halo

        self.v0_Sub = v0_Sub
        self.vDotMag_Sub = vDotMag_Sub
        self.alpha_Sub = alpha_Sub
        self.tbar_Sub = tbar_Sub
        self.frac_Sub = frac_Sub

        self.freqs = freqs

        # 86400 = number of seconds in a day
        self.num_stacked = 86400. * (freqs[1] - freqs[0])
        # Set the seed to a random number each time the class is initiated
        setSeed_Outer(np.random.randint(1e5))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    def makePSD(self, double day):
        """ Make the fake PSD data
        :param day: day with respect to t1, vernal equinox

        :returns: array of power spectral density values in [Wb^2/Hz]
        """

        cdef double num_stacked = self.num_stacked

        cdef double[::1] freqs = self.freqs
        cdef int N_freqs = len(self.freqs)
        cdef double[::1] PSD = np.zeros(N_freqs)

        cdef double A = self.A
        cdef double mass = self.mass
        cdef double v0_Halo = self.v0_Halo
        cdef double vObs_Halo = get_vObs(self.vDotMag_Halo, self.alpha_Halo, 
                                         self.tbar_Halo, day)
        cdef double v0_Sub = self.v0_Sub
        cdef double vObs_Sub = get_vObs(self.vDotMag_Sub, self.alpha_Sub, 
                                        self.tbar_Sub, day)
        cdef double frac_Sub = self.frac_Sub
        cdef double lambdaB = self.lambdaB
        
        cdef Py_ssize_t i
        cdef double exp_mean, freq, v, vSq
        cdef double df = freqs[1] - freqs[0]

        # Calculate the expected mean and then perform a random draw around it
        with nogil:
            for i in range(N_freqs):
                freq = freqs[i]
                vSq = 2.*(2.*pi*freq-mass)/mass

                if vSq > 0:
                    v = sqrt(vSq)
                    exp_mean = A*pi*(1.-frac_Sub)*(f_SHM(v, v0_Halo/c, 
                               vObs_Halo/c)) / mass / v \
                               + A*pi*frac_Sub*(f_SHM(v, v0_Sub/c, 
                               vObs_Sub/c)) / mass / v \
                               + lambdaB

                else:
                    exp_mean = lambdaB

                PSD[i] = next_gamma_rand(num_stacked, exp_mean) / num_stacked
 
        return PSD
