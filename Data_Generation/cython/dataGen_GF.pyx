import numpy as np
import numpy.linalg as LA
cimport numpy as np
cimport cython
from cython.parallel import parallel, prange
import scipy.special as sc



#################################
###   Data Generation Class   ###
#################################
class Generator:

    def __init__(self, ma, A, PSDback, v0_Halo, vDotMag_Halo, alpha_Halo, tbar_Halo,
                 v0_Stream, vDotMag_Stream, alpha_Stream, tbar_Stream, fracStream,
                 freqs, seed = 0):
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
        setSeed_Outer(np.random.randint(1e5))


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    def makePSDFast(self, double day, int includeGF):
        
        cdef double PSDback = self.PSDback
        cdef double[::1] freqs = self.freqs
        cdef int N_freqs = len(self.freqs)
        cdef double[::1] PSD = np.zeros((N_freqs))

        cdef double omega = 2.0*np.pi/365.0

        cdef double A = self.A
        cdef double ma = self.ma
        cdef double v0_Halo = self.v0_Halo
        cdef double vObs_Halo = sqrt(self.vDotMag_Halo**2 + vEarthMag**2 \
                                + 2* self.vDotMag_Halo * vEarthMag \
                                * self.alpha_Halo*cos(omega*(day - self.tbar_Halo)))

        cdef double v0_Stream = self.v0_Stream
        cdef double vObs_Stream = sqrt(self.vDotMag_Stream**2 + vEarthMag**2 \
                                  + 2* self.vDotMag_Stream * vEarthMag \
                                  * self.alpha_Stream*cos(omega*(day - self.tbar_Stream)))
        cdef double fracStream = self.fracStream


        cdef Py_ssize_t i
        cdef double exp_mean, freq, v, vSq

        cdef double df = freqs[1] - freqs[0]

        for i in range(N_freqs):
            freq = freqs[i]
            vSq = 2.0*(2.0*pi*freq-ma)/ma
            
            if vSq > 0:
                v = sqrt(vSq)

                if includeGF == 1:
                    exp_mean = (1.0-fracStream)*(fHalo(v, v0_Halo/c, vObs_Halo/c)+c*f1(v*c,v0_Halo, day)) \
                                + fracStream*(fHalo(v, v0_Stream/c, vObs_Stream/c)+c*f1(v*c,v0_Stream, day))
                    exp_mean = exp_mean * A * pi / ma / v + PSDback

                else:
                    exp_mean = (1.0-fracStream)*(fHalo(v, v0_Halo/c, vObs_Halo/c)) \
                                + fracStream*(fHalo(v, v0_Stream/c, vObs_Stream/c))
                    exp_mean = exp_mean * A * pi / ma / v + PSDback

            else:
                exp_mean = PSDback

            PSD[i] = next_exp_rand(exp_mean)
            self.PSDArray = np.array(PSD)

###########################################
###    Definitions, Necessary Methods   ###
###########################################

DTYPE = np.float
ctypedef np.float_t DTYPE_t

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

cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)

cdef inline double next_exp_rand(double mean) nogil:
    return gsl_ran_exponential(r, mean)

def next_exp_rand_outer(mean):
    return next_exp_rand(mean)

cdef inline void setSeed(int seed) nogil:
    gsl_rng_set(r, seed)

def setSeed_Outer(seed):
    setSeed(seed)


## Physical Constants
cdef double pi = np.pi
cdef double c = 299792.458
cdef double G = 6.674e-20 #km^3 / (kg * s)^2
cdef double Msun = 1.989e30

## Solar System Parameters
cdef double[:] e1 = np.array([.994, .1095, .003116])
cdef double[:] e2 = np.array([-.05174, .4945, -.8677]) 
cdef double vSun_x = 11.0
cdef double vSun_y = 232.0
cdef double vSun_z = 7.0  
cdef double vEarthMag = 29.79 # Earth's speed km/s
cdef double omega = 2. * np.pi / 365. # angular velocity rad/s
cdef double tp = -74.88 # days
cdef double t1 = 0.0 # days
cdef double lambda_p = np.deg2rad(102.0)
cdef double a = 1.496e8
cdef double e = .016722

##Evaluates the velocity distribution for a given velocity, v0, vObs
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double fHalo(double v, double v0, double vObs) nogil:
    cdef double norm = 1.0/sqrt(pi)/v0/vObs
    cdef double f = norm*v*exp(-(v-vObs)**2 / v0**2) -norm*v* exp(- (v + vObs)**2 / v0**2)

    return f


##Evaluates the perturbation to the velocity distribution
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double f1(double vMag, double v0, double t) nogil:

    cdef double g_t = omega * (t-tp)
    cdef double nu = g_t + 2 * e * sin(g_t)*5.0/4.0 * e**2 * sin(2*g_t)
    cdef double r_t = a * (1-e**2) / (1 + e * cos(nu))
    cdef double lambda_t = lambda_p + nu


    cdef double earth_pos_x = r_t * (-sin(lambda_t) * e1[0] + cos(lambda_t) * e2[0])
    cdef double earth_pos_y = r_t * (-sin(lambda_t) * e1[1] + cos(lambda_t) * e2[1])
    cdef double earth_pos_z = r_t * (-sin(lambda_t) * e1[2] + cos(lambda_t) * e2[2])

    cdef double earth_pos_norm = sqrt(earth_pos_x**2 + earth_pos_y**2 + earth_pos_z**2)

    cdef double earth_pos_unit_x = earth_pos_x / earth_pos_norm
    cdef double earth_pos_unit_y = earth_pos_y / earth_pos_norm
    cdef double earth_pos_unit_z = earth_pos_z / earth_pos_norm

    cdef double vEarth_x = vEarthMag * (cos(omega*(t-t1))*e1[0] + sin(omega*(t-t1))*e2[0])
    cdef double vEarth_y = vEarthMag * (cos(omega*(t-t1))*e1[1] + sin(omega*(t-t1))*e2[1])
    cdef double vEarth_z = vEarthMag * (cos(omega*(t-t1))*e1[2] + sin(omega*(t-t1))*e2[2])

    cdef double prefactor = 2 * G * Msun / earth_pos_norm / v0**5 / pi**1.5

    cdef int num_theta_points = 51
    cdef double dTheta = 1.0/(num_theta_points - 1)*pi

    cdef int num_phi_points = 51
    cdef double dPhi = 1.0 / (num_phi_points - 1) * 2 * pi

    cdef Py_ssize_t i, j
    cdef double theta, phi, vx, vy, vz, temp_norm, temp_unit_x, temp_unit_y, temp_unit_z, temp
    cdef double integral = 0.0

    cdef double cos_theta = cos(0.0)
    cdef double sin_theta = sin(0.0)
    cdef double cos_phi = cos(0.0)
    cdef double sin_phi = sin(0.0)

    cdef double old_cos_theta
    cdef double old_cos_phi

    cdef double ci_theta = cos(dTheta)
    cdef double si_theta = sin(dTheta)

    cdef double ci_phi = cos(dPhi)
    cdef double si_phi = sin(dPhi)

    for i in range(num_theta_points):
        for j in range(num_phi_points):
            temp = 0.0

            vx = vMag * sin_theta * cos_phi
            vy = vMag * sin_theta * sin_phi
            vz = vMag * cos_theta

            temp_norm = sqrt((vx + vEarth_x)**2 + (vy + vEarth_y)**2 + (vz + vEarth_z)**2)
            temp_unit_x = (vx + vEarth_x) / temp_norm
            temp_unit_y = (vy + vEarth_y) / temp_norm
            temp_unit_z = (vz + vEarth_z) / temp_norm

            temp = - prefactor * sin_theta * vMag**2 * dTheta * dPhi			
            temp *= exp(-((vx + vEarth_x + vSun_x)**2 + (vy + vEarth_y + vSun_y)**2 + (vz + vEarth_z + vSun_z)**2) / v0**2)
            temp /= temp_norm

            temp *= ((vx + vEarth_x + vSun_x)*(earth_pos_unit_x - temp_unit_x) \
                    + (vy + vEarth_y + vSun_y)*(earth_pos_unit_y - temp_unit_y) \
                    + (vz + vEarth_z + vSun_z)*(earth_pos_unit_z - temp_unit_z))
            
            temp /= (1 - (earth_pos_unit_x * temp_unit_x + earth_pos_unit_y * temp_unit_y + earth_pos_unit_z * temp_unit_z))

            integral += temp

        old_cos_phi = cos_phi
        cos_phi = cos_phi * ci_phi - sin_phi * si_phi
        sin_phi = sin_phi * ci_phi + old_cos_phi * si_phi

        old_cos_theta = cos_theta
        cos_theta = cos_theta * ci_theta - sin_theta * si_theta
        sin_theta = sin_theta * ci_theta + old_cos_theta * si_theta

    return integral
