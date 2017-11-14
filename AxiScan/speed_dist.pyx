###############################################################################
# phase_space_distribution.pyx
###############################################################################
#
# Functions associated with the axion speed distribution
#
###############################################################################


# Import basic functions
import numpy as np
cimport numpy as np
cimport cython

# C functions
cdef extern from "math.h":
    double exp(double x) nogil
    double cos(double x) nogil
    double sin(double x) nogil
    double sqrt(double x) nogil

# Physical Constants
cdef double pi = np.pi
cdef double c = 299792.458 # speed of light [km/s]
cdef double G = 6.674e-20 # gravitation Constant [km^3/kg/s^2]

# Solar System Parameters
cdef double Msun = 1.989e30 # [kg]
# two vectors below define the plane of the Earth's orbit
# see e.g. fig. 4 of 1307.5323 
cdef double[:] e1 = np.array([.994, .1095, .003116])
cdef double[:] e2 = np.array([-.05174, .4945, -.8677]) 
cdef double vEarthMag = 29.79 # Earth's speed [km/s]
cdef double omega = 2.*pi/365. # period of Earth's revolution [rad/day]
cdef double tp = -74.88 # time of the perihelion [day] 
cdef double t1 = 0.0 # vernal equinox, taking to be our origin [day]
cdef double lambda_p = 102.*pi/180. # ecliptic longitude of the perihelion
cdef double a = 1.496e8 # semimajor axis of Earth's orbit [km]
cdef double e = .016722 # eccentricity of Earth's orbit


#########################################################
# External functions									#
#########################################################

## Compute a given day's boost velocity
cdef double get_vObs(double vDotMag, double alpha, double tbar, double day) nogil:
    return sqrt(vDotMag**2 + vEarthMag**2 + 2 * vDotMag * vEarthMag * alpha*cos(omega*(day-tbar)))

## Evaluates the SHM velocity distribution for a given velocity, v0, vObs
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double f_SHM(double v, double v0, double vObs) nogil:
    cdef double norm = 1.0/sqrt(pi)/v0/vObs
    cdef double f = norm*v*exp(-(v-vObs)**2 / v0**2) -norm*v* exp(- (v + vObs)**2 / v0**2)
    return f

## Evaluates the perturbation to the velocity distribution due to gravitational focusing
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double f_GF(double vMag, double v0, double vSun_x, double vSun_y, double vSun_z, double t) nogil:

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
