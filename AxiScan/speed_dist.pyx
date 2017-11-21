###############################################################################
# speed_dist.pyx
###############################################################################
#
# Functions associated with the axion speed distribution
#
###############################################################################


# Import basic functions
import numpy as np
cimport numpy as np
cimport cython

# C math functions
cdef extern from "math.h":
    double pow(double x, double y) nogil
    double exp(double x) nogil
    double cos(double x) nogil
    double sin(double x) nogil
    double sqrt(double x) nogil

# Physical Constants
cdef double pi = np.pi
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


######################
# External functions #
######################

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double get_vObs(double vDotMag, double alpha, double tbar, 
                     double day) nogil:
    """ Earth speed in galactic frame on a given day

    :param vDotMag: velocity of the sun w.r.t. the galactic frame [km/s]
    :param alpha/tbar: scalar quantities defining direction of vDot
    :param day: day with respect to t1, vernal equinox

    :returns: vObs(day) [km/s]
    """

    return sqrt(pow(vDotMag,2.) + pow(vEarthMag,2.) + 2.*vDotMag*vEarthMag 
                *alpha*cos(omega*(day-tbar)))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double f_SHM(double v, double v0, double vObs) nogil:
    """ Standard Halo Model (SHM) at a given speed
    
    :param v: speed to evaluate SHM at [km/s]
    :param v0: velocity dispersion of SHM [km/s]
    :param vObs: lab/observer/Earth speed w.r.t. the galactic frame [km/s]

    :returns: f_SHM(v|v0,vObs) [s/km]
    """
    
    cdef double norm = 1./sqrt(pi)/v0/vObs
    return norm*v*exp(-pow(v-vObs,2.) / pow(v0,2.)) \
           -norm*v*exp(-pow(v + vObs,2.) / pow(v0,2.))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double f_GF(double v, double v0, double vSun_x, double vSun_y, 
                 double vSun_z, double day) nogil:
    """ Gravitational Focusing (GF) addition to the phase space velocity, as
        defined in (91-92) of 1711.0xxxx. This is the perturbation to the dark 
        matter phase-space distribution by the gravitational field of the Sun.
        Code calculates the integral in (92), which converts f_GF from a vel
        to a speed distribution. For more details see 1308.1953 and 1711.03554.

        :param v: speed to evaluate f_GF at [km/s]
        :param v0: velocity dispersion of bulk halo, which GF perturbs [km/s]
        :param vSun_(x,y,z): Sun velocity w.r.t. the bulk halo [kms/s]
        :param day: day with respect to t1, vernal equinox

        :returns: f_GF(v,t) [s/km]
    """

    cdef double g_t = omega * (day-tp)
    cdef double nu = g_t + 2.*e*sin(g_t)*5./4.*pow(e,2.)*sin(2.*g_t)
    cdef double r_t = a*(1.-pow(e,2.)) / (1. + e*cos(nu))
    cdef double lambda_t = lambda_p + nu

    cdef double Earth_pos_x = r_t * (-sin(lambda_t)*e1[0] + cos(lambda_t)*e2[0])
    cdef double Earth_pos_y = r_t * (-sin(lambda_t)*e1[1] + cos(lambda_t)*e2[1])
    cdef double Earth_pos_z = r_t * (-sin(lambda_t)*e1[2] + cos(lambda_t)*e2[2])

    cdef double Earth_pos_norm = sqrt(pow(Earth_pos_x,2.) + pow(Earth_pos_y,2.) 
                                      + pow(Earth_pos_z,2.))

    cdef double Earth_pos_unit_x = Earth_pos_x / Earth_pos_norm
    cdef double Earth_pos_unit_y = Earth_pos_y / Earth_pos_norm
    cdef double Earth_pos_unit_z = Earth_pos_z / Earth_pos_norm

    cdef double vEarth_x = vEarthMag * (cos(omega*(day-t1))*e1[0] 
                                      + sin(omega*(day-t1))*e2[0])
    cdef double vEarth_y = vEarthMag * (cos(omega*(day-t1))*e1[1] 
                                      + sin(omega*(day-t1))*e2[1])
    cdef double vEarth_z = vEarthMag * (cos(omega*(day-t1))*e1[2] 
                                      + sin(omega*(day-t1))*e2[2])

    cdef double prefactor = 2.*G*Msun/Earth_pos_norm/pow(v0,5.)/pow(pi,1.5)

    cdef int num_theta_points = 51
    cdef int num_phi_points = 51
    cdef double dTheta = pi/(num_theta_points-1.)
    cdef double dPhi = 2.*pi/(num_phi_points-1.)

    cdef Py_ssize_t i, j
    cdef double theta, phi, vx, vy, vz, temp_norm, temp_unit_x, temp_unit_y, 
    cdef double temp_unit_z, temp, old_cos_theta, old_cos_phi
    cdef double integral = 0.0

    cdef double cos_theta = cos(0.0)
    cdef double sin_theta = sin(0.0)
    cdef double cos_phi = cos(0.0)
    cdef double sin_phi = sin(0.0)

    cdef double ci_theta = cos(dTheta)
    cdef double si_theta = sin(dTheta)

    cdef double ci_phi = cos(dPhi)
    cdef double si_phi = sin(dPhi)

    for i in range(num_theta_points):
        for j in range(num_phi_points):
            temp = 0.0

            vx = v * sin_theta * cos_phi
            vy = v * sin_theta * sin_phi
            vz = v * cos_theta

            temp_norm = sqrt(pow(vx + vEarth_x,2.) + pow(vy + vEarth_y,2.) 
                           + pow(vz + vEarth_z,2.))
            temp_unit_x = (vx + vEarth_x) / temp_norm
            temp_unit_y = (vy + vEarth_y) / temp_norm
            temp_unit_z = (vz + vEarth_z) / temp_norm

            temp = - prefactor * sin_theta * pow(v,2.) * dTheta * dPhi
            temp *= exp(-(pow(vx + vEarth_x + vSun_x,2.) 
                         +pow(vy + vEarth_y + vSun_y,2.) 
                         +pow(vz + vEarth_z + vSun_z,2.)) / pow(v0,2.))
            temp /= temp_norm

            temp *= ((vx + vEarth_x + vSun_x)*(Earth_pos_unit_x - temp_unit_x) \
                    + (vy + vEarth_y + vSun_y)*(Earth_pos_unit_y - temp_unit_y) \
                    + (vz + vEarth_z + vSun_z)*(Earth_pos_unit_z - temp_unit_z))
            
            temp /= (1. - (Earth_pos_unit_x * temp_unit_x 
                          +Earth_pos_unit_y * temp_unit_y 
                          +Earth_pos_unit_z * temp_unit_z))

            integral += temp

        old_cos_phi = cos_phi
        cos_phi = cos_phi * ci_phi - sin_phi * si_phi
        sin_phi = sin_phi * ci_phi + old_cos_phi * si_phi

        old_cos_theta = cos_theta
        cos_theta = cos_theta * ci_theta - sin_theta * si_theta
        sin_theta = sin_theta * ci_theta + old_cos_theta * si_theta

    return integral
