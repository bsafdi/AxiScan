###############################################################################
# ll_SHM.pyx
###############################################################################
#
# Code to evaluate the ABRACADABRA likelihood in the SHM case
#
###############################################################################

# Import basic functions
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef ABRA_LL(double[::1] freq, double[:, ::1] PSD, int N_Days, double A, 
              double ma, double v0, double vDotMag, double alpha, double tBar,
              double PSDback, int includeGF):
    """ Calculate the ABRACADABRA Log Likelihood as a function of the data and
        for given model parameters.
          - freq: frequencies data is measured at
          - PSD: measurements in the form of PSD
            NB: In the stacked case the data should be the average PSD
          - num_stacked: the number of subintervals over which the data is
            stacked
          - min_Resolve: the minimum relative frequency to resolve
          - v0: velocity dispersion (=sqrt(2) sigma_v), nominally 220 km/s
          - vObs: velocity of the Sun in the Milky Way frame, nominally 232 km/s
          - PSDback: PSDbackground value
    """

    # Setup length of input data
    cdef int N_frq = len(freq)

    # Setup loop variables
    cdef double LambdaK_val
    cdef Py_ssize_t ifrq, iday
    cdef double omega = 2 * pi / 365
    cdef double vCrossMag = 29.79
    cdef double vObs

    # Construct full LL
    cdef double ll = 0.
    with nogil:
        for iday in range(N_Days):
            vObs = sqrt(vDotMag**2 + vCrossMag**2 \
                   + 2*vDotMag * vCrossMag *alpha*cos(omega*(iday - tBar)))

            for ifrq in range(N_frq):
                LambdaK_val = LambdaK(freq[ifrq], ma, A, PSDback, v0, vObs, iday, includeGF)
                ll += -PSD[iday, ifrq] / LambdaK_val - log(LambdaK_val)

    return ll


########################
# Additional Functions #
########################

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

# C math functions
cdef extern from "math.h":
    double log(double x) nogil
    double exp(double x) nogil
    double pow(double x, double y) nogil
    double sqrt(double x) nogil
    double cos(double x) nogil
    double sin(double x) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double LambdaK(double freq, double ma, double A, double PSDback,
                    double v0, double vObs, double day, int includeGF) nogil:
    """ Calculate the mean of the exponential distribution followed by the PSD
          - freq: frequencies to calculate LambdaK [Hz]
          - ma: ma/2pi is the frequency associated with the axion mass [Hz]
          - A: shorthand for the strength of the axion PSD;
            A = (gagg * Bmax * VB * alpha)^2 * rhoDM * (L/Lp) / 4
          - PSDback: the mean expected background PSD
          - v0: velocity dispersion (=sqrt(2) sigma_v), nominally 220 km/s
          - vObs: velocity of the Sun in the Milky Way frame, nominally
            232 km/s
    """

    cdef double vSq = 2.*(2.*pi*freq-ma)/ ma

    if vSq > 0:
        return A * pi * (fSHM(sqrt(vSq), v0/c, vObs/c) + \
               c*f1(c*sqrt(vSq),v0, day, includeGF)) / ma / sqrt(vSq) + PSDback
 
    else:
        return PSDback

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



##Evaluates the perturbation to the velocity distribution
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double f1(double vMag, double v0, double t, int includeGF) nogil:

    if includeGF == 0:
        return 0.0

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
    cdef double theta, phi, vx, vy, vz, temp_norm, \
                       temp_unit_x, temp_unit_y, \
                       temp_unit_z, temp
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

            temp_norm = sqrt((vx + vEarth_x)**2 \
                        + (vy + vEarth_y)**2 \
                        + (vz + vEarth_z)**2)

            temp_unit_x = (vx + vEarth_x) / temp_norm
            temp_unit_y = (vy + vEarth_y) / temp_norm
            temp_unit_z = (vz + vEarth_z) / temp_norm

            temp = - prefactor * sin_theta * vMag**2 * dTheta * dPhi
            temp *= exp(-((vx + vEarth_x + vSun_x)**2 \
                          + (vy + vEarth_y + vSun_y)**2 \
                          + (vz + vEarth_z + vSun_z)**2) / v0**2)

            temp /= temp_norm
            temp *= ((vx + vEarth_x + vSun_x)*(earth_pos_unit_x - temp_unit_x) \
                    + (vy + vEarth_y + vSun_y)*(earth_pos_unit_y - temp_unit_y) \
                    + (vz + vEarth_z + vSun_z)*(earth_pos_unit_z - temp_unit_z))

            temp /= (1 - (earth_pos_unit_x * temp_unit_x \
                          + earth_pos_unit_y * temp_unit_y \
                          + earth_pos_unit_z * temp_unit_z))

            integral += temp

        old_cos_phi = cos_phi
        cos_phi = cos_phi * ci_phi - sin_phi * si_phi
        sin_phi = sin_phi * ci_phi + old_cos_phi * si_phi

        old_cos_theta = cos_theta
        cos_theta = cos_theta * ci_theta - sin_theta * si_theta
        sin_theta = sin_theta * ci_theta + old_cos_theta * si_theta

    return integral

