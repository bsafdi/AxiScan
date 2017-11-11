###############################################################################
# phase_space_distribution.pyx
###############################################################################
#
# Class to handle details of the axion velocity distribution
#
###############################################################################

# Import basic functions
import numpy as np
import numpy.linalg as LA
cimport numpy as np
cimport cython
from cython.parallel import parallel, prange
import scipy.special as sc

## Compute a given day's boost velocity
cdef double get_vObs(double vDotMag, double alpha, double tbar, double day) nogil

## Evaluates the SHM velocity distribution for a given velocity, v0, vObs
cdef double f_SHM(double v, double v0, double vObs) nogil

## Evaluates the perturbation to the velocity distribution due to gravitational focusing
cdef double f_GF(double vMag, double v0, double vSun_x, double vSun_y, double vSun_z, double t) nogil

