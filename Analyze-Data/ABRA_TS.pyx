###############################################################################
# ABRA_TS.pyx
###############################################################################
#
# Evaluate the Test Statistic for ABRACADABRA at a series of input values
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
cpdef TS_Scan(double[::1] PSD, double[::1] freqs, 
			  double[::1] mass_TestSet, double[::1] A_TestSet, 
			  double[::1] PSDback_TestSet, double v0_Halo, double vObs_Halo,
			  double num_stacked, int min_Resolve):
    """ Calculate the Test Statistic for a given input dataset (PSDs and freqs)
        at a number of different mass and A value points
          - PSD: measured power spectral densities
          - freqs: frequencies associated with those PSDs [Hz]
          - mass_TestSet: array of axion mass values; ma/2pi is the frequency 
            associated with the axion mass [Hz]
          - A_TestSet: array of A values; strength of axion PSD
            A = (gagg * Bmax * VB * alpha)^2 * rhoDM * (L/Lp) / 4
          - PSDback_TestSet: array of mean expected background PSDs to scan over
          - v0_Halo: velocity dispersion (=sqrt(2) sigma_v), nominally 220 km/s
          - vObs_Halo: velocity of the Sun in the Milky Way frame, nominally
            232 km/s
          - num_stacked: the number of subintervals over which the data is 
            stacked each of length Delta T = total T / num_stacked
          - min_Resolve: minimum relative frequency size to resolve 
    """

    # Setup the length of input and output arrays
    cdef int N_freqs = len(freqs)
    cdef int N_masses = len(mass_TestSet)
    cdef int N_A = len(A_TestSet)
    cdef int N_PSDb = len(PSDback_TestSet)
    cdef double[:, :, ::1] TS_Array = np.zeros((N_masses, N_A, N_PSDb))
    
    # Setup loop variables
    cdef double LambdakA, Lambdak0, fmin, fmax
    cdef double df = freqs[1]-freqs[0]
    cdef int fminIndex, fmaxIndex
    cdef Py_ssize_t im, iA, iPSDb, ifrq

    # Loop through masses and A values and calculate the TS for each
    for im in range(N_masses):
        # Only look at a range of frequencies around the mass
        fmin = mass_TestSet[im] / 2.0 / pi
        fmax = fmin * (1+3*(v0_Halo+vObs_Halo)**2 / c**2)
        fminIndex = np.searchsorted(freqs, fmin)+1
        fmaxIndex = int_min(np.searchsorted(freqs, fmax), N_freqs - 1)

        # Skip if below the minimum resolved relative frequency size
        if (fmaxIndex - fminIndex) < min_Resolve: continue

        for iA in range(N_A):
            for iPSDb in range(N_PSDb):
                for ifrq in range(fminIndex, fmaxIndex):
                    # Lambda_k associated with Signal + Background
                    LambdakA = Lambdak(freqs[ifrq], mass_TestSet[im], 
                                       A_TestSet[iA], PSDback_TestSet[iPSDb],
                                       v0_Halo, vObs_Halo)
                    # Lambda_k associated with Background only
                    Lambdak0 = Lambdak(freqs[ifrq], mass_TestSet[im], 
                                       0.0, PSDback_TestSet[iPSDb], 
                                       v0_Halo, vObs_Halo)

                    # Calculate the TS = 2*[LL(S+B) - LL(B)]
                    TS_Array[im, iA, iPSDb] += 2*(-PSD[ifrq]
                                            *(1/LambdakA-1/Lambdak0) 
				                            - log(LambdakA/Lambdak0)) \
                                            * num_stacked
    
    return np.array(TS_Array)


########################
# Additional Functions #
########################

# C math functions
cdef extern from "math.h":
    double log(double x) nogil
    double exp(double x) nogil
    double pow(double x, double y) nogil
    double sqrt(double x) nogil
    double erf(double x) nogil

cdef extern from "complex.h":
    double cabs(complex z) nogil # complex absolute value

cdef inline int int_min(int a, int b) nogil: return a if a<= b else b


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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double Lambdak(double freq, double ma, double A, double PSDback,
            double v0_Halo, double vObs_Halo) nogil:
    """ Calculate the mean of the exponential distribution followed by the PSD
          - freq: frequencies to calculate Lambdak [Hz]
          - ma: ma/2pi is the frequency associated with the axion mass [Hz]
          - A: shorthand for the strength of the axion PSD;
            A = (gagg * Bmax * VB * alpha)^2 * rhoDM * (L/Lp) / 4
          - PSDback: the mean expected background PSD
          - v0_Halo: velocity dispersion (=sqrt(2) sigma_v), nominally 220 km/s
          - vObs_Halo: velocity of the Sun in the Milky Way frame, nominally
            232 km/s
    """

    cdef double v = sqrt(2.*(2.*pi*freq-ma)/ ma)
    return A * pi * fSHM(v, v0_Halo/c, vObs_Halo/c) / ma / v + PSDback


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef FFTtoPSD(complex[::1] FFT, double collectionTime):
    """ Convert an array of FFTs into PSDs
    """

    cdef int N = len(FFT)
    cdef float Nf = float(len(FFT))
    cdef double A = collectionTime / pow(Nf, 2.) # = dt^2/T
    cdef Py_ssize_t iN

    cdef double[::1] PSD = np.zeros(N)

    with nogil:
        for iN in range(N):
            PSD[iN] = pow(cabs(FFT[iN]), 2.0) * A 
    
    return np.array(PSD)
