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
cpdef PSD_Scan(double[::1] PSD, double[::1] freqs, 
              double[::1] mass_TestSet, double[::1] PSDback_TestSet,
              double v0_Halo, double vObs_Halo,
              double num_stacked):

    # Setup the length of input and output arrays
    cdef int N_freqs = len(freqs)
    cdef int N_masses = len(mass_TestSet)
    cdef int N_PSDb = len(PSDback_TestSet)
    cdef double[:, ::1] LL_Array = np.zeros((N_masses, N_PSDb))
    cdef double[::1] PSD_ScanResults = np.zeros((N_masses))
    
    # Setup loop variables
    cdef double LambdaK, fmin, fmax
    cdef double df = freqs[1]-freqs[0]
    cdef int fminIndex, fmaxIndex, maxLoc
    cdef Py_ssize_t im, iPSDb, ifrq

    # Loop through masses and A values and calculate the TS for each
    for im in range(N_masses):
        # Only look at a range of frequencies around the mass
        fmin = mass_TestSet[im] / 2.0 / pi
        fmax = fmin * (1+3*(v0_Halo+vObs_Halo)**2 / c**2)
        fminIndex = np.searchsorted(freqs, fmin)+1
        fmaxIndex = int_min(np.searchsorted(freqs, fmax), N_freqs - 1)

        for iPSDb in range(N_PSDb):
            for ifrq in range(fminIndex, fmaxIndex):
                # Lambda_K associated with Background only
                LambdaK = Lambdak(freqs[ifrq], mass_TestSet[im], 
                                   0.0, PSDback_TestSet[0], 
                                   v0_Halo, vObs_Halo)

                LL_Array[im, iPSDb] += log(gamma_PDF(PSD[ifrq], num_stacked, LambdaK / num_stacked))

    for im in range(N_masses):
        maxLoc = 0
        for iPSDb in range(N_PSDb):
            if LL_Array[im, iPSDb] > LL_Array[im, maxLoc]:
                maxLoc = iPSDb
        PSD_ScanResults[im] = PSDback_TestSet[maxLoc]

    return np.array(PSD_ScanResults)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef TS_Scan(double[::1] PSD, double[::1] freqs, double[::1] mass_TestSet,
              double[::1] A_TestSet, double[::1] PSDback,
              double v0, double vObs, double num_stacked, int min_Resolve):

    # Setup the length of input and output arrays
    cdef int N_freqs = len(freqs)
    cdef int N_masses = len(mass_TestSet)
    cdef int N_A = len(A_TestSet)
    cdef double[:, ::1] TS_Array = np.zeros((N_masses, N_A))

    # Setup loop variables
    cdef double LambdakA, Lambdak0, fmin, fmax
    cdef double df = freqs[1] - freqs[0]
    cdef int fminIndex, fmaxIndex
    cdef Py_ssize_t im, iA, ifrq

    # Loop through masses and A values and calculate the TS for each
    for im in range(N_masses):
        # Only look at a range of frequencies around the mass
            fmin = mass_TestSet[im] / 2.0 / pi
            fmax = fmin*(1+3*(v0 + vObs)**2 / c**2)
            fminIndex = np.searchsorted(freqs, fmin)+1
            fmaxIndex = int_min(np.searchsorted(freqs, fmax), N_freqs - 1)


            # Skip if below the minimum resolved relative frequency size
            for iA in range(N_A):
                for ifrq in range(fminIndex, fmaxIndex):
                    # Lambda_k associated with Signal + Background
                    LambdakA = Lambdak(freqs[ifrq], mass_TestSet[im],
                                       A_TestSet[iA], PSDback[im], v0, vObs)
                    # Lambda_k associated with Background only
                    Lambdak0 = Lambdak(freqs[ifrq], mass_TestSet[im],
                                       0.0, PSDback[im], v0, vObs)

                    TS_Array[im, iA] += 2*log( gamma_PDF(PSD[ifrq], num_stacked, LambdakA / num_stacked))
                    TS_Array[im, iA] -= 2*log( gamma_PDF(PSD[ifrq], num_stacked, Lambdak0 / num_stacked))


    return TS_Array

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
    double exp(double x) nogil

cdef extern from "complex.h":
    double cabs(complex z) nogil # complex absolute value

cdef extern from "gsl/gsl_randist.h":
    double gsl_ran_gamma_pdf(double x, double a, double b) nogil

cpdef inline double gamma_PDF(double x, double a, double b) nogil:
    return gsl_ran_gamma_pdf(x, a, b)

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
