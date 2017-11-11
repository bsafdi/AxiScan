###############################################################################
# LL_Scan.pyx
###############################################################################
#
# Evaluate the Test Statistic for ABRACADABRA at a series of input values
#
###############################################################################


# Import basic functions
import numpy as np
cimport numpy as np
cimport cython
cimport likelihoods as LL

# Useful constants
cdef double pi = np.pi
cdef double c = 299792.458 # speed of light [km/s]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef TS_Scan(double[::1] PSD, double[::1] freqs, double[::1] mass_TestSet,
              double[::1] A_TestSet, double PSDback, double v0, double vObs,
              double num_stacked):

    cdef int N_masses = len(mass_TestSet)
    cdef int N_AVals = len(A_TestSet)


    cdef double[:, ::1] TS_Array = np.zeros((N_masses, N_AVals))

    cdef Py_ssize_t iM, iA

    for iM in range(N_masses):
        for iA in range(N_AVals):
   
            TS_Array[iM, iA] += LL.stacked_likelihood(freqs, PSD, mass_TestSet[iM],  A_TestSet[iA],
                                                     v0, vObs, PSDback, num_stacked)

            TS_Array[iM, iA] -= LL.stacked_likelihood(freqs, PSD, mass_TestSet[iM],  0,
                                                     v0, vObs, PSDback, num_stacked)

            TS_Array[iM, iA] *= 2


    return TS_Array


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef double SHM_AnnualMod_Likelihood(double[::1] freqs, double[:, ::1] PSD, double mass, double A,
                                     double v0, double vDotMag, double alpha, double tbar,
                                     double PSDback, double num_stacked) nogil:

    return LL.SHM_AnnualMod_Likelihood(freqs, PSD, mass, A, v0, vDotMag, alpha, tbar, PSDback, num_stacked)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef double Substructure_AnnualMod_Likelihood(double[::1] freqs, double[:, ::1] PSD, double mass, double A,
                                              double v0_Halo, double vDotMag_Halo, double alpha_Halo, double tbar_Halo,
                                              double v0_Sub, double vDotMag_Sub, double alpha_Sub, double tbar_Sub,
                                              double frac_Sub, double PSDback, double num_stacked) nogil:


    return LL.Substructure_AnnualMod_Likelihood(freqs, PSD, mass, A, v0_Halo, vDotMag_Halo, alpha_Halo, tbar_Halo,
                                                v0_Sub, vDotMag_Sub, alpha_Sub, tbar_Sub, frac_Sub, PSDback, num_stacked) 

