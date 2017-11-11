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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double[:, ::1] TS_Scan(double[::1] PSD, double[::1] freqs, double[::1] mass_TestSet,
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

