###############################################################################
# axion_ll.pxd
###############################################################################
#
# Here we predefine all functions in axion_ll.pyx so that they are compiled 
# simultaneously - this way the code optimizes all functions at once and allows 
# functions to be called as pure C
#
###############################################################################


cdef int getIndex(double[::1] freqs, double target) nogil

cdef double stacked_ll(double[::1] freqs, double[::1] PSD, double mass, 
                       double A, double v0, double vObs, double PSDback, 
                       double num_stacked) nogil

cdef double SHM_AnnualMod_ll(double[::1] freqs, double[:, ::1] PSD, double mass, 
                             double A, double v0, double vDotMag, double alpha, 
                             double tbar, double PSDback, 
                             double num_stacked) nogil

cdef double Sub_AnnualMod_ll(double[::1] freqs, double[:, ::1] PSD, double mass, 
                             double A, double v0_Halo, double vDotMag_Halo, 
                             double alpha_Halo, double tbar_Halo, double v0_Sub, 
                             double vDotMag_Sub, double alpha_Sub, 
                             double tbar_Sub, double frac_Sub, double PSDback, 
                             double num_stacked) nogil
