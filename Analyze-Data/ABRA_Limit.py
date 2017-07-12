###############################################################################
# ABRA_Limit.py
###############################################################################
#
# Convert input ABRACADABRA data into an axion parameter limit plot
#
###############################################################################


# Import basic functions
import numpy as np
import scipy
import scipy.stats
import ABRA_TS # Module to compute the TS

# Useful constants
c = 299792.458 # speed of light [km/s]


def axion_limit_params(data, freqs, PSDback_min, PSDback_max, PSDback_bins, 
                       num_stacked=1, min_Resolve=150., v0=220., vObs=232., 
                       dataisFFT=False):
    """ Calculate what is required to make an axion limit plot
          - data: measurements in the form of PSD or FFT
          - freqs: frequencies data is measured at
          - PSDback_min: minimum PSD background to sample over
          - PSDback_max: maximum PSD background to sample over
          - PSDback_bins: number of bins to scan PSD background over
          - num_stacked: the number of subintervals over which the data is
            stacked
          - min_Resolve: minimum number of frequency bins needed to resolve
          - v0: velocity dispersion (=sqrt(2) sigma_v), nominally 220 km/s
          - vObs: velocity of the Sun in the Milky Way frame, nominally 232 km/s
          - dataisFFT: by default data is PSD, if FFT set to true
    """
    
    #############################
    # Establish Scan Parameters #
    #############################
    
    # Infer the collection time from the frequencies
    collectionTime = 1.0/(freqs[1]-freqs[0])

    # If data is FFT, convert to PSD
    if dataisFFT:
        PSD = ABRA_TS.FFTtoPSD(data, collectionTime)
    else:
        PSD = data

    # Setup array of masses to scan over given the frequencies
    N_testMass = int(np.log(freqs[-1] / freqs[0])
                 / np.log(1. + v0**2. / 2. / c**2.) + 1.)
    mass_TestSet = freqs[0]*(1. + v0**2. / 2. / c**2.)**np.arange(N_testMass) \
                   * 2*np.pi
    num_Masses = c**2 / v0**2 * np.log(freqs[-1]/freqs[0])

    # Determine the exclusion and detection thersholds
    exclusionP = .9 # This is 95% because we have a one sided distribution
    detectionP = 1. - 2.*(1.-scipy.stats.norm.cdf(5.)) # 5 sigma
    exclusion_TS = scipy.stats.chi2.ppf(exclusionP, 1)

    # Find the associated A values, which is related to physical parameters by:
    # A = (gagg * Bmax * VB * alpha)^2 * rhoDM * (L/Lp) / 4
    # - gagg: axion photon coupling
    # - Bmax: magnetic field at inner radius of the toroid
    # - VB: geometric factor depending on the size of the toroid
    # - alpha: factor depending on the SQUID geometry
    # - rhoDM: local dark matter density, 0.3-0.4 GeV/cm^3
    # - L: SQUID inductance
    # - Lp: pickup loop inductance
    exclusionA = solveForA(mass_TestSet, exclusion_TS, num_stacked, 
                           collectionTime, v0, vObs, PSDback)

    sigmaA = getSigma_A(mass_TestSet, num_stacked, collectionTime, v0, vObs, 
                        PSDback)
    exclusionA_1SigUp = zScore(1) * sigmaA
    exclusionA_1SigLo = zScore(-1) * sigmaA
    exclusionA_2SigUp = zScore(2) * sigmaA
    
    # Determine the TS threshold for detection and the A associated 
    TS_Thresh = 2.*scipy.special.erfinv(1.-2.*(1.-detectionP)/num_Masses)**2.
    detectionA = solveForA(mass_TestSet, TS_Thresh, num_stacked, collectionTime, 
                           v0, vObs, PSDback)

    # We are using the CLs method, so we do not go below 1 sigma lower
    A_Min = np.amin(exclusionA_1SigLo)*0.01
    # Also no need to go too far above detection
    A_Max = np.amax(detectionA)*10.

    # This establishes the range of As to scan over
    A_TestSet = np.sort(np.append(10**np.linspace(np.log10(A_Min), 
                        np.log10(A_Max), 100), 0.0))

    # Setup PSD background scan range
    PSDback_TestSet = np.linspace(PSDback_min, PSDback_max, PSDback_bins)

    ##################################
    # Perform Scan, Calculate Limits #
    ##################################
    
    # Now calculate the TS array
    TS_Array_raw = ABRA_TS.TS_Scan(PSD, freqs, mass_TestSet, A_TestSet, 
                                   PSDback_TestSet, v0, vObs, num_stacked, 
                                   min_Resolve)
    
    # Profile over PSDback
    TS_Array_all = np.amax(TS_Array_raw, axis = 2)
    
    # At each mass value subtract off the maximum TS, so the maximum value is 0
    TS_Array = np.zeros(TS_Array_all.shape)
    for i in range(N_testMass):
        TS_Array[i] = TS_Array_all[i] - np.amax(TS_Array_all[i])

    # Now at each mass find where TS drops below 2.71 from the max, this is the
    # 95% limit
    A_limits = np.zeros(N_testMass)
    for i in range(N_testMass):
        TS_single_mass = TS_Array[i]
        maxIndex = np.where(TS_single_mass == 0)[0][0]
        TS_single_mass[:maxIndex] = 0.0 # flatten to the left of maximum
        A_limits[i] = A_TestSet[find_nearest(TS_single_mass, 
                                -scipy.stats.chi2.ppf(exclusionP, 1))]

    return mass_TestSet, np.maximum(A_limits, exclusionA_OneSigmaLower), \
           exclusionA_TwoSigmaUpper, detectionA


########################
# Additional Functions #
########################

def find_nearest(array, value):
    """ Find the closest location to a value in an array
    """

    return (np.abs(array-value)).argmin()


def solveForA(mass, TS, num_stacked, collectionTime, v0, vObs, PSDback):
    """ Calculate the A associated with a given set of parameters assuming the
        Asimov dataset and a standard halo model
          - mass: axion mass; ma/2pi is the frequency associated with the axion 
            mass [Hz]
          - TS: Test Statistic value to solve for
          - num_stacked: the number of subintervals over which the data is 
            stacked
          - collectionTime: length of time of each subinterval
          - v0: velocity dispersion (=sqrt(2) sigma_v), nominally 220 km/s
          - vObs: velocity of the Sun in the Milky Way frame, nominally 232 km/s
          - PSDback: the mean expected background PSD
    """
    
    # Convert velocities to natural units
    v0 /= c
    vObs /= c

    # Break the calculation into three factors
    factor1 = num_stacked * collectionTime * np.pi / (2. * mass * PSDback**2.)
    factor2 = scipy.special.erf(np.sqrt(2.) * vObs / v0)
    factor3 = 1. / (np.sqrt(2.*np.pi) * v0 * vObs)

    return np.sqrt(TS / (factor1 * factor2 * factor3))


def getSigma_A(mass, num_stacked, collectionTime, v0, vObs, PSDback):
    """ Calculate the uncertainty on A associated with a given set of parameters
        assuming the Asimov dataset and a standard halo model. Parameters are
        as for solveForA
    """
    
    # Convert velocities to natural units
    v0 /= c
    vObs /= c

    # Break the calculation into three factors
    factor1 = num_stacked * collectionTime * np.pi / (2. * mass * PSDback**2.)
    factor2 = scipy.special.erf(np.sqrt(2.) * vObs / v0)
    factor3 = 1. / (np.sqrt(2.*np.pi) * v0 * vObs)

    return 1. / np.sqrt(factor1 * factor2 * factor3)


def zScore(N):
    """ Appropriate factor for the N-sigma confidence limit derived using the
        Asimov dataset
    """

    alpha = .1
    return scipy.stats.norm.ppf(1-alpha*scipy.stats.norm.cdf(N))+N
