###############################################################################
# ABRA_Limit_BackTemplate.py
###############################################################################
#
# Convert input ABRACADABRA data into an axion parameter limit plot
#
###############################################################################


# Import basic functions
import numpy as np
import scipy
import scipy.stats
import ABRA_TS_BackTemplate # Module to compute the TS

# Useful constants
c = 299792.458 # speed of light [km/s]


def axion_limit_params(PSD, freqs, PSDback_min, PSDback_max, PSDback_bins, 
                       num_stacked=1, min_Resolve=150., v0=220., vObs=232.):
    """ Calculate what is required to make an axion limit plot
          - PSD: measurements in the form of PSD
            NB: In the stacked case the data should be the average PSD
          - freqs: frequencies data is measured at
          - PSDback_min: minimum PSD background to sample over
          - PSDback_max: maximum PSD background to sample over
          - PSDback_bins: number of bins to scan PSD background over
          - num_stacked: the number of subintervals over which the data is
            stacked
          - min_Resolve: the minimum relative frequency to resolve
          - v0: velocity dispersion (=sqrt(2) sigma_v), nominally 220 km/s
          - vObs: velocity of the Sun in the Milky Way frame, nominally 232 km/s
    """
    
    #############################
    # Establish Scan Parameters #
    #############################
    
    # Infer the collection time from the frequencies
    collectionTime = 1.0/(freqs[1]-freqs[0])

    # Setup array of masses to scan over given the frequencies
    N_testMass = int(np.log(freqs[-1] / freqs[0])
                 / np.log(1. + v0**2. / 2. / c**2.) + 1.)
    mass_TestSet = freqs[0]*(1. + v0**2. / 2. / c**2.)**np.arange(N_testMass) \
                   * 2*np.pi
    num_Masses = c**2 / v0**2 * np.log(freqs[-1]/freqs[0])

    # Determine the exclusion and detection thresholds
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

    # First we need to compute the PSDback at each test mass signal window
    PSDback_TestSet = np.linspace(PSDback_min, PSDback_max, PSDback_bins)
    scanned_PSDback = ABRA_TS.PSD_Scan(PSD, freqs, PSDback_TestSet, v0, vObs, 
                                       num_stacked)

    # Now that we have the PSDback at each test mass signal window, we can compute
    # the detection and exclusion lines. These are for A ~ gagg**2

    sigmaA = getSigma_A(mass_TestSet, num_stacked, collectionTime, v0, vObs, 
                        scanned_PSDback)
    exclusionA = zScore(0)*sigmaA
    exclusionA_1SigUp = zScore(1) * sigmaA
    exclusionA_1SigLo = zScore(-1) * sigmaA
    exclusionA_2SigUp = zScore(2) * sigmaA
    
    # Determine the TS threshold for detection and the A associated 
    TS_Thresh = 2.*scipy.special.erfinv(1.-2.*(1.-detectionP)/num_Masses)**2.
    detectionA = solveForA(mass_TestSet, TS_Thresh, num_stacked, collectionTime, 
                           v0, vObs, scanned_PSDback)


    # We are using the CLs method, so we do not go below 1 sigma lower
    A_Min = np.amin(exclusionA_1SigLo)*0.01
    # Also no need to go too far above detection
    A_Max = np.amax(detectionA)*10.

    # This establishes the range of As to scan over
    A_TestSet = np.sort(np.append(10**np.linspace(np.log10(A_Min), 
                        np.log10(A_Max), 100), 0.0))

    ##################################
    # Perform Scan, Calculate Limits #
    ##################################
    
    # Now calculate the TS array
    TS_Array_raw = ABRA_TS.TS_Scan(PSD, freqs, mass_TestSet, A_TestSet, 
                                   scanned_PSDback, v0, vObs, num_stacked, 
                                   min_Resolve)

    # At each mass value subtract off the maximum TS, so the maximum value is 0
    TS_Array = np.zeros(TS_Array_raw.shape)
    for i in range(N_testMass):
        TS_Array[i] = TS_Array_raw[i] - np.amax(TS_Array_raw[i])

    # Now at each mass find where TS drops below 2.71 from the max, this is the
    # 95% limit
    A_limits = np.zeros(N_testMass)
    for i in range(N_testMass):
        TS_single_mass = TS_Array[i]
        maxIndex = np.where(TS_single_mass == 0)[0][0]
        TS_single_mass[:maxIndex] = 0.0 # flatten to the left of maximum
        A_limits[i] = A_TestSet[find_nearest(TS_single_mass, 
                                -scipy.stats.chi2.ppf(exclusionP, 1))]

    G_limits = np.sqrt(A_limits)

    # Now we compute the real sensitivity curves
    exclusionG = 1.3645322997 * np.sqrt(sigmaA)
    exclusionG_1SigUp = 1.61359288843 * np.sqrt(sigmaA)
    exclusionG_1SigLo = 1.14644076042 * np.sqrt(sigmaA)
    exclusionG_2SigUp = 1.86642501392 * np.sqrt(sigmaA)
    exclusionG_2SigLo = .974408677484 * np.sqrt(sigmaA)
    detectionG = np.sqrt(detectionA)

    return mass_TestSet, np.maximum(G_limits, exclusionG_1SigLo), exclusionG, \
           exclusionG_1SigLo, exclusionG_1SigUp, exclusionG_2SigUp, \
           exclusionG_2SigLo, detectionG
      
def axion_limit_params_template(data, freqs, PSDback_Template, num_stacked=1,
                                min_Resolve=150., v0=220., vObs=232., 
                                dataisFFT=False):
    """ Calculate what is required to make an axion limit plot
          - data: measurements in the form of PSD or FFT
          - freqs: frequencies data is measured at
          - PSDback_Template: Expectation for exponential noise at each frequency
          - num_stacked: the number of subintervals over which the data is
            stacked
          - min_Resolve: the minimum relative frequency to resolve
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

    # Determine the exclusion and detection thresholds
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

    # These are all preliminary values
    sigmaA = getSigma_A(mass_TestSet, num_stacked, collectionTime, v0, vObs, 
                        np.amin(PSDBack_Template))
    exclusionA = zScore(0)*sigmaA
    exclusionA_1SigUp = zScore(1) * sigmaA
    exclusionA_1SigLo = zScore(-1) * sigmaA
    exclusionA_2SigUp = zScore(2) * sigmaA
    
    # Determine the TS threshold for detection and the A associated 
    TS_Thresh = 2.*scipy.special.erfinv(1.-2.*(1.-detectionP)/num_Masses)**2.
    detectionA = solveForA(mass_TestSet, TS_Thresh, num_stacked, collectionTime, 
                           v0, vObs, np.amax(PSDBack_Template))


    # We are using the CLs method, so we do not go below 1 sigma lower
    A_Min = np.amin(exclusionA_1SigLo)*0.01
    # Also no need to go too far above detection
    A_Max = np.amax(detectionA)*10.

    # This establishes the range of As to scan over
    A_TestSet = np.sort(np.append(10**np.linspace(np.log10(A_Min), 
                        np.log10(A_Max), 100), 0.0))

    ##################################
    # Perform Scan, Calculate Limits #
    ##################################
    
    # Now calculate the TS array
    TS_Array_raw, PSDback_Min, PSDback_Max  = ABRA_TS.TS_Scan(PSD, freqs,
                                              mass_TestSet, A_TestSet, 
                                              scanned_PSDback, v0, vObs,
                                              num_stacked, min_Resolve)

    # At each mass value subtract off the maximum TS, so the maximum value is 0
    TS_Array = np.zeros(TS_Array_raw.shape)
    for i in range(N_testMass):
        TS_Array[i] = TS_Array_raw[i] - np.amax(TS_Array_raw[i])

    # Now at each mass find where TS drops below 2.71 from the max, this is the
    # 95% limit
    A_limits = np.zeros(N_testMass)
    for i in range(N_testMass):
        TS_single_mass = TS_Array[i]
        maxIndex = np.where(TS_single_mass == 0)[0][0]
        TS_single_mass[:maxIndex] = 0.0 # flatten to the left of maximum
        A_limits[i] = A_TestSet[find_nearest(TS_single_mass, 
                                -scipy.stats.chi2.ppf(exclusionP, 1))]

    G_limits = np.sqrt(A_limits)

    # Now we compute the real sensitivity curves
    sigmaA = getSigma_A(mass_TestSet, num_stacked, collectionTime, v0, vObs, 
                        PSDBack_Max)
    exclusionA = zScore(0)*sigmaA
    exclusionA_1SigUp = zScore(1) * sigmaA
    exclusionA_1SigLo = zScore(-1) * sigmaA
    exclusionA_2SigUp = zScore(2) * sigmaA
    
    # Determine the TS threshold for detection and the A associated 
    TS_Thresh = 2.*scipy.special.erfinv(1.-2.*(1.-detectionP)/num_Masses)**2.
    detectionA = solveForA(mass_TestSet, TS_Thresh, num_stacked, collectionTime, 
                           v0, vObs, PSDBack_Min)

    exclusionG = 1.3645322997 * np.sqrt(sigmaA)
    exclusionG_1SigUp = 1.61359288843 * np.sqrt(sigmaA)
    exclusionG_1SigLo = 1.14644076042 * np.sqrt(sigmaA)
    exclusionG_2SigUp = 1.86642501392 * np.sqrt(sigmaA)
    exclusionG_2SigLo = .974408677484 * np.sqrt(sigmaA)
    detectionG = np.sqrt(detectionA)

    return mass_TestSet, np.maximum(G_limits, exclusionG_1SigLo), exclusionG, \
           exclusionG_1SigLo, exclusionG_1SigUp, exclusionG_2SigUp, \
           exclusionG_2SigLo, detectionG



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
