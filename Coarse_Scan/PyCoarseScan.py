import numpy as np
import scipy
import scipy.stats
import sys, os

sys.path.append('./cython/')
import coarseScanner as scanner

##################################
###   Load the Existing Data   ###
##################################

data_dir = '/nfs/turbo/bsafdi/fosterjw/github/ABRA-DATA/Data/'

freqs = np.load(data_dir + 'freqs.npy')
PSDFile = ''

fileList = os.listdir(data_dir)
for fileName in fileList:
    if fileName.startswith('Stacked'):
        PSDFile = fileName

PSD = np.load(data_dir + PSDFile)
N_stacked = float(PSDFile[11:PSDFile.index('.')])

###################################
###   Set the Scan Parameters   ###
###################################

c = 299792.458 # km/s
PSDback = 163539.36 #eV^{-1}
v0 = 220 # km/s
vObs = 232 # km/s

###########################
###   Methods We Need   ###
###########################

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

def find_nearest_index(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx




########################
###   Run the Scan   ###
########################

# Infer the collection time from the frequency resolution
collectionTime = 1.0/(freqs[1] - freqs[0])

# Construct the test set of masses
N_testMass = int(np.log(freqs[-1] / freqs[0])  / np.log(1. + v0**2. / 2. / c**2.))
mass_TestSet = freqs[0]*(1. + v0**2. / 2. / c**2.)**np.arange(N_testMass) * 2*np.pi
num_Masses = c**2 / v0**2 * np.log(freqs[-1]/freqs[0])

# Determine our 1 sigma lower bound on the 95% exclusion line
sigmaA = getSigma_A(mass_TestSet, N_stacked, collectionTime, v0, vObs, PSDback)
exclusionA = zScore(-1)*sigmaA

# Determine the TS threshold for detection and the A associated
detectionP = 1. - 2.*(1.-scipy.stats.norm.cdf(5.)) # 5 sigma
TS_Thresh = 2.*scipy.special.erfinv(1.-2.*(1.-detectionP)/num_Masses)**2.
detectionA = solveForA(mass_TestSet, TS_Thresh, N_stacked, collectionTime,
                       v0, vObs, PSDback)


# We are using the CLs method, so we do not go below 1 sigma lower
A_Min = np.amin(exclusionA)*0.01
# Also no need to go too far above detection
A_Max = np.amax(detectionA)*10.

# This establishes the range of As to scan over
A_TestSet = np.sort(np.append(10**np.linspace(np.log10(A_Min),
                    np.log10(A_Max), 100), 0.0))


# Now calculate the TS array
TS_Array_raw = scanner.TS_Scan(PSD, freqs, mass_TestSet, A_TestSet,
                                   PSDback, v0, vObs, N_stacked)

TS_Array = np.zeros(TS_Array_raw.shape)

for i in range(len(mass_TestSet)):
    TS_Array[i] = TS_Array_raw[i] - np.amax(TS_Array_raw[i])

# Extract limits on A at each mass from the TS results
A_Limits = []
for i in range(len(mass_TestSet)):
    A_Limit = A_TestSet[find_nearest_index(TS_Array[i], -1.645)]
    A_Limits.append(A_Limit)

A_Limits = np.array(A_Limits)


np.save(data_dir + 'A_Limits.npy', A_Limits)
np.save(data_dir + 'Sigma_A.npy', sigmaA)
np.save(data_dir + 'Mass_TestSet.npy', mass_TestSet)
np.save(data_dir + 'Detection_Threshold.npy', detectionA)
