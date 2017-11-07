import sys, os
import matplotlib as mpl
mpl.use('Agg')
import pymultinest
import numpy as np
import ll_AnnualMod_StreamHalo

c = 299792.458

data_dir = '/nfs/turbo/bsafdi/fosterjw/github/ABRA-DATA/Data/'
PSD_dir = '/nfs/turbo/bsafdi/fosterjw/github/ABRA-DATA/Data/PSD_Data/'
chains_dir = '/nfs/turbo/bsafdi/fosterjw/github/ABRA-DATA/Data/Stream_Halo_Chains/'

v0 = 220.0
vDotMag = 232.36
PSDback = 163539.36

includeGF = 0

##############################
###   Multinest Settings   ###
##############################

# Default multinest options
nlive = 50

pymultinest_options = {'importance_nested_sampling': False,
                        'resume': False, 'verbose': True,
                        'sampling_efficiency': 'model',
                        'init_MPI': False, 'evidence_tolerance': 0.5,
                        'const_efficiency_mode': False}

################################
###   Load the Simple Data   ###
################################

freqs = np.load(data_dir + 'freqs.npy')

massRange = np.array([.99999*(5.5e5*2*np.pi), 1.00001*(5.5e5*2*np.pi)])
ARange = np.array([4000.0, 260000.0])

os.chdir(PSD_dir)
fileList = os.listdir(os.getcwd())
N_days = len(fileList)


##################################################
###   Set Up Priors, Data for Halo Multinest   ###
##################################################

# I'm scanning over everything but PSDback
massPrior = [np.amin(massRange), np.amax(massRange)]
APrior = [.9*np.amin(ARange), 10*np.amax(ARange)]

v0_Halo_Prior = [.7*v0, 1.3*v0]
vDotMag_Halo_Prior = [.7*vDotMag, 1.3*vDotMag]
alpha_Halo_Prior = [0.0, 1.0]
tbar_Halo_Prior = [0.0, 364.0]

v0_Stream_Prior = [1.0, 100.0]
vDotMag_Stream_Prior = [220.0, 600.0]
alpha_Stream_Prior = [0.0, 1.0]
tbar_Stream_Prior = [0, 364.0]
frac_Stream_Prior = [0.0, .15]

# Find the window of frequencies that we want
freqMin = massPrior[0] / 2 / np.pi
freqMax = massPrior[1] * (1 + 3*(v0_Halo_Prior[1] + vDotMag_Halo_Prior[1])**2 / c**2) / 2 / np.pi
freqMin_Index = np.searchsorted(freqs, freqMin)
freqMax_Index = min(np.searchsorted(freqs, freqMax), len(freqs)-1)
freqsRestricted = freqs[freqMin_Index:freqMax_Index]

# Load the PSD data
PSD_Data = np.zeros((len(fileList), len(freqsRestricted)))
Day_Array = np.zeros((len(fileList)), dtype = int)
for i in range(len(fileList)):
    print(i)
    Day_Array[i] = int(fileList[i][7:fileList[i].index('.')])
    PSD_Data[i] = np.load(fileList[i])[freqMin_Index:freqMax_Index]

PSD_Data_Sorted = np.zeros(PSD_Data.shape)
Day_Array_Sorted = np.zeros(Day_Array.shape)
for i in range(len(Day_Array)):
    PSD_Data_Sorted[Day_Array[i]] = PSD_Data[i]
    Day_Array_Sorted[Day_Array[i]] = Day_Array[i]

######################################
###   SHM Annual Modulation Scan   ###
######################################

theta_min = [APrior[0], massPrior[0], v0_Halo_Prior[0], vDotMag_Halo_Prior[0], alpha_Halo_Prior[0], tbar_Halo_Prior[0], \
             v0_Stream_Prior[0], vDotMag_Stream_Prior[0], alpha_Stream_Prior[0], tbar_Stream_Prior[0], frac_Stream_Prior[0]]
theta_max = [APrior[1], massPrior[1], v0_Halo_Prior[1], vDotMag_Halo_Prior[1], alpha_Halo_Prior[1], tbar_Halo_Prior[1], \
             v0_Stream_Prior[1], vDotMag_Stream_Prior[1], alpha_Stream_Prior[1], tbar_Stream_Prior[1], frac_Stream_Prior[1]]

theta_interval = list(np.array(theta_max) - np.array(theta_min))
n_params = len(theta_min) # number of parameters to fit for

def prior_cube(cube, ndim=1, nparams=1):
    """ Cube of priors - in the format required by MultiNest
    """

    for i in range(ndim):
        cube[i] = cube[i] * theta_interval[i] + theta_min[i]
    return cube


def LL_StreamHalo(theta, ndim = 1, nparams = 1):
    return ll_AnnualMod_StreamHalo.ABRA_LL(freqsRestricted, PSD_Data_Sorted, N_days, theta[0], theta[1], theta[2], theta[3],
                                           theta[4], theta[5], theta[6], theta[7], theta[8], theta[9], theta[10],
                                           PSDback, includeGF)

pymultinest.run(LL_StreamHalo, prior_cube, n_params,
                outputfiles_basename=chains_dir,
                n_live_points=nlive, **pymultinest_options)


