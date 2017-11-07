import sys, os
import numpy as np
import dataGen_GF as dataGen


#######################
###   Directories   ###
#######################
data_dir = '/nfs/turbo/bsafdi/fosterjw/github/ABRA-DATA/Data/'
PSD_dir = '/nfs/turbo/bsafdi/fosterjw/github/ABRA-DATA/Data/PSD_Data/'
save_tag = 'PSD_Day'


########################
###    Seed Values   ###
########################
c = 299792.458

ma = 5.5e5*2*np.pi
A = 8000.0*26
PSDback= 163539.36

v0_Halo = 220.0
vDotMag_Halo = 232.36
alpha_Halo = .49
tbar_Halo = 72.40

v0_Stream = 10.0
vDotMag_Stream = 418.815
alpha_Stream = 0.65903 
tbar_Stream = 279.51
fracStream = 0.05


##############################
###   Data Ouput Options   ###
##############################
freqs = np.linspace(.99998, 1.00002, 10000)*5.5e5 
dayToGen = sys.argv[1]
includeGF = 0.0


#############################
###   Generate the Data   ###
#############################
gen = dataGen.Generator(ma, A, PSDback, v0_Halo, vDotMag_Halo,
                        alpha_Halo, tbar_Halo, v0_Stream, vDotMag_Stream,
                        alpha_Stream, tbar_Stream, fracStream, freqs)

gen.makePSDFast(float(dayToGen), includeGF)

PSD = gen.PSDArray
np.save(PSD_dir + save_tag + str(dayToGen) + '.npy', PSD)

if int(dayToGen) == 0:
    np.save(data_dir + 'freqs.npy', freqs)

