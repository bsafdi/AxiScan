import numpy as np
import os

data_dir = '/nfs/turbo/bsafdi/fosterjw/github/ABRA-DATA/Data/'
PSD_dir = '/nfs/turbo/bsafdi/fosterjw/github/ABRA-DATA/Data/PSD_Data/'

freqs = np.load(data_dir + 'freqs.npy')
PSD = np.zeros(freqs.shape)

fileList = os.listdir(PSD_dir)

count = 0
for fileName in fileList:
    print(fileName)
    PSD += np.load(PSD_dir + fileName)
    count += 1
    print(count)

num_stacked = len(fileList)
print(num_stacked)


np.save(data_dir + 'Stacked_PSD' + str(num_stacked) + '.npy', PSD/num_stacked)




