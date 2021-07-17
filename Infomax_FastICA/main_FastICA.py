# Simple implementation of the Infomax FastICA algorithm
# Number of Sources == Number of Mixtures == 2

from scipy.io import wavfile
from scipy.io.wavfile import write


import numpy as np # Operations matricielles
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

_, s1 = wavfile.read('wavfile.wav')
_, s2 = wavfile.read('wavfile2.wav')

s1=s1[0:74414]

s=np.c_[s1,s2]

A = np.array(([0.5,0.9],[1,0.2]))

X = np.dot(s,A.T)
soun= X[:,0]

ica = FastICA(n_components=2)
S_estime = ica.fit_transform(X)  # Reconstruct signals
A_estime = ica.mixing_  # Get estimated mixing matrix

#scaled = np.int16(S_estime/np.max(np.abs(data)) * 32767)

write('test.wav', 8000, s1)

write('teslllt.wav', 8000, soun.T)
