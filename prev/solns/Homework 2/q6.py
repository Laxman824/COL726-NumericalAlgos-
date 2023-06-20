import math
import copy
import numpy as np
import matplotlib as mpl
import scipy.io as spio
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
mat = spio.loadmat('data.mat')
mat1 = spio.loadmat('dataclass.mat')

f = mat['fea'].toarray()
f[f>0] = 1
g = mat['gnd']
s = np.subtract(np.transpose(mat1['sampleIdx'])[0],1)
z = np.subtract(mat1['zeroIdx'],1)

freqMap = np.array([f[i,:] for i in s])
catMap = np.array([g[i,0] for i in s])

cmap = {}
t = 0
for i in range(np.shape(freqMap)[0]):
	if (cmap.get(catMap[i]) == None):
		cmap[catMap[i]] = t
		t += 1
freqMap = np.delete(freqMap, z, axis=1)

kmeans = KMeans(n_clusters=8, random_state=0).fit(freqMap)
labels = kmeans.predict(freqMap)

u, s, vh = np.linalg.svd(freqMap, full_matrices=True)
v1 = np.transpose(vh[:20, :])

freqMap1 = np.matmul(freqMap, v1)

kmeans1 = KMeans(n_clusters=8, random_state=0).fit(freqMap1)
labels1 = kmeans1.predict(freqMap1)

confusionMat = np.zeros((8,8))
for i in range(np.shape(freqMap)[0]):
	confusionMat[cmap[catMap[i]],labels[i]] += 1
for i in range(7):
	index = np.argmax(confusionMat[:,i:], axis=1)[i] + i
	temp = copy.deepcopy(confusionMat[:,index])
	confusionMat[:,index] = confusionMat[:,i]
	confusionMat[:,i] = temp
print ("Confusion Matrix without SVD\n")
print (confusionMat)

confusionMat1 = np.zeros((8,8))
for i in range(np.shape(freqMap)[0]):
	confusionMat1[cmap[catMap[i]],labels1[i]] += 1
for i in range(7):
	index = np.argmax(confusionMat1[:,i:], axis=1)[i] + i
	temp = copy.deepcopy(confusionMat1[:,index])
	confusionMat1[:,index] = confusionMat1[:,i]
	confusionMat1[:,i] = temp
print ("\nConfusion Matrix with SVD\n")
print (confusionMat1)


