import math
import copy
import numpy as np
import matplotlib as mpl
import scipy.io as spio
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Gives W & R
def house(A):
	R = np.matrix(A, dtype=float)
	(m,n) = np.shape(R)
	W = np.zeros((m,n))
	for i in range(0,n):
		w = np.matrix(R[i:m,i])
		w1 = np.zeros(np.shape(w))
		w1[0] = np.linalg.norm(w)
		w += w1*np.sign(w[0])
		norm = np.linalg.norm(w)
		w = w/float(norm)
		x = np.dot(np.transpose(w),R[i:m,i:n])
		R[i:m,i:n] = np.subtract(R[i:m,i:n],np.multiply(2*x,w))
		W[i:m,[i]] = w[:,0]
	# print R
	# print W
	R = R.tolist()
	W = W.tolist()
	return (W,R)

def formQ(W):
	W = np.matrix(W, dtype=float)
	(m,n) = np.shape(W)
	Q = np.eye(m)
	for i in range(0,n):
		Q = Q.dot(np.eye(m)-(2*np.outer(W[:,i],W[:,i])))
	W = W.tolist()
	Q = Q.tolist()
	return Q
def leastSquare(A,b):
	(m,n) = np.shape(A)
	(W,R) = house(A)
	Q = formQ(W)
	R = np.matrix(R)
	Q = np.matrix(Q)
	R = R[:n,:]
	y = np.matmul(np.transpose(Q),np.transpose(np.matrix(b)))
	y = y[:n,:]
	x = np.matmul(np.linalg.inv(R),y)
	return x.tolist()
A = [[1,1],[1,1.18],[1,2.2]]
b = [1,2,3]
x = leastSquare(A,b)
print (x)
