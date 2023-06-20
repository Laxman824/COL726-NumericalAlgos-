import math
import copy
import numpy as np
import matplotlib as mpl
import scipy.io as spio
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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
# Part (a)

x1 = [1.02, 0.95, 0.87, 0.77, 0.67, 0.56, 0.44, 0.30, 0.16, 0.01]
y1 = [0.39, 0.32, 0.27, 0.22, 0.18, 0.15, 0.13, 0.12, 0.13, 0.15]
A = [np.square(y1), np.multiply(x1,y1), x1, y1, np.ones(np.shape(x1))]
A = np.transpose(np.array(A))
b = np.square(x1)

coeff = leastSquare(A,b)
print (coeff)

mpl.rcParams['lines.color'] = 'k'
mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color', ['k'])

x = np.linspace(-2, 2, 400)
y = np.linspace(-2, 2, 400)
x, y = np.meshgrid(x, y)

def axes():
    plt.axhline(0, alpha=.1)
    plt.axvline(0, alpha=.1)

axes()
plt.contour(x, y, (coeff[0]*y**2 + coeff[1]*x*y + coeff[2]*x + coeff[3]*y + coeff[4] - x**2), [0], colors='k', label='Elliptic Orbit')
plt.plot(x1,y1,'ro', label='Given sample points')
plt.contour(x, y, (coeff[0]*y**2 + coeff[1]*x*y + coeff[2]*x + coeff[3]*y + coeff[4] - x**2), [0], colors='k', label='Elliptic Orbit')
plt.xlabel('Input points')
plt.ylabel('Plot')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)


# Part (b)

perturbx = np.random.uniform(-0.005,0.005,10)
perturby = np.random.uniform(-0.005,0.005,10)
x2 = x1+perturbx
y2 = y1+perturby

A1 = [np.square(y2), np.multiply(x2,y2), x2, y2, np.ones(np.shape(x2))]
A1 = np.transpose(np.array(A1))
b1 = np.square(x2)

coeff1 = leastSquare(A1,b1)
plt.contour(x, y, (coeff1[0]*y**2 + coeff1[1]*x*y + coeff1[2]*x + coeff1[3]*y + coeff1[4] - x**2), [0], colors='k', label='Elliptic Orbit')
print (coeff1)
plt.show()
