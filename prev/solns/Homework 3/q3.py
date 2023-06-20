import numpy as np
import math
import random

m = 21
n = 12
epsilon = math.pow(10,-10)

matA = np.matrix(np.ones((m,n)))
for i in range(m):
	ti = i/float(m-1)
	temp = 1
	for j in range(n):
		matA[i,j] = temp
		temp *= ti

matX = np.ones((n,1))	# Values of xi's
# print matX

matB = np.matmul(matA,matX)  # Values of yi's

for i in range(m):
	u = random.uniform(0,1)
	matB[i] += (2*u - 1)*epsilon

# Cholesky Factorisation

Ac = np.matmul(np.transpose(matA),matA)
L = np.linalg.cholesky(Ac)

y = np.linalg.solve(L,np.matmul(np.transpose(matA),matB))
matX1 = np.linalg.solve(np.transpose(L),y)
print (matX1)
error1 = np.linalg.norm(matX1-matX)/np.linalg.norm(matX)
print ("Error in Cholesky factorisation: ", error1)

# QR Factorisation

Q, R = np.linalg.qr(matA)

matX2 = np.linalg.solve(R, np.matmul(np.transpose(Q),matB))
print (	matX2)
error2 = np.linalg.norm(matX2-matX)/np.linalg.norm(matX)
print ("Error in QR factorisation: ", error2)
