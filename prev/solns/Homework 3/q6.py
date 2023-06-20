import numpy as np
import math

A = np.tril(np.multiply(-1,np.ones((60,60))),-1) + np.identity(60)
for i in range(60):
	A[i,59] = 1 

x = np.ones((60,1))
print "Original X = ", np.array(np.transpose(x)) 
B = np.matmul(A,x)

x1 = np.linalg.solve(A,B)
print "Recalculated X = ", np.array(np.transpose(x1))

error = (np.linalg.norm(x-x1)/np.linalg.norm(x))
print "Error = ", error