import numpy as np
from scipy import optimize

# (a)

A = np.matrix([[6, 2, 1], [2, 3, 1], [1, 1, 1]])
xA = np.matrix([[0], [0], [1]])
epsilon = 0.00000000001

print "(a)"
ini = xA
diff = np.linalg.norm(ini)
while (diff >= epsilon):
	xA = np.dot(A, xA)
	xA = xA / np.linalg.norm(xA)
	final = xA
	diff = np.linalg.norm(np.subtract(ini,final))
	ini = final
	
lambd = np.dot(np.transpose(xA), np.dot(A, xA))	
eigenvalueA = lambd[0,0]
print "Largest Eigenvalue:",eigenvalueA
print "Corresponding Eigenvector:"
print xA
print ""

B = np.linalg.inv(A)
xB = np.matrix([[0], [0], [1]])

ini = xB
diff = np.linalg.norm(ini)
while (diff >= epsilon):
	xB = np.dot(B, xB)
	xB = xB / np.linalg.norm(xB)
	final = xB
	diff = np.linalg.norm(np.subtract(ini,final))
	ini = final
	
lambd = np.dot(np.transpose(xB), np.dot(B, xB))	
eigenvalueB = 1/(lambd[0,0])
print "Smallest Eigenvalue:",eigenvalueB
print "Corresponding Eigenvector:"
print xB
print ""

# (b)

A = np.array(A)
def f_lagrange(x):
	return np.array(list(2.0 * (np.matmul(A, x[:-1]) - x[-1] * x[:-1])) + [1 - np.linalg.norm(x[:-1])**2])

minVal = optimize.root(f_lagrange, np.array([1, 1, 1, 0]))
minEigenvector = minVal.x[:-1]
minEigenvalue = minVal.x[-1]
maxVal = optimize.root(f_lagrange, np.array([1, 1, 1, 10]))
maxEigenvector = maxVal.x[:-1]
maxEigenvalue = maxVal.x[-1]

print "(b)"
print "Smallest Eigenvalue: ", minEigenvalue
print "Corresponding Eigenvector: ", minEigenvector
print ""
print "Largest Eigenvalue: ", maxEigenvalue
print "Corresponding Eigenvector: ", maxEigenvector