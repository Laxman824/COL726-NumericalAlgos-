import numpy as np

######### QR Helper functions #################
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

################################################
######### Power Iteration ######################

A = np.matrix([[2, 3, 2], [10, 3, 4], [3, 6, 1]])
xA = np.matrix([[0], [0], [1]])
epsilon = 0.00000000001

ini = xA
diff = np.linalg.norm(ini)
i = 0
while (diff >= epsilon):
	xA = np.dot(A, xA)
	xA = xA / np.linalg.norm(xA)
	final = xA
	diff = np.linalg.norm(np.subtract(ini,final))
	ini = final
	i += 1
	
lambd = np.dot(np.transpose(xA), np.dot(A, xA))	
eigenvalueA = lambd[0,0]
print "--------- Power Iteration ----------"
print "Final Eigenvector:"
print xA
print "Final Eigenvalue:",eigenvalueA
print "Number of Iterations:", i
print ""

########## Inverse Iteration ###################

B = np.matrix([[6, 2, 1], [2, 3, 1], [1, 1, 1]])
xB = np.matrix([[0], [0], [1]])
epsilon = 0.00000001
mu = 2

ini = xB
diff = np.linalg.norm(ini)
i = 0
while (diff >= epsilon):
	(W,R) = house(np.subtract(B, np.multiply(mu,np.identity(B.shape[0]))))
	Q = formQ(W)
	xB = np.linalg.solve(R, np.matmul(np.transpose(Q),xB))
	xB = xB / np.linalg.norm(xB)
	final = xB
	diff = np.linalg.norm(np.subtract(ini,final))
	ini = final
	i += 1
	
lambd = np.dot(np.transpose(xB), np.dot(B, xB))	
eigenvalueB = lambd[0,0]
print "--------- Inverse Iteration ----------"
print "Final Eigenvector:"
print xB
print "Final Eigenvalue:",eigenvalueB
print "Number of Iterations:", i
print ""

########## Rayleigh Quotient ####################

xB = np.matrix([[0], [0], [1]])
ini = xB
diff = np.linalg.norm(ini)
mu = 2
i = 0
while (diff >= epsilon):
	(W,R) = house(np.subtract(B, np.multiply(mu,np.identity(B.shape[0]))))
	Q = formQ(W)
	xB = np.linalg.solve(R, np.matmul(np.transpose(Q),xB))
	xB = xB / np.linalg.norm(xB)
	final = xB
	diff = np.linalg.norm(np.subtract(ini,final))
	ini = final
	mu = np.dot(np.transpose(xB), np.dot(B, xB))[0,0]
	i += 1
	
eigenvalueB = mu
print "--------- Rayleigh Quotient Iteration ----------"
print "Final Eigenvector:"
print xB
print "Final Eigenvalue:",eigenvalueB
print "Number of Iterations:", i
print ""
