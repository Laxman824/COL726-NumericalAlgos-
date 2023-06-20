import numpy as np
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['lines.color'] = 'k'
mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color', ['k'])

n = 100
p = 30
A = np.random.rand(p,n)
while (np.linalg.matrix_rank(A) != p):
	A = np.random.rand(p,n)
x = np.random.rand(n,1)
B = np.dot(A,x)

# pickle.dump(A,open('A.p','wb'))
# pickle.dump(x,open('x.p','wb'))
# pickle.dump(B,open('B.p','wb'))

# A = pickle.load( open("A1.p", "rb"))
# B = pickle.load( open("B.p", "rb"))
# x = pickle.load( open("x.p", "rb"))

alpha = 0.5
beta = 0.01

def f(v):
	return ((np.dot(np.transpose(v),np.log(v))))[0][0]

def df(v):
	return (np.log(v))+1

def ddf(v):
	return np.multiply(np.transpose(np.reciprocal([float(i) for i in v])),np.eye(n))

def newtonMethod(xin):
	ini = xin
	diff = np.linalg.norm(ini)
	iteration = 0
	plt.plot(iteration,f(xin),'bo',markersize=2)
	
	while (diff > 0.0000000000001):
		P = np.concatenate((A,np.zeros((p,p))),axis=1)
		Q = np.concatenate((ddf(xin),np.transpose(A)),axis=1)
		R = np.concatenate((P,Q),axis=0)
		S = np.concatenate((np.zeros((p,1)),-(df(xin))),axis=0)
		T = np.linalg.solve(R,S)
		h = T[0:n,:]
		eta = 1
		while f(xin + eta*h) > f(xin) - eta*alpha*np.dot(np.transpose(h),h):
			eta = eta*beta
		xin = xin + eta*h
		iteration += 1
		plt.plot(iteration,f(xin),'bo',markersize=2)
		final = xin
		diff = np.linalg.norm(np.subtract(ini,final))
		ini = final
	plt.xlabel('Iteration Number')
	plt.ylabel('f(x)')
	plt.title('Newton\'s Method')
	plt.show()
	return xin

xfin = newtonMethod(x)
print (f(xfin))
