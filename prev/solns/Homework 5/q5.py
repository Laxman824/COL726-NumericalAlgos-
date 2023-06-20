import numpy as np
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['lines.color'] = 'k'
mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color', ['k'])


m = 5
n = 7
A = np.random.rand(m,n)
# A = np.array([[0.95076275, 0.83236336, 0.34537922, 0.24690624, 0.04748528],
#                   [0.37370073, 0.21605892, 0.11763723, 0.25192572, 0.68245852],
#                   [0.51450851, 0.61823476, 0.30407372, 0.02861826, 0.96630316],
#                   [0.08451906, 0.91803414, 0.22658561, 0.78210196, 0.05865102],
#                   [0.36197798, 0.13480535, 0.81397247, 0.35540204, 0.99090677],
#                   [0.09197906, 0.11617436, 0.42917253, 0.16267329, 0.84262108],
#                   [0.8779699, 0.31353977, 0.89347705, 0.62908374, 0.42251657]])
x = np.zeros((n,1))

alpha = 0.5
beta = 0.1

# pickle.dump(A,open('A.p','wb'))
# pickle.dump(x,open('x.p','wb'))
# pickle.dump(B,open('B.p','wb'))

# A = pickle.load( open("A.p", "rb"))
# B = pickle.load( open("B.p", "rb"))
# x = pickle.load( open("x.p", "rb"))

def f(v):
	p = - np.sum(np.log(1 - (np.dot(A,v))))
	q = - np.sum(np.log(1 - (np.square(v))))
	return (p+q)

def df(v):
	p = np.dot(np.transpose(A),(np.reciprocal(1 - (np.dot(A,v)))))
	q = np.multiply(np.multiply(2,x),(np.reciprocal(1-(np.square(x)))))
	return p+q

def ddf(v):
	ans = np.zeros((n,n))
	for i in range(n):
		for j in range(n):
			if (i == j):
				s = np.dot(np.transpose(np.square(A[:,i])),np.square(np.reciprocal(1 - (np.dot(A,v)))))
				s += (2 + 2*(v[i]**2))/((1-(v[i]**2))**2)
				ans[i][j] = s
			else:
				s = np.dot(np.transpose(np.multiply(A[:,i],A[:,j])),np.square(np.reciprocal(1 - (np.dot(A,v)))))
				ans[i][j] = s
	return ans

# (a)

objectivefn = []
iterations = []
stepsize = []
def gradientDescent(xin):
	ini = xin
	diff = 1
	i = 0
	while (diff > 0.0000000000000001):
		h = -df(xin)
		eta = 0.2
		while f(xin + eta*h) > f(xin) - eta*alpha*np.dot(np.transpose(h),h):
			eta = eta*beta
		xin = xin + eta*h
		final = xin
		diff = np.linalg.norm(np.subtract(ini,final))
		ini = final
		i += 1
		objectivefn.append(f(xin))
		iterations.append(i)
		stepsize.append(eta)
	print "Number of iterations: ", i
	plt.plot(iterations,objectivefn,'ro', markersize=1)
	plt.xlabel('Iteration Number')
	plt.ylabel('f(x)')
	plt.title('Gradient Descent Method')
	plt.show()
	plt.plot(iterations,objectivefn-f(xin),'ro', markersize=1)
	plt.xlabel('Iteration Number')
	plt.ylabel('f(x)-f(x*)')
	plt.title('Gradient Descent Method')
	plt.show()
	plt.plot(iterations,stepsize,'ro', markersize=1)
	plt.xlabel('Iteration Number')
	plt.ylabel('Stepsize')
	plt.title('Gradient Descent Method')
	plt.show()
	return xin

print "(a) Gradient Descent:"
xfin = gradientDescent(x)
print "f(x*) = ", f(xfin)
print ""

# (b)

objectivefn = []
iterations = []
stepsize = []
def newtonMethod(xin):
	ini = xin
	diff = 1
	i = 0
	while (diff > 0.000000000000000001):
		ddx = ddf(xin)
		dx = df(xin)
		h = - np.linalg.solve(ddx, dx)
		eta = 0.2
		while f(xin + eta*h) > f(xin) + eta*alpha*np.dot(np.transpose(dx),h):
			eta = eta*beta
		xin = xin + np.multiply(eta,h)
		final = xin
		diff = np.linalg.norm(np.subtract(ini,final))
		ini = final
		i += 1
		objectivefn.append(f(xin))
		iterations.append(i)
		stepsize.append(eta)
	print "Number of iterations: ", i
	plt.plot(iterations,objectivefn,'ro', markersize=1)
	plt.xlabel('Iteration Number')
	plt.ylabel('f(x)')
	plt.title('Newton\'s Method')
	plt.show()
	plt.plot(iterations,objectivefn-f(xin),'ro', markersize=1)
	plt.xlabel('Iteration Number')
	plt.ylabel('f(x)-f(x*)')
	plt.title('Newton\'s Method')
	plt.show()
	plt.plot(iterations,stepsize,'ro', markersize=1)
	plt.xlabel('Iteration Number')
	plt.ylabel('Stepsize')
	plt.title('Newton\'s Method')
	plt.show()
	return xin

print "(b) Newton Method: "
xfin1 = newtonMethod(x)
print "f(x*) = ", f(xfin1)