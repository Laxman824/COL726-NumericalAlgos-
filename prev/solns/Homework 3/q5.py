import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt

t = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
y = [6.80, 3.00, 1.50, 0.75, 0.48, 0.25, 0.20, 0.15]
x1 = t
y1 = y

# (a)

# Assume some initial values for x1 & x2 and then apply gradient descent
xA = [0, 0]

def computeJ(x1, x2, t, y):
	e = np.exp(np.multiply(x2, t))
	v = np.multiply(np.multiply(2, np.subtract(np.multiply(x1, e),y)), e)
	v1 = np.multiply(v, np.multiply(x1,t))
	J = [np.sum(v), np.sum(v1)]
	return J

n_iter = 0
eta = 0.02
delta = [10, 10]

while (abs(np.linalg.norm(delta)) > 0.0000001 and n_iter<=10000):	
	delta = computeJ(xA[0], xA[1], t, y)
	xA -= np.multiply(eta, delta)
	n_iter += 1

print "(x1,x2) = (", xA[0], ",", xA[1], ")"

error1 = 0
for i in range(len(t)):
	error1 += math.pow((y[i] - (xA[0] * math.exp(xA[1]*t[i]))),2)
print "Error in Gradient Descent: ", error1


# (b)

B = np.transpose(np.matrix(np.log(y)))
A = np.transpose(np.ones((2,len(y))))
A[:,1] = t

Q, R = np.linalg.qr(A)

xB = np.linalg.solve(R, np.matmul(np.transpose(Q),B))
xB[0,0] = np.exp(xB[0,0])

print "(x1,x2) = (", xB[0,0], ",", xB[1,0], ")"

error2 = 0
for i in range(len(t)):
	error2 += math.pow((y[i] - (xB[0] * math.exp(xB[1]*t[i]))),2)

print "Error in Linear Least Squares: ", error2

# Graph
mpl.rcParams['lines.color'] = 'k'
mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color', ['k'])

x = np.linspace(-1, 6, 400)
y = np.linspace(-1, 10, 400)
x, y = np.meshgrid(x, y)

def axes():
    plt.axhline(0, alpha=.1)
    plt.axvline(0, alpha=.1)

axes()
l1 = plt.contour(x, y, (y - xA[0] * (math.e ** (xA[1]*x))), [0], colors='red')
l1.collections[0].set_label('Non-linear Least Squares')
l2 = plt.contour(x, y, (y - xB[0,0] * (math.e ** (xB[1,0]*x))), [0], colors='green')
l2.collections[0].set_label('Linear Least Squares')
plt.plot(x1,y1,'bo')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
plt.legend()
plt.show()