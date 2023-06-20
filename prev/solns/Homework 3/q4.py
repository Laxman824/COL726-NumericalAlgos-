import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt

A = np.matrix(np.ones((5,2)))
A[:,0] = A1 = [[-2.3], [-4.5], [1.3], [2.1], [-10.5]]

normVector = np.transpose(np.matrix(np.linalg.norm(A, axis = 1)))
A = np.divide(A, normVector)

B = np.matrix([[-0.6], [-5.7], [7], [10], [-24]])
B = np.divide(B, normVector)

Q, R = np.linalg.qr(A)

X = np.linalg.solve(R, np.matmul(np.transpose(Q),B))
print "x1 = ", X[0,0]
print "x2 = ", X[1,0]

mpl.rcParams['lines.color'] = 'k'
mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color', ['k'])

x = np.linspace(0, 8, 400)
y = np.linspace(0, 6, 400)
x, y = np.meshgrid(x, y)

def axes():
    plt.axhline(0, alpha=.1)
    plt.axvline(0, alpha=.1)

axes()
l1 = plt.contour(x, y, (2.3*x - y - 0.6), [0], colors='red')
l1.collections[0].set_label('y = 2.3x - 0.6')
l2 = plt.contour(x, y, (4.5*x - y - 5.7), [0], colors='yellow')
l2.collections[0].set_label('y = 4.5x - 5.7')
l3 = plt.contour(x, y, (-1.3*x - y + 7), [0], colors='green')
l3.collections[0].set_label('y = -1.3x + 7')
l4 = plt.contour(x, y, (-2.1*x - y + 10), [0], colors='violet')
l4.collections[0].set_label('y = -2.1x + 10')
l5 = plt.contour(x, y, (10.5*x - y - 24), [0], colors='maroon')
l5.collections[0].set_label('y = 10.5x -24')
plt.plot(X[0],X[1],'bo', label=('({},{})').format(round(X[0,0],5),round((X[1,0]),5)))
plt.xlabel('x')
plt.ylabel('y')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
plt.legend()
plt.show()