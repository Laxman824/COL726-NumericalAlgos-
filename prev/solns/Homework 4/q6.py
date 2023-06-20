import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.special import gamma
mpl.rcParams['lines.color'] = 'k'
mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color', ['k'])

def axes():
    plt.axhline(0, alpha=.1)
    plt.axvline(0, alpha=.1)

def func(x, i):
	if (i==1):
		return (x**4) - (14*(x**3)) + (60*(x**2)) - (70*x)
	elif	(i==2):
		return (0.5*(x**2)) - (np.sin(x))
	elif (i==3):
		return (x**2) + (4*np.cos(x))
	else:
		return gamma(x)

def findMin(i):
	xl = 0
	xh = 3
	r = 0.618
	while (xh-xl>0.0001):
		x1 = xh - (r * (xh - xl))
		x2 = xl + (r * (xh - xl))
		if (func(x1,i) > func(x2,i)):
			xl = x1
		else:
			xh = x2
	return (xl,func(xl,i))

a = findMin(1)
b = findMin(2)
c = findMin(3)
d = findMin(4)

print "a) ", a
print "b) ", b
print "c) ", c
print "d) ", d


################  Graphs ##########################

# (a)

x = np.linspace(0, 3, 400)
y = np.linspace(-30, 40, 400)
x, y = np.meshgrid(x, y)

axes()
l = plt.contour(x, y, ((x**4) - (14*(x**3)) + (60*(x**2)) - (70*x) - y), [0], colors='r')
l.collections[0].set_label('y = x^4 - 14x^3 + 60x^2 - 70x')
plt.plot(a[0],a[1],'bo',label=('({},{})').format(a[0],a[1]))
plt.xlabel('x')
plt.ylabel('y')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
plt.legend()
plt.show()

# (b)

x = np.linspace(0, 3, 400)
y = np.linspace(-2, 5, 400)
x, y = np.meshgrid(x, y)

axes()
l = plt.contour(x, y, ((0.5*(x**2)) - (np.sin(x)) - y), [0], colors='r')
l.collections[0].set_label('y = 0.5x^2 - sin(x)')
plt.plot(b[0],b[1],'bo',label=('({},{})').format(b[0],b[1]))
plt.xlabel('x')
plt.ylabel('y')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
plt.legend()
plt.show()

# (c)

x = np.linspace(0, 3, 400)
y = np.linspace(2, 5, 400)
x, y = np.meshgrid(x, y)

axes()
l = plt.contour(x, y, ((x**2) + (4*np.cos(x)) - y), [0], colors='r')
l.collections[0].set_label('y = x^2 + 4cos(x)')
plt.plot(c[0],c[1],'bo',label=('({},{})').format(c[0],c[1]))
plt.xlabel('x')
plt.ylabel('y')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
plt.legend()
plt.show()

# (d)

x = np.linspace(0, 3, 400)
y = np.linspace(0, 40, 400)
x, y = np.meshgrid(x, y)

axes()
l = plt.contour(x, y, (gamma(x) - y), [0], colors='r')
l.collections[0].set_label('y = gamma(x)')
plt.plot(d[0],d[1],'bo',label=('({},{})').format(d[0],d[1]))
plt.xlabel('x')
plt.ylabel('y')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
plt.legend()
plt.show()