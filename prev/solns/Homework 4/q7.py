import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['lines.color'] = 'k'
mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color', ['k'])

def axes():
    plt.axhline(0, alpha=.1)
    plt.axvline(0, alpha=.1)

def f(x,y):
	return (100*((y - (x**2))**2)) + ((1-x)**2)

def dfx(x,y):
	return (400*(x**3)) - (400*x*y) + (2*x) - 2

def dfy(x,y):
	return (200*y) - (200*(x**2))

def dfxx(x,y):
	return (1200*(x**2)) - (400*y) + 2

def dfxy(x,y):
	return (-400*x)

def dfyx(x,y):
	return (-400*x)

def dfyy(x,y):
	return 200


def steepestDescent(xy):
	axes()
	x = np.linspace(-1, 2, 400)
	y = np.linspace(-1, 1, 400)
	x, y = np.meshgrid(x, y)

	ini = xy
	plt.plot(xy[0],xy[1],'bo',markersize=2)
	n = 1
	diff = np.linalg.norm(ini)
	i = 1
	while (diff > 0.000000000001):
		x1 = xy - np.multiply(n, [dfx(xy[0],xy[1]), dfy(xy[0],xy[1])])
		while ((f(x1[0],x1[1])) > (f(xy[0],xy[1]))):
			n = 0.1*n
			x1 = xy - np.multiply(n, [dfx(xy[0],xy[1]), dfy(xy[0],xy[1])])	
		xy = x1
		final = xy
		diff = np.linalg.norm(np.subtract(ini,final))
		ini = final
		i += 1
		if (i%100 == 0):
			plt.plot(xy[0],xy[1],'bo',markersize=2)
	print "Number of iterations:",i
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title('Steepest Descent Method')
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
	plt.legend()
	plt.show()
	return xy

def newtonMethod(xy):
	axes()
	x = np.linspace(-1, 2, 400)
	y = np.linspace(-1, 1, 400)
	x, y = np.meshgrid(x, y)
	
	ini = xy
	plt.plot(xy[0],xy[1],'bo',markersize=2)
	diff = np.linalg.norm(ini)
	i = 0
	while (diff > 0.00000000000001):
		h = ([[dfxx(xy[0][0], xy[1][0]), dfxy(xy[0][0], xy[1][0])], [dfyx(xy[0][0], xy[1][0]), dfyy(xy[0][0], xy[1][0])]])
		delf = ([[dfx(xy[0][0], xy[1][0])], [dfy(xy[0][0], xy[1][0])]])
		s = np.linalg.solve(h, delf)
		xy = xy - s
		final = xy
		diff = np.linalg.norm(np.subtract(ini,final))
		ini = final
		plt.plot(xy[0],xy[1],'bo',markersize=2)
		i += 1
	print "Number of iterations:",i
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title('Newton Method')
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
	plt.legend()
	plt.show()
	return xy

def dampedNewton(xy):
	axes()
	x = np.linspace(-1, 2, 400)
	y = np.linspace(-1, 1, 400)
	x, y = np.meshgrid(x, y)
	
	ini = xy
	plt.plot(xy[0],xy[1],'bo',markersize=2)
	diff = np.linalg.norm(ini)
	i = 1
	while (diff > 0.00000000000001):
		h = ([[dfxx(xy[0][0], xy[1][0]), dfxy(xy[0][0], xy[1][0])], [dfyx(xy[0][0], xy[1][0]), dfyy(xy[0][0], xy[1][0])]])
		delf = ([[dfx(xy[0][0], xy[1][0])], [dfy(xy[0][0], xy[1][0])]])
		s = np.linalg.solve(h, delf)
		xy = xy - np.multiply(0.01,s)
		final = xy
		diff = np.linalg.norm(np.subtract(ini,final))
		ini = final
		i += 1
		if (i%10 == 0):
			plt.plot(xy[0],xy[1],'bo',markersize=2)
	print "Number of iterations:",i
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title('Damped Newton Method')
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
	plt.legend()
	plt.show()
	return xy

print "Steepest Descent : "
print "(-1,1): ",steepestDescent([-1,1])
print "(0,1): ",steepestDescent([0,1])
print "(2,1): ",steepestDescent([2,1])
print ""

print "Newton Method : "
print "(-1,1): ",newtonMethod([[-1], [1]])
print "(0,1): ",newtonMethod([[0], [1]])
print "(2,1): ",newtonMethod([[2], [1]])
print ""

print "Damped Newton Method : "
print "(-1,1): ",dampedNewton([[-1], [1]])
print "(0,1): ",dampedNewton([[0], [1]])
print "(2,1): ",dampedNewton([[2], [1]])