import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['lines.color'] = 'k'
mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color', ['k'])

def axes():
	plt.axhline(0, alpha=.1)
	plt.axvline(0, alpha=.1)

def f(x):
	return math.log(math.exp(x) + math.exp(-x))

def dfx(x):
	return (math.exp(2*x) - 1)/(math.exp(2*x) + 1)

def dfxx(x):
	return (4*math.exp(2*x))/((math.exp(2*x) + 1)**2)

def newtonMethodf(xin):
	axes()
	x = np.linspace(0, 10, 400)
	y = np.linspace(0.5, 1.5, 400)
	x, y = np.meshgrid(x, y)
	
	diff = xin
	iteration = 0
	plt.plot(iteration,f(xin),'bo',markersize=2)
		
	while (diff > 0.00000000000001):
		try:
			ddf = dfxx(xin)
			df = dfx(xin)
			s = df/ddf
		except:
			plt.xlabel('Iteration Number')
			plt.ylabel('f(x)')
			plt.title('Newton\'s Method')
			plt.show()			
		plt.plot(iteration,f(xin),'bo',markersize=2)
		xin = xin - s
		diff = abs(s)
		iteration += 1 
	plt.xlabel('Iteration Number')
	plt.ylabel('f(x)')
	plt.title('Newton\'s Method')
	plt.show()
	return xin

try:
	print("(a) x(0) = 1 : ", round(newtonMethodf(1)))
except:
	print("Error")

try:
	print("(a) x(0) = 1.1 : ", newtonMethodf(1.1))
except:
	print("Error")

def g(x):
	return x - math.log(x)

def dgx(x):
	return 1 - 1/x

def dgxx(x):
	return 1/(x**2)

def newtonMethodg(xin):
	axes()
	x = np.linspace(0, 10, 400)
	y = np.linspace(0, 5, 400)
	x, y = np.meshgrid(x, y)
	
	diff = xin
	iteration = 0
	plt.plot(iteration,g(xin),'bo',markersize=2)
		
	while (diff > 0.00000000000001):
		try:
			ddg = dgxx(xin)
			dg = dgx(xin)
			s = dg/ddg
		except:
			plt.xlabel('Iteration Number')
			plt.ylabel('f(x)')
			plt.title('Newton\'s Method')
			plt.show()
		plt.plot(iteration,g(xin),'bo',markersize=2)
		xin = xin - s
		diff = abs(s)
		iteration += 1 
	plt.xlabel('Iteration Number')
	plt.ylabel('f(x)')
	plt.title('Newton\'s Method')
	plt.show()
	return xin

try:
	print ("(b) x(0) = 3 : ", newtonMethodg(3))
except:
	print ("Error")
