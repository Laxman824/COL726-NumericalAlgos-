import math
import numpy
import matplotlib.pyplot as plt

x = numpy.linspace(0,10,1000)

def pifunc(x,p1):
	y = (math.pi * p1)/(p1 + (math.pi-p1)*(math.exp(math.pow(x,2))))
	return y

# pi: 8 digit rounded approximation
p = 3.14159265
y = [pifunc(i,p) for i in x]

# pi: 9 digit rounded approximation
p = 3.141592654
y1 = [pifunc(i,p) for i in x]

plt.plot(x, y, 'r', label='y0: 8 digit approx.')
plt.plot(x, y1, 'b', label='y0: 9 digit approx.')
plt.ylim(-30,30)
plt.xlabel('x')
plt.ylabel('y(x)')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
plt.show()