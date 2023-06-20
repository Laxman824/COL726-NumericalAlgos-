import math
import numpy as np
import matplotlib.pyplot as plt

def poly1(x):
	x = np.float32(x)
	return np.float32(math.pow(x-1,6))

def poly2(x):
	x = np.float32(x)
	val = math.pow(x,6) - (6*math.pow(x,5)) + (15*math.pow(x,4)) - (20*math.pow(x,3)) + (15*math.pow(x,2)) - (6*math.pow(x,1)) + 1
	return np.float32(val)

x1 = range(9950,10051,1)
x = [i / 10000.0 for i in x1]
p1 = [poly1(i) for i in x]
p2 = [poly2(i) for i in x]
plt.plot(x, p1, 'r', label='Compact form')
plt.plot(x, p2, 'b', label='Expanded form')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
plt.show()