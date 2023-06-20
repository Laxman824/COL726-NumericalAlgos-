import math
import matplotlib.pyplot as plt

def diffeq(n):
	val = [(1/3.0), (1/12.0)]
	if n < 2:
		return val
	for i in range(2,n):
		x = (2.25*val[i-1] - 0.5*val[i-2])
		val.append(x)
	return val

def diffeq1(n):
	return [(math.pow(4,1-i)/3.0) for i in range(1,n+1)]

n = 80
x = range(1,n+1)
y1 = diffeq(n)
y2 = diffeq1(n)
plt.yscale('log')
plt.plot(x, y1, 'r', label='Using Difference Equation')
plt.plot(x, y2, 'b', label='Using Exact Solution')
plt.xlabel('k')
plt.ylabel('x(k)')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
plt.show()