import math
import matplotlib.pyplot as plt

def integral(n):
	val = [1 - math.exp(-1)]
	if n < 1:
		return val
	for i in range(1,n+1):
		val.append(1 - i*val[i-1])
	return val

n = 20
x = range(0,n+1)
y = integral(n)
plt.plot(x, y, 'r')
plt.xlabel('k')
plt.ylabel('I(k)')
plt.show()