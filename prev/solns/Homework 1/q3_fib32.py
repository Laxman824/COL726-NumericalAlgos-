# Fibonacci in single precision

import math
import numpy as np
import matplotlib.pyplot as plt

# a) fn and pn on a graph
def fibonacci(n):
	fib = [np.float32(1)]*2
	if n < 2:
		return fib
	for i in range(2,n+1):
		x = np.float32(fib[i-1] + fib[i-2])
		fib.append(x)
	return fib

def pibonacci(n):
	pib = [np.float32(1)]*2
	if n < 2:
		return pib
	for i in range(2,n+1):
		x = np.float32((pib[i-1] * np.float32(1 + math.sqrt(3)/100.0)) + pib[i-2])
		pib.append(x)
	return pib

n = 100
fib = fibonacci(n)
pib = pibonacci(n)
x = range(0,n+1)
sprecision = [np.float32(math.pow(2,23))]*(n+1)
dprecision = [np.float32(math.pow(2,52))]*(n+1)
plt.plot(x, fib, 'r', label='log fn')
plt.plot(x, pib, 'b', label='log pn')
plt.yscale('log')
plt.plot(x, sprecision, 'g', label='1/Emach, 32bits')
plt.plot(x, dprecision, 'orange', label='1/Emach, 64bits')
plt.xlabel('n')
plt.ylabel('log fn or log pn')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
plt.show()

# b) Difference while back calculation

def fibonacciback(l,n):
	if n < 2:
		return 0
	i = n-1
	low = l[n-1]
	high = l[n]
	while (i > 0):
		temp = high-low
		high = np.float32(low)
		low = np.float32(temp)
		i -= 1
	return low

fibback = [np.float32(abs(fibonacciback(fib,i)-fib[0])) for i in xrange(0,n+1)]
plt.plot(x, fibback, 'r')
plt.xlabel('n')
plt.ylabel('abs(f0(original) - f0(calculated))')
plt.show()

# c) Difference while back calculation

def pibonacciback(l,n):
	if n < 2:
		return 0
	i = n-1
	low = l[n-1]
	high = l[n]
	while (i > 0):
		temp = high - (low * np.float32(1 + math.sqrt(3)/100.0))
		high = np.float32(low)
		low = np.float32(temp)
		i -= 1
	return low

pibback = [np.float32(abs(pibonacciback(pib,i)-pib[0])) for i in xrange(0,n+1)]
plt.plot(x, pibback, 'b')
plt.xlabel('n')
plt.ylabel('abs(f0(original) - f0(calculated))')
plt.show()
