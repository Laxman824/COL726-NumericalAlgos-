###chat
import numpy as np

def evaluate(c, x):
    p, q, r, s, t, u = c
    return p * x[0] ** 2 + q * x[0] * x[1] + r * x[1] ** 2 + s * x[0] + t * x[1] + u

def jacobian(c, x):
    p, q, r, s, t, u = c
    return np.array([2 * p * x[0] + q * x[1] + s, q * x[0] + 2 * r * x[1] + t])

# def newtonStep(c1, c2, x0):
#     j = np.vstack((jacobian(c1, x0), jacobian(c2, x0)))
#     f = np.array([evaluate(c1, x0), evaluate(c2, x0)])
#     return x0 - np.linalg.solve(j, f)
from scipy.optimize import fsolve

def newtonStep(c1, c2, x0):
    def f(x):
        return np.array([evaluate(c1,x), evaluate(c2,x)])
    def jack(x):
        return np.array([jacobian(c1,x), jacobian(c2,x)])
    return fsolve(f,x0,fprime=jack)

def linearize(c, x, y):
    return evaluate(c, x) + np.dot(jacobian(c, x), y - x)

# (a) Testing the implementation
c = [1, -2, -3, -4, -5, -6]  # Example c vector
x = np.array([1.5, -2.5])  # Example x vector
v = np.array([0.3, 0.2])  # Example v vector
h_values = [10 ** -i for i in range(1, 8)]  # Different h values

print("h\t\t| Difference")
print("-----------------------")
for h in h_values:
    f_x_plus_hv = evaluate(c, x + h * v)
    f_taylor = evaluate(c, x) + np.dot(jacobian(c, x), h * v)
    difference = np.abs(f_x_plus_hv - f_taylor)
    print(f"{h:.8f}\t| {difference:.18f}")

# (b) Solving the system of equations
c1 = [1, 0, 1, 0, 0, -1]  # Coefficients for the first conic
c2 = [2, 0, 1, 0, 0, -1]  # Coefficients for the second conic
x0 = [1.000000000,1.000000000]  # Initial guess

iterations = 0
tolerance = 1e-10
table = []
while iterations < 100:
    f1_value = evaluate(c1, x0)
    f2_value = evaluate(c2, x0)
    table.append([iterations, x0, f1_value, f2_value])
    x0 = newtonStep(c1, c2, x0)
    iterations += 1
    if np.abs(f1_value) < tolerance and np.abs(f2_value) < tolerance:
         break

print()
print("PartB")
print("Iteration| x\t\t\t\t| f1(x)\t| f2(x)")
print("-----------------------------------------------")
for row in table:
    iteration, x, f1_value, f2_value = row
    print(f"{iteration}\t| {x} | {f1_value}\t| {f2_value}")


print()
print("PartC")

# (c) Interpretation of linearization
x = np.array([1.5, -2.5])  # Point on the curve f(x) = 0
y = np.array([2.0, -1.0])  # Arbitrary point

f_taylor = linearize(c, x, y)
print(f"f~(y): {f_taylor}")


