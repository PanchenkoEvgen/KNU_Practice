import numpy as np
import matplotlib.pyplot as plt

x = np.array([0, 1, 3, 10, 13])
y = np.array([10, 2, -2, 5, -4])

h = np.zeros(len(x)-1)

for i in range(len(x)-1):
    h[i] = x[i+1] - x[i]

A = np.zeros((len(h)-1,len(h)-1))

for i in range(1, len(h)):
    A[i-1][i-1] = (h[i] + h[i-1]) / 3

for i in range(1, len(h)-1):
    A[i-1][i] = h[i] / 6
    A[i][i-1] = h[i] / 6

H = np.zeros((len(h)-1,len(h)+1))

for i in range(len(h)-1):
        H[i][i] = 1/h[i]
        H[i][i+1] = -1/h[i] - 1/h[i+1]
        H[i][i+2] = 1/h[i+1]

Hf = np.matmul(H, y)
m = np.linalg.solve(A, Hf)

m = np.concatenate(([0], m, [0]))

def g(arg, i):
    A = y[i-1] - (m[i-1]*(h[i-1]**2))/6
    B = y[i] - (m[i]*h[i-1]**2)/6
    P1 = (m[i-1]*(x[i]-arg)**3 + m[i]*(arg-x[i-1])**3 )/(6*h[i-1])
    return P1 + A*(x[i]-arg)/h[i-1] + B*(arg-x[i-1])/h[i-1]

x_points = np.zeros(len(h)*1000)
y_points = np.zeros(len(h)*1000)

for i in range(1, len(x)):
     for j in range(1000):
          x_points[j+(i-1)*1000] = x[i-1]+h[i-1]*j/1000
          y_points[j+(i-1)*1000] = g(x[i-1]+h[i-1]*j/1000, i) 

# Plot the original points
plt.scatter(x, y, color='red', label="Data Points")
plt.plot(x_points, y_points, 'o', ms=0.3)

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Cubic Spline Interpolation")
plt.grid(True)
plt.show()
