import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg as la

data = np.genfromtxt("data.csv", delimiter=",")

print(data)

x1 = data[:, 0]
y1 = data[:, 1]
x2 = data[:, 2]
y2 = data[:, 3]
x3 = data[:, 4]
y3 = data[:, 5]
x4 = data[:, 6]
y4 = data[:, 7]

print(x1)
print(x3)
print(y4)


print("Srednie")
print(np.mean(y1))
print(np.mean(y2))
print(np.mean(y3))
print(np.mean(y4))
print('\n'*3)
print("Wariancje")
print(np.var(y1))
print(np.var(y2))
print(np.var(y3))
print(np.var(y4))

x_line = np.linspace(4, 15, 100)


def compute_values(x, y):
    return (np.mean(y), np.var(y))


def compute_y_fit(x, y):
    A = np.vstack([x**0, x**1]).T
    sol, r, rank, sv = la.lstsq(A, y1)
    y_fit = sol[0] + sol[1]*x_line
    return y_fit


A = np.vstack([x1**0, x1**1]).T

print(A)


fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(x1, y1, 'o', label='$x_1')
ax.plot(x_line, y_fit, 'k--')
plt.show()
