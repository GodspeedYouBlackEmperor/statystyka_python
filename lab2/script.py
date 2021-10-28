import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg as la
from scipy import stats

data = np.genfromtxt("data.csv", delimiter=",")

x1 = data[:, 0]
y1 = data[:, 1]
x2 = data[:, 2]
y2 = data[:, 3]
x3 = data[:, 4]
y3 = data[:, 5]
x4 = data[:, 6]
y4 = data[:, 7]

x_line = np.linspace(4, 15, 100)


def compute_values(x, y):
    return (np.mean(y), np.var(y), stats.pearsonr(x, y))


def print_values(x, y, index):
    (mean, var, pcor) = compute_values(x, y)
    print(
        f'Series {index} Mean: {mean}, Var: {var} Pearson Correlation: {pcor}')


def compute_y_fit(x, y, index):
    A = np.vstack([x**0, x**1]).T
    sol, r, rank, sv = la.lstsq(A, y)
    y_fit = sol[0] + sol[1]*x_line

    print(f'Series {index} Beta0: {sol[0]} Beta1: {sol[1]}')

    return y_fit


fig, axs = plt.subplots(nrows=4, figsize=(12, 16))


def plot(x, y, index):
    axs[index].set_title(f'Series: {index + 1}')
    axs[index].plot(x, y, 'o', label='$x_1')
    axs[index].plot(x_line, compute_y_fit(x, y, index + 1), 'k--')


print_values(x1, y1, 1)
print_values(x2, y2, 2)
print_values(x3, y3, 3)
print_values(x4, y4, 4)

plot(x1, y1, 0)
plot(x2, y2, 1)
plot(x3, y3, 2)
plot(x4, y4, 3)

fig.tight_layout(pad=3.0)
fig.savefig('plots.png')
