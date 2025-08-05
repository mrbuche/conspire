from conspire.math.special import lambert_w, langevin, inverse_langevin, rosenbrock

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext


num = 333

x = np.linspace(-10, 10, num)
f = np.zeros(len(x))
for i, xi in enumerate(x):
    f[i] = langevin(xi)

plt.plot(x, f, "#000005")

ax = plt.gca()
ax.set_ylim([-1, 1])
ax.set_xlabel("$x$", fontsize=12)
ax.set_ylabel("$\mathcal{L}(x)$", fontsize=12)

ax.xaxis.label.set_color("#000002")
ax.yaxis.label.set_color("#000002")
ax.tick_params(axis="x", colors="#000002")
ax.tick_params(axis="y", colors="#000002")

plt.savefig("special/langevin.svg", transparent=True)

ax.clear()

y = np.zeros(len(f))
for i, fi in enumerate(f):
    y[i] = inverse_langevin(fi)

plt.plot(f, y, "#000005")

ax = plt.gca()
ax.set_xlim([-1, 1])
ax.set_xlabel("$y$", fontsize=12)
ax.set_ylabel("$\mathcal{L}^{-1}(y)$", fontsize=12)

ax.xaxis.label.set_color("#000002")
ax.yaxis.label.set_color("#000002")
ax.tick_params(axis="x", colors="#000002")
ax.tick_params(axis="y", colors="#000002")

plt.savefig("special/inverse_langevin.svg", transparent=True)

ax.clear()

x = np.linspace(-1 / np.exp(1), 5.5, num)
w = np.zeros(len(x))
for i, xi in enumerate(x):
    w[i] = lambert_w(xi)

plt.plot(x, w, "#000005")

ax = plt.gca()
ax.set_xlim([-1, 6])
ax.set_xlabel("$x$", fontsize=12)
ax.set_ylabel("$W_0(x)$", fontsize=12)

ax.xaxis.label.set_color("#000002")
ax.yaxis.label.set_color("#000002")
ax.tick_params(axis="x", colors="#000002")
ax.tick_params(axis="y", colors="#000002")

plt.savefig("special/lambert_w.svg", transparent=True)

ax.clear()

a = 1
b = 100
levels = np.logspace(-1, 4, 6)

x = np.linspace(-2, 2, num)
y = np.linspace(-1, 3, num)
f = np.zeros((num, num))
for i, y_i in enumerate(y):
    for j, x_j in enumerate(x):
        f[i, j] = rosenbrock([x_j, y_i], a, b)


ax = plt.gca()
contour = plt.contour(x, y, f, levels, colors="#000005", linewidths=0.5)

ax.set_xlabel("$x$", fontsize=12)
ax.set_ylabel("$y$", fontsize=12)
plt.clabel(contour, levels[1:], colors="#000002", fontsize=8, inline=True)

ax.xaxis.label.set_color("#000002")
ax.yaxis.label.set_color("#000002")
ax.tick_params(axis="x", colors="#000002")
ax.tick_params(axis="y", colors="#000002")

plt.savefig("special/rosenbrock.svg", transparent=True)
