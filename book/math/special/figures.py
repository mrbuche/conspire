from conspire.math.special import lambert_w, langevin, inverse_langevin

import numpy as np
import matplotlib.pyplot as plt


num = 333

x = np.linspace(-10, 10, 333)
f = np.zeros(len(x))
for i, xi in enumerate(x):
    f[i] = langevin(xi)

plt.plot(x, f, '#000005')

ax = plt.gca()
ax.set_ylim([-1, 1])
ax.set_xlabel('$x$', fontsize=12)
ax.set_ylabel('$\mathcal{L}(x)$', fontsize=12)

ax.xaxis.label.set_color('#000002')
ax.yaxis.label.set_color('#000002')
ax.tick_params(axis='x', colors='#000002')
ax.tick_params(axis='y', colors='#000002')

plt.savefig('special/langevin.svg', transparent=True)

ax.clear()

y = np.zeros(len(f))
for i, fi in enumerate(f):
    y[i] = inverse_langevin(fi)

plt.plot(f, y, '#000005')

ax = plt.gca()
ax.set_xlim([-1, 1])
ax.set_xlabel('$y$', fontsize=12)
ax.set_ylabel('$\mathcal{L}^{-1}(y)$', fontsize=12)

ax.xaxis.label.set_color('#000002')
ax.yaxis.label.set_color('#000002')
ax.tick_params(axis='x', colors='#000002')
ax.tick_params(axis='y', colors='#000002')

plt.savefig('special/inverse_langevin.svg', transparent=True)

ax.clear()

x = np.linspace(-1 / np.exp(1), 5.5, num)
w = np.zeros(len(x))
for i, xi in enumerate(x):
    w[i] = lambert_w(xi)

plt.plot(x, w, '#000005')

ax = plt.gca()
ax.set_xlim([-1, 6])
ax.set_xlabel('$x$', fontsize=12)
ax.set_ylabel('$W_0(x)$', fontsize=12)

ax.xaxis.label.set_color('#000002')
ax.yaxis.label.set_color('#000002')
ax.tick_params(axis='x', colors='#000002')
ax.tick_params(axis='y', colors='#000002')

plt.savefig('special/lambert_w.svg', transparent=True)
