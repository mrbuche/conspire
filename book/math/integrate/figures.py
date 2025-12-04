from conspire.math.integrate import BogackiShampine
import matplotlib.pyplot as plt
import numpy as np


integrator = BogackiShampine(abs_tol=1e-3, rel_tol=1e-3)
t, y, _ = integrator.integrate(
    lambda t, y: [y[1], (1 - y[0] ** 2) * y[1] - y[0]], [0, 20], [2, 0]
)

plt.plot(t, y, "o-", markerfacecolor="none")
plt.xlim([0, 20])
plt.ylim([-3, 3])

plt.savefig("integrate/explicit/bogacki_shampine.svg", transparent=True)
