# ruff: noqa: I001
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ANCHOR: snippet
from conspire.physics.molecular.single_chain import (
    ArbitraryPotentialFreelyJointedChain,
    ExtensibleFreelyJointedChain,
    Potential,
)
from conspire.physics import BOLTZMANN_CONSTANT, ROOM_TEMPERATURE
from conspire.math.special import langevin
import numpy as np

for kappa in [10, 25, 100, 1000]:
    eta = np.geomspace(1e-2, 0.3 * kappa, 100)
    efjc = ExtensibleFreelyJointedChain(
        ensemble="isotensional",
        link_length=1,
        link_stiffness=kappa * BOLTZMANN_CONSTANT * ROOM_TEMPERATURE,
        number_of_links=3,
        temperature=ROOM_TEMPERATURE,
    )
    ufjc = ArbitraryPotentialFreelyJointedChain(
        ensemble="isotensional",
        number_of_links=3,
        potential=Potential.Harmonic(
            rest_length=1,
            stiffness=kappa * BOLTZMANN_CONSTANT * ROOM_TEMPERATURE,
        ),
        temperature=ROOM_TEMPERATURE,
    )
    gamma_efjc = np.zeros_like(eta)
    gamma_ufjc = np.zeros_like(eta)
    gamma_aprx = np.zeros_like(eta)
    for i, eta_i in enumerate(eta):
        gamma_efjc[i] = efjc.nondimensional_extension(eta_i)
        gamma_ufjc[i] = ufjc.nondimensional_extension(eta_i)
        gamma_aprx[i] = langevin(eta_i) + eta_i / kappa
    # ANCHOR_END: snippet
    plt.plot(gamma_efjc, eta / kappa, "#000005")
    plt.plot(gamma_ufjc, eta / kappa, ":", color="#000002", linewidth=2)
    plt.plot(gamma_aprx, eta / kappa, "--", color="#000002")

ax = plt.gca()
ax.set_xlabel("$\\gamma(\\eta)$", fontsize=12)
ax.set_ylabel("$\\eta/\\kappa$", fontsize=12)

ax.xaxis.label.set_color("#000002")
ax.yaxis.label.set_color("#000002")
ax.tick_params(axis="x", colors="#000002")
ax.tick_params(axis="y", colors="#000002")

ax.legend(
    handles=[
        Line2D([], [], color="#000005", linestyle="-", label="exact"),
        Line2D([], [], color="#000002", linestyle=":", label="asymptotic"),
        Line2D([], [], color="#000002", linestyle="--", label="reduced"),
    ],
    labelcolor="#000002",
    frameon=False,
)

plt.savefig("figure_1.svg", transparent=True)
