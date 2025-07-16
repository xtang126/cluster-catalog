import numpy as np
import matplotlib.pyplot as plt

mass = np.loadtxt("mf_xt/debug_outputs/mass_bins.txt")
obs = np.loadtxt("mf_xt/debug_outputs/observed_dndlnmh.txt")
model = np.loadtxt("mf_xt/debug_outputs/model_dndlnmh.txt")

plt.figure(figsize=(8, 6))
plt.plot(mass, obs, label="Observed", lw=2)
plt.plot(mass, model, label="Model", lw=2, ls='--')
plt.xlabel("Mass $M_{200c}$ [$M_\\odot/h$]")
plt.ylabel("d$n$/dln$M$")
plt.xscale("log")
plt.yscale("log")
plt.xlim(1e11, 1e16)
plt.ylim(1e-13, 1e-1)
plt.legend()
plt.title("Mass Function: Observed vs. Model")
plt.grid(True)
plt.tight_layout()
plt.savefig("mf_xt/debug_outputs/mass_function_fit.png")
plt.show()
