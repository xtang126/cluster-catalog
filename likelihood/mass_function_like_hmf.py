# likelihood/mass_function_like_hmf.py
# manual likelihood using fixed synthetic data

import numpy as np
import os
from cosmosis.datablock import names, option_section
from hmf import MassFunction
import matplotlib.pyplot as plt


def setup(options):
    z_list_str = options.get_string(option_section, "z_indices", default="0")
    z_indices = [int(z.strip()) for z in z_list_str.split(",")]
    rel_sigma = options.get_double(option_section, "relative_sigma", default=0.01)

    obs_data = []
    sigma_data = []
    mass_data = []

    for z_index in z_indices:
        z = 0.1 * z_index
        mf = MassFunction(
            z=z,
            sigma_8=0.8,  # fixed fiducial for generating synthetic data
            cosmo_params={"H0": 70, "Om0": 0.3, "Ob0": 0.04},
            Mmin=13,
            Mmax=15,
            dlog10m=0.4,
            hmf_model="Tinker08",
            n=0.96,
        )
        sigma = rel_sigma * mf.dndlnm

        obs_data.append(mf.dndlnm)
        sigma_data.append(sigma)
        mass_data.append(mf.m)

    return z_indices, obs_data, sigma_data, mass_data


def execute(block, config):
    z_indices, obs_data, sigma_data, mass_data = config

    chi2 = 0.0

    for i, z_index in enumerate(z_indices):
        z = 0.1 * z_index
        mf = MassFunction(
            z=z,
            sigma_8=block[names.cosmological_parameters, "sigma8_input"],
            cosmo_params={"H0": 70, "Om0": block[names.cosmological_parameters, "omega_m"], "Ob0": 0.04},
            Mmin=13,
            Mmax=15,
            dlog10m=0.4,
            hmf_model="Tinker08",
            n=0.96,
        )

        model_mf = mf.dndlnm
        chi2 += np.sum(((model_mf - obs_data[i]) / sigma_data[i]) ** 2)

        # Plot
        #plt.errorbar(np.log10(mass_data[i]), obs_data[i], yerr=sigma_data[i], fmt='o', label=f"Data z={z:.1f}")
        #plt.plot(np.log10(mass_data[i]), model_mf, '-', label=f"Model z={z:.1f}")
    '''
    plt.xlabel("log10(Mass) [Msun/h]")
    plt.ylabel("dn/dlnM")
    plt.legend()
    plt.title("Mass Function Model vs. Data")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("mf_xt/debug_outputs/mf_model_vs_data.png")
    plt.close()
    '''
    like = -0.5 * chi2
    print(f"Likelihood: {like}")
    block["likelihoods", "mass_function_xt"] = like
    return 0


def cleanup(config):
    pass
