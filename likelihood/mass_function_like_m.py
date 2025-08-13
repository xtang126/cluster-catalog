# mass_function_like_hmf_vectorized.py

import numpy as np
from cosmosis.datablock import names, option_section
from hmf import MassFunction
import scipy.interpolate
import scipy.linalg

def setup(options):
    z_list_str = options.get_string(option_section, "z_indices", default="1")
    z_indices = [int(z.strip()) for z in z_list_str.split(",")]
    rel_sigma = options.get_double(option_section, "relative_sigma", default=0.05)

    all_mass = []
    all_dndlnm = []
    all_sigma = []

    for z_idx in z_indices:
        z = 0.1 * z_idx
        mf = MassFunction(
            z=z,
            sigma_8=0.8,
            cosmo_params={"H0": 70, "Om0": 0.3, "Ob0": 0.04},
            Mmin=13,
            Mmax=15,
            dlog10m=0.4,
            hmf_model="Tinker08",
            n=0.96,
        )

        mass = mf.m
        dndlnm = mf.dndlnm
        sigma = rel_sigma * dndlnm

        all_mass.append(mass)
        all_dndlnm.append(dndlnm)
        all_sigma.append(sigma)

    return {
        "z_indices": z_indices,
        "data_mass": np.concatenate(all_mass),
        "data_dndlnm": np.concatenate(all_dndlnm),
        "data_sigma": np.concatenate(all_sigma),
        "mf_cache": {}
    }

def execute(block, config):
    omega_m = block[names.cosmological_parameters, "omega_m"]
    sigma8 = block[names.cosmological_parameters, "sigma8_input"]

    model_all = []
    idx = 0

    for z_idx in config["z_indices"]:
        z = 0.1 * z_idx
        key = (round(z, 4), round(omega_m, 6), round(sigma8, 6))

        if key in config["mf_cache"]:
            model_interp = config["mf_cache"][key]
        else:
            mf = MassFunction(
                z=z,
                sigma_8=sigma8,
                cosmo_params={"H0": 70, "Om0": omega_m, "Ob0": 0.04},
                Mmin=13,
                Mmax=15,
                dlog10m=0.4,
                hmf_model="Tinker08",
                n=0.96,
            )
            model_interp = scipy.interpolate.interp1d(
                np.log(mf.m), mf.dndlnm,
                kind="linear", bounds_error=False, fill_value="extrapolate"
            )
            config["mf_cache"][key] = model_interp

        # Get number of bins for this z
        n = len(config["data_mass"]) // len(config["z_indices"])
        mass_slice = config["data_mass"][idx:idx + n]
        model_vals = model_interp(np.log(mass_slice))
        model_all.append(model_vals)
        idx += n

    model_vec = np.concatenate(model_all)
    obs_vec = config["data_dndlnm"]
    sigma_vec = config["data_sigma"]

    # Chi-square and log-likelihood
    residual = (obs_vec - model_vec) / sigma_vec
    chi2 = np.sum(residual ** 2)
    loglike = -0.5 * chi2

    block["likelihoods", "mass_function_like_like"] = loglike
    return 0

def cleanup(config):
    return 0
