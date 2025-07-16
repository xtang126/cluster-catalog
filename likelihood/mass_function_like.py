# likelihood/mass_function_like.py
# manually compute likelihood for mass function data

import numpy as np
import os
from cosmosis.datablock import names
from cosmosis.datablock import option_section

mf = names.mass_function

def setup(options):
    section = option_section
    mass_file = options.get_string(section, "mass_file")
    dndlnmh_file = options.get_string(section, "dndlnmh_file")
    z_index = options.get_int(section, "z_index", 0)
    rel_sigma = options.get_double(section, "relative_sigma", 0.01)
    use_poisson = options.get_bool(section, "use_poisson", False)
    os.makedirs("mf_xt/debug_outputs", exist_ok=True)
    
   # Load observed data
    m_h = np.loadtxt(mass_file)
    dndlnmh = np.loadtxt(dndlnmh_file)[z_index, :]
    sigma = np.maximum(rel_sigma * dndlnmh, 1e-3)
    #norm = 0.5 * np_sum(log(2 * np.pi * sigma**2))

    return m_h, dndlnmh, sigma, z_index, use_poisson #, norm

def execute(block, config):
    m_h, dndlnmh_obs, sigma, z_index, use_poisson = config

    m_model = block[mf, "m_h"]
    mf_model = block[mf, "dndlnmh"][z_index, :]

    if not np.allclose(m_h, m_model):
        raise ValueError("Model and observation mass bins do not match!")

    # Save debug output
    np.savetxt("mf_xt/debug_outputs/mass_bins.txt", m_h)
    np.savetxt("mf_xt/debug_outputs/observed_dndlnmh.txt", dndlnmh_obs)
    np.savetxt("mf_xt/debug_outputs/model_dndlnmh.txt", mf_model)

    if use_poisson:
        eps = 1e-10
        d = np.maximum(dndlnmh_obs, eps)
        m = np.maximum(mf_model, eps)
        logL = np.sum(d * np.log(m) - m - np.log(d))
    else:
        chi2 = np.sum((mf_model - dndlnmh_obs)**2 / sigma**2)
        logL = -0.5 * chi2

    block[names.likelihoods, "mass_function_xt"] = logL
    return 0

def cleanup(config):
    return 0


