# scatter_base.py
# example config file: scatter_const_fit.ini (old version)
# no scatter applied, just base poisson likelihood

'''
Custom Poisson likelihood for halo mass function:
- Compares mock cluster counts to HMF prediction (Tinker08, hmf package)
- Assumes single redshift/mass bin, selection function = 1
- Uses cached MassFunction and precomputed quantities for speed

Contact: Xin Tang (xt52@sussex.ac.uk)

'''

import numpy as np
from functools import lru_cache
from numpy.polynomial.legendre import leggauss

from cosmosis.datablock import names, option_section
from hmf import MassFunction
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from scipy.integrate import quad
from scipy.special import gammaln

h = 0.7
H0 = h * 100

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def volume_shell(zmin, zmax, Om0, area_deg2):
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
    V = (cosmo.comoving_volume(zmax) - cosmo.comoving_volume(zmin)).to(u.Mpc**3).value
    V /= h**3
    f_sky = area_deg2 / 41253.0
    return V * f_sky

def V_shell_cached(zmin, zmax, Om0, area_deg2):
    return _V_shell_cached(round(zmin, 4), round(zmax, 4),
                           round(Om0, 5), round(area_deg2, 3))

@lru_cache(maxsize=2000)
def _V_shell_cached(z1, z2, om_r, area_r):
    cosmo = FlatLambdaCDM(H0=H0, Om0=float(om_r))
    V = (cosmo.comoving_volume(float(z2)) - cosmo.comoving_volume(float(z1))).to(u.Mpc**3).value
    return V / h**3 * (float(area_r) / 41253.0)

# ------------------------------------------------------------------
# setup()
# ------------------------------------------------------------------
def setup(options):
    Om0 = 0.318
    sigma8 = 0.8

    zmin = options.get_double(option_section, "z_min", default=0.3)
    zmax = options.get_double(option_section, "z_max", default=0.8)
    area_deg2 = options.get_double(option_section, "area_deg2", default=1000.0)
    mmin = options.get_double(option_section, "mass_min", default=1e14)
    mmax = options.get_double(option_section, "mass_max", default=1e15)

    z_mid = 0.5 * (zmin + zmax)

    mf = MassFunction(
        z=z_mid,
        sigma_8=sigma8,
        cosmo_params={"H0": H0, "Om0": Om0},
        Mmin=np.log10(mmin), Mmax=np.log10(mmax),
        dlog10m=0.1,
        hmf_model="Tinker08",
    )

    # --- Mass bins ---
    nM = 10
    M_edges_logh = np.linspace(np.log10(mmin), np.log10(mmax), nM + 1)

    # --- Compute noiseless mock counts per bin ---
    N_obs_bins = []
    V = volume_shell(zmin, zmax, Om0, area_deg2)
    for logMmin, logMmax in zip(M_edges_logh[:-1], M_edges_logh[1:]):
        mf.update(Mmin=logMmin, Mmax=logMmax)
        n_mid = np.trapz(mf.dndm, mf.m)  # h^3 Mpc^-3
        N_obs_bins.append(n_mid * V)
    N_obs_bins = np.array(N_obs_bins)

    sigma_obs = np.sqrt(N_obs_bins) #Poisson sigma = sqrt(N_obs)
    
    config = {
        "zmin": zmin,
        "zmax": zmax,
        "area_deg2": area_deg2,
        "M_edges_logh": M_edges_logh,
        "N_obs": N_obs_bins,
        "sigma_obs": sigma_obs,
        "mf": mf,
    }
    
    return config

def execute(block, config):
    omegam = block[names.cosmological_parameters, "omega_m"]
    sigma8 = block[names.cosmological_parameters, "sigma8_input"]
    
    zmin = config["zmin"]
    zmax = config["zmax"]
    area_deg2 = config["area_deg2"]
    N_obs = config["N_obs"]
    sigma_obs = config["sigma_obs"]
    mf = config["mf"]
    M_edges_logh = config["M_edges_logh"]
    
    #z_mid = 0.5 * (zmin + zmax)
    V = V_shell_cached(zmin, zmax, omegam, area_deg2)
    loglike_total = 0.0

    for i, (logMmin, logMmax) in enumerate(zip(M_edges_logh[:-1], M_edges_logh[1:])):
        mf.update(
            #z=z_mid,
            Mmin=logMmin, Mmax=logMmax,
            sigma_8=sigma8,
            cosmo_params={"H0": H0, "Om0": omegam},
        )
        n_mid = np.trapz(mf.dndm, mf.m)
        N_model = n_mid * V

        N_model_safe = max(N_model, 1e-12)
        N_obs_bin = N_obs[i]

        loglike_bin = N_obs_bin * np.log(N_model_safe) - N_model_safe - gammaln(N_obs_bin + 1)
        loglike_total += loglike_bin

    block["likelihoods", "hmf_like"] = loglike_total
    
    return 0

def cleanup(config):
    return 0
