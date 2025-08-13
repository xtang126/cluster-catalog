# mass_function_like_m_n.py
# Halo mass function likelihood: one mass bin, one redshift bin
# Speed-ups:
#   - Build MassFunction once in setup -> reuse mass grid across all steps
#   - Reuse dM (bin widths)
#   - Precompute Gauss–Legendre nodes/weights for z integration
#   - Cache comoving volume
#   - Avoid re-allocations inside execute

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
H0 = h * 100  # km/s/Mpc

def volume_shell(zmin, zmax, Om0, area_deg2):
    """Return h^-3 Mpc^3 so it pairs with h^3 Mpc^-3 number densities."""
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
    V = (cosmo.comoving_volume(zmax) - cosmo.comoving_volume(zmin)).to(u.Mpc**3).value
    V /= h**3                            # -> h^-3 Mpc^3
    f_sky = area_deg2 / 41253.0
    return V * f_sky

'''
# ---- cached number density at z ----
def n_of_z_cached(z, sigma8, Om0, Mmin_h, Mmax_h):
    return _n_of_z_cached(
        round(z, 4),               # z quantization
        round(sigma8, 5),
        round(Om0, 5),
        round(Mmin_h, 3),
        round(Mmax_h, 3),
    )

@lru_cache(maxsize=20000)
def _n_of_z_cached(z_r, s8_r, om_r, mminh_r, mmaxh_r):
    mf = MassFunction(
        z=float(z_r),
        sigma_8=float(s8_r),
        cosmo_params={"H0": H0, "Om0": float(om_r)},
        Mmin=np.log10(float(mminh_r)),
        Mmax=np.log10(float(mmaxh_r)),
        hmf_model="Tinker08",
        # dlog10m=0.1,   # optional speed knob
    )
    M = mf.m
    return float(np.sum(mf.dndm * np.gradient(M)))   # h^3 Mpc^-3
'''

# ---- cached comoving volume shell ----
def V_shell_cached(zmin, zmax, Om0, area_deg2):
    return _V_shell_cached(
        round(zmin, 4), round(zmax, 4),
        round(Om0, 5),  round(area_deg2, 3)
    )

@lru_cache(maxsize=2000)
def _V_shell_cached(z1, z2, om_r, area_r):
    cosmo = FlatLambdaCDM(H0=H0, Om0=float(om_r))
    V = (cosmo.comoving_volume(float(z2)) - cosmo.comoving_volume(float(z1))).to(u.Mpc**3).value
    return V / h**3 * (float(area_r) / 41253.0)

def integrate_fixed(n_z, zmin, zmax, N=4):
    x, w = leggauss(N)
    z_nodes = 0.5*(zmax - zmin)*x + 0.5*(zmax + zmin)
    weights = 0.5*(zmax - zmin)*w
    return np.sum(weights * np.array([n_z(z) for z in z_nodes]))

def setup(options):
    # ----- Fiducial cosmology for MOCK generation only -----
    Om0 = 0.318
    sigma8 = 0.8
       
    # Redshift bin
    zmin = options.get_double(option_section, "z_min", default=0.3)
    zmax = options.get_double(option_section, "z_max", default=0.8)
    
    # Survey area
    area_deg2 = options.get_double(option_section, "area_deg2", default=1000.0)
    
    Mmin = options.get_double(option_section, "mass_min", default=5e14)
    Mmax = options.get_double(option_section, "mass_max", default=1e15)
    if not (Mmax > Mmin > 0):
        raise ValueError("mass_min/mass_max must be positive and mass_max > mass_min")
    
    Mmin_h = Mmin / h
    Mmax_h = Mmax / h
    
    # Build ONE MassFunction to set up mass grid
    mf = MassFunction(
        z=0.5,              # any value; we will update per node
        sigma_8=sigma8,         # fixed fiducial
        cosmo_params={"H0": H0, "Om0": Om0},
        Mmin=np.log10(Mmin_h),
        Mmax=np.log10(Mmax_h),
        dlog10m=0.1,
        hmf_model="Tinker08",
        #n=0.96,
    )
    
    # Fixed mass grid and bin widths (Msun/h; dM has same units)
    M = mf.m
    dM = np.gradient(M)
    
    def number_density_z(z):
        mf.update(z=z) #reuse precomputed mass grid
        n_z = np.sum(mf.dndm * dM)   # -> h^3 Mpc^-3
        return n_z
    
    n_z = lambda z: number_density_z(z)
    N_int, err = quad(n_z, zmin, zmax)
    V = volume_shell(zmin, zmax, Om0, area_deg2)
    N_obs = V * N_int  # deimensionless
    #N_obs_zint_err = err * V  # h^-3 Mpc^3 * h
    
    sigma_obs = np.sqrt(max(N_obs, 1.0)) #Poisson sigma = sqrt(N_obs)
    
    print(f"Observed N: {N_obs}, sigma_obs: {sigma_obs}")

    config = {
        "zmin": zmin,
        "zmax": zmax,
        "area_deg2": area_deg2,
        #"Mmin": Mmin,
        #"Mmax": Mmax,
        "N_obs": N_obs,
        #"N_obs_zint_err": N_obs_zint_err,
        "sigma_obs": sigma_obs,
        "mf": mf,  # precomputed MassFunction
        "dM": dM  # precomputed bin widths
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
    dM = config["dM"]
    
    def number_density_z(z):
        mf.update(
            z=z,
            sigma_8=sigma8,
            cosmo_params={"H0": H0, "Om0": omegam}
        ) #reuse precomputed mass grid
        n_z = np.sum(mf.dndm * dM)   # -> h^3 Mpc^-3
        return n_z
    
    n_z = lambda z: number_density_z(z)
    V= V_shell_cached(zmin, zmax, omegam, area_deg2)
    #N_int, err = quad(n_z, zmin, zmax)
    N_int = integrate_fixed(n_z, zmin, zmax, 4)
    N_model = V * N_int  # deimensionless
    
    print(f"Model N: {N_model}")
    
    '''
    residual = (N_obs - N_model) / sigma_obs
    chi2 = residual ** 2
    loglike = -0.5 * chi2
    '''
    
    # guard against zero/negative model due to numerical issues
    eps = 1e-12
    N_model_safe = max(N_model, eps)
    N_obs_int = int(round(N_obs))  # if your "obs" is mock, it should be an integer

    #exact Poisson loglike (constant term included; harmless)
    loglike = N_obs_int*np.log(N_model_safe) - N_model_safe - gammaln(N_obs_int+1)

    
    block["likelihoods", "mass_function_like_m_n_like"] = loglike
    
    return 0

def cleanup(config):
    return 0
