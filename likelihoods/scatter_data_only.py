# example config file: scatter_data_only.ini
# only generate scattered mock data, no model scatter applied

'''
Custom Poisson likelihood for halo mass function:
- Compares cluster counts to HMF prediction (Tinker08, hmf package)
- Assumes selection function = 1
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
from math import erf, sqrt

h = 0.7
H0 = h * 100  # km/s/Mpc

def volume_shell(zmin, zmax, Om0, area_deg2):
    """Return h^-3 Mpc^3 so it pairs with h^3 Mpc^-3 number densities."""
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
    V = (cosmo.comoving_volume(zmax) - cosmo.comoving_volume(zmin)).to(u.Mpc**3).value
    V /= h**3                            # -> h^-3 Mpc^3
    f_sky = area_deg2 / 41253.0
    return V * f_sky

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

def gaussian_cdf(x, mu, sigma):
    # scalar Gaussian CDF in log10(M)
    t = (x - mu) / (sqrt(2.0) * sigma)
    return 0.5 * (1.0 + erf(t))

def build_migration_matrix(M_edges_logh, sigma_logM):
    """
    P[i, j] = Prob(halo in true bin j is *observed* in bin i).
    Columns j normalised to 1.
    """
    M_edges_logh = np.asarray(M_edges_logh)
    bin_centres = 0.5 * (M_edges_logh[1:] + M_edges_logh[:-1])
    n_bins = len(bin_centres)

    P = np.zeros((n_bins, n_bins))

    for j in range(n_bins):
        mu = bin_centres[j]
        for i in range(n_bins):
            lo = M_edges_logh[i]
            hi = M_edges_logh[i + 1]
            cdf_hi = gaussian_cdf(hi, mu, sigma_logM)
            cdf_lo = gaussian_cdf(lo, mu, sigma_logM)
            P[i, j] = cdf_hi - cdf_lo

        col_sum = P[:, j].sum()
        if col_sum > 0.0:
            P[:, j] /= col_sum

    return P

def compute_counts_per_bin(mf, M_edges_logh, V, z_mid):
    """
    Expected counts per *true* mass bin (no scatter).
    """
    N_counts = []
    for logMmin, logMmax in zip(M_edges_logh[:-1], M_edges_logh[1:]):
        mf.update(Mmin=logMmin, Mmax=logMmax, z=z_mid)
        if len(mf.m) < 2:
            N_counts.append(0.0)
            continue
        dM = np.gradient(mf.m)
        n_mid = np.sum(mf.dndm * dM)  # h^3 Mpc^-3
        N_counts.append(n_mid * V)
    return np.array(N_counts, dtype=float)

def setup(options):
    # ----- Fiducial cosmology for MOCK generation only -----
    Om0 = 0.318
    sigma8 = 0.8
       
    # Redshift bin
    zmin = options.get_double(option_section, "z_min", default=0.3)
    zmax = options.get_double(option_section, "z_max", default=0.8)
    
    # Survey area
    area_deg2 = options.get_double(option_section, "area_deg2", default=1000.0)
    
    mmin = options.get_double(option_section, "mass_min", default=1e14)
    mmax = options.get_double(option_section, "mass_max", default=1e15)
    if not (mmax > mmin > 0):
        raise ValueError("mass_min/mass_max must be positive and mass_max > mass_min")
    
    z_mid = 0.5 * (zmin + zmax)
    
    # Build ONE MassFunction to set up mass grid
    mf = MassFunction(
        z=z_mid,              # any value; we will update per node
        sigma_8=sigma8,         # fixed fiducial
        cosmo_params={"H0": H0, "Om0": Om0},
        Mmin=np.log10(mmin),
        Mmax=np.log10(mmax),
        dlog10m=0.1,
        hmf_model="Tinker08",
        #n=0.96,
    )
    
    # Fixed mass grid and bin widths (Msun/h; dM has same units)
    #M = mf.m
    #dM = np.gradient(M)
    '''
    def number_density_z(z):
        mf.update(z=z) #reuse precomputed mass grid
        n_z = np.sum(mf.dndm * dM)   # -> h^3 Mpc^-3
        return n_z
    '''

    # define logarithmic mass bins
    # --- Mass bins (log10 M/h) ---
    nM = 10
    M_edges_logh = np.linspace(np.log10(mmin), np.log10(mmax), nM + 1)

    # volume of the shell for mock generation (fiducial Om0, sigma8)
    V = volume_shell(zmin, zmax, Om0, area_deg2)
    z_mid = 0.5 * (zmin + zmax)

    # true (no-scatter) counts per true mass bin
    N_true = compute_counts_per_bin(mf, M_edges_logh, V, z_mid)

    # mass scatter for the mock (your choice: 0.1 dex etc.)
    sigma_logM = options.get_double(option_section, "sigma_logM", default=0.1)

    # migration matrix -> mis-binned observed counts
    P = build_migration_matrix(M_edges_logh, sigma_logM)
    N_obs_bins = P @ N_true   # this is your "data" vector

    # Poisson error for chi2 / Gaussian approx
    N_obs_bins = np.clip(N_obs_bins, np.finfo(float).eps, None)
    sigma_obs = np.sqrt(N_obs_bins)

    config = {
        "zmin": zmin,
        "zmax": zmax,
        "area_deg2": area_deg2,
        "N_obs": N_obs_bins,          # <-- scattered mock data
        "sigma_obs": sigma_obs,
        "M_edges_logh": M_edges_logh, # true bin edges
        "mf": mf,
    }

    return config


def execute(block, config):
    omegam = block[names.cosmological_parameters, "omega_m"]
    sigma8 = block[names.cosmological_parameters, "sigma8_input"]

    zmin = config["zmin"]
    zmax = config["zmax"]
    area_deg2 = config["area_deg2"]
    mf = config["mf"]
    M_edges_logh = config["M_edges_logh"]
    N_obs = config["N_obs"]
    sigma_obs = config["sigma_obs"]

    # update mf cosmology
    mf.cosmo_params["Om0"] = omegam
    mf.sigma_8 = sigma8

    V = V_shell_cached(zmin, zmax, omegam, area_deg2)
    z_mid = 0.5 * (zmin + zmax)

    # model WITHOUT scatter: just mass function in each bin
    N_model = compute_counts_per_bin(mf, M_edges_logh, V, z_mid)

    # Gaussian/chi2 likelihood (since you're already using that)
    residual = (N_obs - N_model) / sigma_obs
    chi2 = np.sum(residual**2)
    loglike = -0.5 * chi2

    block["likelihoods", "hmf_bins_like"] = loglike
    return 0


def cleanup(config):
    return 0
