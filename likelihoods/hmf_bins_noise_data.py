# example config file: hmf_bins_noise_data.ini

'''
Custom Poisson likelihood for halo mass function:
- Compares cluster counts to HMF prediction (Tinker08, hmf package)
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

    # Optionally add Gaussian scatter to mass bin edges ---
    add_noise = options.get_bool(option_section, "add_noise", default=True)
    noise_sigma = options.get_double(option_section, "noise_sigma", default=0.1)

    if add_noise:
        # perturb edges in log-space
        M_edges_logh_noisy = M_edges_logh + np.random.normal(0, noise_sigma, len(M_edges_logh))
        M_edges_logh_noisy = np.sort(M_edges_logh_noisy)  # keep ascending order
    else:
        M_edges_logh_noisy = M_edges_logh.copy()

    # Compute mock number counts per mass bin ---
    V = volume_shell(zmin, zmax, Om0, area_deg2)
    N_obs_bins = []
    for logMmin, logMmax in zip(M_edges_logh_noisy[:-1], M_edges_logh_noisy[1:]):
        mf.update(Mmin=logMmin, Mmax=logMmax)
        n_mid = np.trapz(mf.dndm, mf.m)  # h^3 Mpc^-3
        N_obs_bins.append(n_mid * V)
    N_obs_bins = np.array(N_obs_bins)

    N_obs_bins = np.clip(N_obs_bins, np.finfo(float).eps, None)
    sigma_obs = np.sqrt(N_obs_bins)


    
    #n_z = lambda z: number_density_z(z)
    #N_int, err = quad(n_z, zmin, zmax)
    #V = volume_shell(zmin, zmax, Om0, area_deg2)
    #N_obs = V * N_int  # deimensionless
    #N_obs_zint_err = err * V  # h^-3 Mpc^3 * h
    
    #sigma_obs = np.sqrt(max(N_obs, 1.0)) #Poisson sigma = sqrt(N_obs)
    
    # Add Gaussian random noise to simulate measurement uncertainty ---
    # 0.05 = 5% relative scatter
    #noise_fraction = 0.05
    #noise = np.random.normal(0, noise_fraction * N_obs)
    #N_obs_noisy = N_obs + noise

    # Optionally enforce N_obs > 0
    #N_obs_noisy = max(N_obs_noisy, 1.0)
    
    #print(f"Observed N: {N_obs}, sigma_obs: {sigma_obs}")
    #print(f"Observed N (noisy mock): {N_obs_noisy:.3f}, sigma_obs: {sigma_obs:.3f} (noise added)")
    
    #print(f"Observed N: {N_obs}, sigma_obs: {sigma_obs}")

    config = {
        "zmin": zmin,
        "zmax": zmax,
        "area_deg2": area_deg2,
        #"Mmin": Mmin,
        #"Mmax": Mmax,
        "N_obs": N_obs_bins,
        #"N_obs_zint_err": N_obs_zint_err,
        "M_edges_logh": M_edges_logh,              # true edges
        "M_edges_logh_noisy": M_edges_logh_noisy,  # scattered edges used for mock
        "sigma_obs": sigma_obs,
        "mf": mf,  # precomputed MassFunction
        #"dM": dM  # precomputed bin widths
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
    M_edges_true = config["M_edges_logh"]
    M_edges_obs  = config["M_edges_logh_noisy"]
    
    #z_mid = 0.5 * (zmin + zmax)
    V = V_shell_cached(zmin, zmax, omegam, area_deg2)
    loglike_total = 0.0

    for i, (logMmin, logMmax) in enumerate(zip(M_edges_true[:-1], M_edges_true[1:])):
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

    block["likelihoods", "hmf_bins_like"] = loglike_total


    '''
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
    
    
    residual = (N_obs - N_model) / sigma_obs
    chi2 = residual ** 2
    loglike = -0.5 * chi2
    
    
    # guard against zero/negative model due to numerical issues
    eps = 1e-12
    N_model_safe = max(N_model, eps)
    N_obs_int = int(round(N_obs))  # if your "obs" is mock, it should be an integer

    #exact Poisson loglike (constant term included; harmless)
    loglike = N_obs_int*np.log(N_model_safe) - N_model_safe - gammaln(N_obs_int+1)
    
    
    block["likelihoods", "hmf_like"] = loglike_total
    '''

    return 0

def cleanup(config):
    return 0
