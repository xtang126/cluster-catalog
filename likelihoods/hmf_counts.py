# hmf_counts.py
# example config file: hmf_counts.ini

'''
Custom Poisson likelihood for halo mass function:
- Compares cluster counts to HMF prediction (Tinker08, hmf package)
- Assumes flat wCDM, single redshift/mass bin, selection function = 1
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
from astropy.cosmology import FlatwCDM

h = 0.7
H0 = h * 100  # km/s/Mpc 
Om0 = 0.318
sigma8 = 0.8

def volume_shell(cosmo, zmin, zmax, area_deg2):
    """Return h^-3 Mpc^3 so it pairs with h^3 Mpc^-3 number densities."""
    V = (cosmo.comoving_volume(zmax) - cosmo.comoving_volume(zmin)).to(u.Mpc**3).value
    V /= h**3                            # -> h^-3 Mpc^3
    f_sky = area_deg2 / 41253.0
    return V * f_sky

@lru_cache(maxsize=4096)
def _V_shell_cached_scalar(Om0, w0, H0, z1, z2, area_deg2):
    cosmo = FlatwCDM(H0=H0, Om0=Om0, w0=w0, Tcmb0=2.7255*u.K, name="vol")
    V = (cosmo.comoving_volume(z2) - cosmo.comoving_volume(z1)).to(u.Mpc**3).value
    return V / h**3 * (area_deg2 / 41253.0)


def integrate_fixed(n_z, zmin, zmax, N=4):
    x, w = leggauss(N)
    z_nodes = 0.5*(zmax - zmin)*x + 0.5*(zmax + zmin)
    weights = 0.5*(zmax - zmin)*w
    return np.sum(weights * np.array([n_z(z) for z in z_nodes]))

def setup(options):
    # ----- Fiducial cosmology for MOCK generation only -----
    #Om0 = 0.318
    #sigma8 = 0.8
    cosmo = FlatwCDM(H0=H0, Om0=Om0, w0=-1.0, Tcmb0=2.7255*u.K)
       
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
        z=0.5,       # any value; we will update per node
        cosmo_model=FlatwCDM(H0=H0, Om0=Om0, w0=-1.0, Tcmb0=2.7255*u.K, name="fid"),
        sigma_8=sigma8,         # fixed fiducial
        cosmo_params={"H0": H0, "Om0": Om0, "Ob0": 0.049, "w0": -1.0, "Tcmb0": 2.7255*u.K},
        Mmin=np.log10(Mmin_h),
        Mmax=np.log10(Mmax_h),
        dlog10m=0.1,
        hmf_model="Tinker08",
        #n=0.96,
        transfer_params={"extrapolate_with_eh": True},
    )
    
    # Fixed mass grid and bin widths (Msun/h; dM has same units)
    M = mf.m
    dM = np.gradient(M)
    
    def number_density_z(z):
        mf.update(z=z) #reuse precomputed mass grid
        n_z = np.sum(mf.dndm * dM)   # -> h^3 Mpc^-3
        return n_z
    
    n_z = lambda z: number_density_z(z)
    N_int, err = quad(n_z, float(zmin), float(zmax))  # unpack the tuple
    N_int = float(N_int)  # (optional) ensure scalar
    V = float(volume_shell(cosmo, zmin, zmax, area_deg2))
    N_obs_true = V * N_int  # noiseless expected count
    #N_obs_zint_err = err * V  # h^-3 Mpc^3 * h
    
    sigma_obs = np.sqrt(max(N_obs_true, 1.0)) #Poisson sigma = sqrt(N_obs)

    # Add Gaussian random noise to simulate measurement uncertainty ---
    # 0.05 = 5% relative scatter
    #noise_fraction = 0.05
    #noise = np.random.normal(0, noise_fraction * N_obs_true)
    #N_obs_noisy = N_obs_true + noise

    # Optionally enforce N_obs > 0
    #N_obs_noisy = max(N_obs_noisy, 1.0)
    
    print(f"Observed N: {N_obs_true}, sigma_obs: {sigma_obs}")
    #print(f"Observed N (noisy mock): {N_obs_noisy:.3f}, sigma_obs: {sigma_obs:.3f} (noise added)")
    
    noise_fraction = 0.05  # 5% relative error
    model_noise = np.random.normal(0, noise_fraction)

    config = {
        "zmin": zmin,
        "zmax": zmax,
        "area_deg2": area_deg2,
        #"Mmin": Mmin,
        #"Mmax": Mmax,
        "N_obs": N_obs_true, #N_obs_noisy,
        #"N_obs_zint_err": N_obs_zint_err,
        "sigma_obs": sigma_obs,
        "mf": mf,  # precomputed MassFunction
        "dM": dM,  # precomputed bin widths
        "model_noise": model_noise,
    }
    
    return config

def execute(block, config):
    omegam = block[names.cosmological_parameters, "omega_m"]
    sigma8 = block[names.cosmological_parameters, "sigma8_input"]
    #w = block[names.cosmological_parameters, "w"]
    
    zmin = config["zmin"]
    zmax = config["zmax"]
    area_deg2 = config["area_deg2"]
    N_obs = config["N_obs"]
    sigma_obs = config["sigma_obs"]
    mf = config["mf"]
    dM = config["dM"]
    model_noise = config["model_noise"]

    # for V_shell_cached only
    cosmo_w = FlatwCDM(H0=H0, Om0=omegam, w0=-1, Tcmb0=2.7255*u.K,name="model")
    
    def number_density_z(z):
        mf.update(
            z=z,
            sigma_8=sigma8,
            #cosmo_params={"w0": w},
        ) #reuse precomputed mass grid
        return np.sum(mf.dndm * dM)   # -> h^3 Mpc^-3
    
    n_z = lambda z: number_density_z(z)
    N_int = integrate_fixed(n_z, zmin, zmax, N=4)  # bump nodes
    V = _V_shell_cached_scalar(float(omegam), float(-1), float(H0), float(zmin), float(zmax), float(area_deg2))
    N_model = V * N_int
    
    #print(f"Model N: {N_model}")
    
    '''
    residual = (N_obs - N_model) / sigma_obs
    chi2 = residual ** 2
    loglike = -0.5 * chi2
    '''
    
    # --- Add Gaussian noise to model prediction ---
    # Keep the same scale as data uncertainty (5% or sqrt(N) style)
    N_model_noisy = N_model * (1 + model_noise)

    # Prevent negative values
    N_model_noisy = max(N_model_noisy, 1e-12)

    print(f"Model N (with noise): {N_model_noisy:.3f} (original: {N_model:.3f})")

    # Use the noisy model for likelihood
    N_model_safe = N_model_noisy
    
    # guard against zero/negative model due to numerical issues
    #eps = 1e-12
    #N_model_safe = max(N_model, eps)
    N_obs_int = int(round(N_obs))  # if your "obs" is mock, it should be an integer

    #exact Poisson loglike (constant term included; harmless)
    loglike = N_obs_int*np.log(N_model_safe) - N_model_safe - gammaln(N_obs_int+1)

    
    block["likelihoods", "hmf_counts_like"] = loglike
    
    return 0

def cleanup(config):
    return 0
