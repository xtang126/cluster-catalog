# desy1_abundance_like.py
# Simplified DES Y1 abundance-only likelihood
# Fixed Y1 mass-richness relation parameters from the DES Y1 cosmology paper (http://arxiv.org/abs/2002.11124)
# Vary only omega_m and sigma8
#Include DES-Y1-style intrinsic richness scatter:
# P(lambda_true|M) = Poisson + Gaussian convolution


import math
import numpy as np
from functools import lru_cache

from cosmosis.datablock import names, option_section
from hmf import MassFunction
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from scipy.special import gammaln


# -------------------------------------------------
# Fixed Y1 posterior values from DES Y1
# -------------------------------------------------
LOGM_MIN_FIXED  = 11.13
LOGM1_FIXED     = 12.37
ALPHA_FIXED     = 0.748
ETA_FIXED       = -0.07
Z_PIVOT         = 0.45
SIGMA_INTR_FIXED = 0.2

# -------------------------------------------------
# Helpers
# -------------------------------------------------

def load_data_file(path):
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[None, :]

    if data.shape[1] != 5:
        raise ValueError("Data file must have 5 columns: z_min z_max lambda_min lambda_max N_obs")

    return {
        "zmin":    data[:, 0],
        "zmax":    data[:, 1],
        "lam_min": data[:, 2],
        "lam_max": data[:, 3],
        "n_obs":   data[:, 4],
    }

'''
def mean_lambda_sat_y1(M, z, Mmin, M1, alpha, eta, z_pivot=0.45):
    """
    DES Y1 mean satellite richness:
      <lambda_sat|M,z> = ((M-Mmin)/(M1-Mmin))^alpha * ((1+z)/(1+z_pivot))^eta
    for M > Mmin, else 0.
    """
    M = np.asarray(M, dtype=float)
    mu = np.zeros_like(M)

    denom = M1 - Mmin
    if denom <= 0.0:
        raise ValueError("Need M1 > Mmin")

    mask = M > Mmin
    mu[mask] = ((M[mask] - Mmin) / denom) ** alpha
    mu[mask] *= ((1.0 + z) / (1.0 + z_pivot)) ** eta

    return mu
'''

def dVdz_fullsky(z, cosmo):
    """
    Full-sky differential comoving volume element [Mpc^3 / z]
    """
    return (cosmo.differential_comoving_volume(z) * 4.0 * np.pi * u.sr).to(u.Mpc**3).value


@lru_cache(maxsize=128)
def cached_cosmo(H0_rounded, Om0_rounded):
    return FlatLambdaCDM(H0=float(H0_rounded), Om0=float(Om0_rounded))


def gaussian_bin_prob(lam_lo, lam_hi, mean, sigma):
    """
    Probability that a Gaussian N(mean, sigma^2) falls in [lam_lo, lam_hi).
    """
    if sigma <= 0.0:
        return 1.0 if (lam_lo <= mean < lam_hi) else 0.0

    a = (lam_lo - mean) / (math.sqrt(2.0) * sigma)
    b = (lam_hi - mean) / (math.sqrt(2.0) * sigma)
    return 0.5 * (math.erf(b) - math.erf(a))


def p_lambda_true_bin_y1(lam_lo, lam_hi, M, z, Mmin, M1, alpha, eta, sigma_intr, z_pivot=0.45):
    """
    Compute P(lambda_true in [lam_lo, lam_hi) | M, z)
    using the DES-Y1 intrinsic model:

      lambda_true = lambda_cen + lambda_sat

    where lambda_cen is deterministic:
      lambda_cen = 1 if M >= Mmin else 0

    and P(lambda_true|M) is a convolution of:
      - Poisson for the stochastic satellite part
      - Gaussian scatter with width sigma_intr * <lambda_sat|M,z>

    We implement this by summing over Poisson satellite realizations n
    and placing a Gaussian around lambda_true = lambda_cen + n.
    """

    # Central contribution
    lambda_cen = 1.0 if M >= Mmin else 0.0

    # Mean satellite richness
    if M > Mmin:
        mu_sat = ((M - Mmin) / (M1 - Mmin)) ** alpha
        mu_sat *= ((1.0 + z) / (1.0 + z_pivot)) ** eta
    else:
        mu_sat = 0.0

    mu_sat = max(mu_sat, 0.0)

    # If no satellites expected, richness is deterministic at lambda_cen
    if mu_sat < 1e-12:
        return 1.0 if (lam_lo <= lambda_cen < lam_hi) else 0.0

    # DES-Y1-style Gaussian width
    sigma_g = sigma_intr * mu_sat

    # Safe truncation of the Poisson sum
    n_max = int(max(50, math.ceil(mu_sat + 8.0 * math.sqrt(mu_sat + 1.0))))

    # Recursive Poisson probabilities for numerical stability
    p_n = math.exp(-mu_sat)   # P(n=0)
    p_total = 0.0

    for n in range(n_max + 1):
        # lambda_true = lambda_cen + n, Gaussian-broadened
        lam_true_mean = lambda_cen + n

        p_bin_given_n = gaussian_bin_prob(lam_lo, lam_hi, lam_true_mean, sigma_g)
        p_total += p_n * p_bin_given_n

        if n < n_max:
            p_n = p_n * mu_sat / (n + 1.0)

    # Clamp tiny numerical drift
    return max(min(p_total, 1.0), 0.0)


def expected_count_one_bin(
    mf,
    cosmo,
    area_deg2,
    z1,
    z2,
    lam1,
    lam2,
    Mmin,
    M1,
    alpha,
    eta,
    sigma_intr,
    mass_min,
    mass_max,
    n_z=12,
    dlog10m=0.02,
):
    """
    Compute expected count in one (z, lambda) bin:
      N = f_sky * ∫ dz (dV_full/dz) ∫ dlnM (dn/dlnM) P(lambda_true in bin | M,z)
    """

    f_sky = area_deg2 / 41253.0

    z_grid = np.linspace(z1, z2, n_z)
    if len(z_grid) < 2:
        z_grid = np.array([z1, z2])

    n_of_z = []

    for z in z_grid:
        mf.update(
            z=float(z),
            Mmin=np.log10(mass_min),
            Mmax=np.log10(mass_max),
            dlog10m=dlog10m
        )

        M = mf.m
        dndm = mf.dndm
        dndlnM = dndm * M

        pbin = np.array([
            p_lambda_true_bin_y1(
                lam_lo=lam1,
                lam_hi=lam2,
                M=m,
                z=z,
                Mmin=Mmin,
                M1=M1,
                alpha=alpha,
                eta=eta,
                sigma_intr=sigma_intr,
            )
            for m in M
        ])

        integrand = dndlnM * pbin
        n_lambda_z = np.trapz(integrand, x=np.log(M))
        n_of_z.append(n_lambda_z)

    n_of_z = np.asarray(n_of_z)

    vol_integrand = np.array([dVdz_fullsky(z, cosmo) for z in z_grid]) * n_of_z
    N = f_sky * np.trapz(vol_integrand, x=z_grid)

    return max(N, 1e-30)


# -------------------------------------------------
# CosmoSIS setup
# -------------------------------------------------

def setup(options):
    data_file = options.get_string(option_section, "data_file")
    area_deg2 = options.get_double(option_section, "area_deg2", default=1500.0)

    mass_min = options.get_double(option_section, "mass_min", default=1e13)
    mass_max = options.get_double(option_section, "mass_max", default=5e15)
    dlog10m = options.get_double(option_section, "dlog10m", default=0.02)
    n_z = options.get_int(option_section, "n_z", default=12)

    h = options.get_double(option_section, "h0", default=0.7)
    H0 = 100.0 * h

    data = load_data_file(data_file)

    mf = MassFunction(
        z=0.2,
        sigma_8=0.8,
        cosmo_params={"H0": H0, "Om0": 0.3},
        Mmin=np.log10(mass_min),
        Mmax=np.log10(mass_max),
        dlog10m=dlog10m,
        hmf_model="Tinker08",
    )

    config = {
        "data": data,
        "area_deg2": area_deg2,
        "mass_min": mass_min,
        "mass_max": mass_max,
        "dlog10m": dlog10m,
        "n_z": n_z,
        "h": h,
        "H0": H0,
        "mf": mf,
    }
    return config


# -------------------------------------------------
# CosmoSIS execute
# -------------------------------------------------

def execute(block, config):
    data = config["data"]
    area_deg2 = config["area_deg2"]
    mass_min = config["mass_min"]
    mass_max = config["mass_max"]
    dlog10m = config["dlog10m"]
    n_z = config["n_z"]
    H0 = config["H0"]
    mf = config["mf"]

    omega_m = block[names.cosmological_parameters, "omega_m"]
    sigma8 = block[names.cosmological_parameters, "sigma8_input"]

    # Fixed Y1 posterior values
    Mmin = 10.0 ** LOGM_MIN_FIXED
    M1   = 10.0 ** LOGM1_FIXED
    alpha = ALPHA_FIXED
    eta   = ETA_FIXED
    sigma_intr = SIGMA_INTR_FIXED

    if M1 <= Mmin:
        block["likelihoods", "hmf_like"] = -1.0e30
        return 0

    mf.update(
        sigma_8=float(sigma8),
        cosmo_params={"H0": H0, "Om0": float(omega_m)},
        Mmin=np.log10(mass_min),
        Mmax=np.log10(mass_max),
        dlog10m=dlog10m,
    )

    cosmo = cached_cosmo(round(H0, 6), round(float(omega_m), 6))

    n_obs = data["n_obs"]
    n_model = np.zeros_like(n_obs, dtype=float)

    for i in range(len(n_obs)):
        n_model[i] = expected_count_one_bin(
            mf=mf,
            cosmo=cosmo,
            area_deg2=area_deg2,
            z1=data["zmin"][i],
            z2=data["zmax"][i],
            lam1=data["lam_min"][i],
            lam2=data["lam_max"][i],
            Mmin=Mmin,
            M1=M1,
            alpha=alpha,
            eta=eta,
            sigma_intr=sigma_intr,
            mass_min=mass_min,
            mass_max=mass_max,
            n_z=n_z,
            dlog10m=dlog10m,
        )

    n_model_safe = np.clip(n_model, 1e-20, None)

    loglike = np.sum(
        n_obs * np.log(n_model_safe) - n_model_safe - gammaln(n_obs + 1.0)
    )

    block["likelihoods", "hmf_like"] = loglike
    return 0


def cleanup(config):
    return 0