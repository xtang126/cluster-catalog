# generate_mock_abundance.py
#
# Build a mock abundance data file for the current simplified likelihood.
# It keeps the first 4 columns (zmin, zmax, lam_min, lam_max) from an
# existing bin-definition file, and replaces the 5th column with model counts.
#
# You can choose:
#   - noiseless mock: N_mock = N_model
#   - Poisson mock:   N_mock ~ Poisson(N_model)
#
# Output format matches the current likelihood:
#   z_min  z_max  lambda_min  lambda_max  N_mock

import math
import numpy as np
from hmf import MassFunction
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

# -------------------------------------------------
# Fiducial cosmology for mock generation
# -------------------------------------------------
h = 0.7
H0 = h * 100.0

OMEGA_M_FID = 0.3406
SIGMA8_FID = 0.8049

# -------------------------------------------------
# Fixed Y1-style richness parameters
# Keep these consistent with the likelihood
# -------------------------------------------------
LOGM_MIN_FIXED = 11.13
LOGM1_FIXED = 12.37
ALPHA_FIXED = 0.748
ETA_FIXED = -0.07
Z_PIVOT = 0.45
SIGMA_INTR_FIXED = 0.2

# -------------------------------------------------
# Numerical settings
# Keep these consistent with the likelihood test
# -------------------------------------------------
AREA_DEG2 = 1500.0
MASS_MIN = 5e13
MASS_MAX = 3e15
DLOG10M = 0.02
N_Z = 4

# -------------------------------------------------
# Mock options
# -------------------------------------------------
INPUT_BIN_FILE = "/global/homes/x/xintang/cosmosis-standard-library/mf_xt/data/desy1_abundance_data.txt"   # existing file with 5 columns
OUTPUT_MOCK_FILE = "/global/homes/x/xintang/cosmosis-standard-library/mf_xt/data/mock_y1abundance_fid018_085_v2.txt"

USE_POISSON_NOISE = True   # False = noiseless closure test first
RNG_SEED = 1234


def load_data_file(path):
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[None, :]

    if data.shape[1] != 5:
        raise ValueError("Input file must have 5 columns: z_min z_max lambda_min lambda_max N")

    return {
        "zmin": data[:, 0],
        "zmax": data[:, 1],
        "lam_min": data[:, 2],
        "lam_max": data[:, 3],
        "n_obs": data[:, 4],   # not used for mock generation, only columns 1-4 matter
    }


def dVdz_fullsky_h3(z, cosmo, h):
    """
    Full-sky differential comoving volume element [h^-3 Mpc^3 / z]
    """
    dv_dz = (cosmo.differential_comoving_volume(z) * 4.0 * np.pi * u.sr).to(u.Mpc**3).value
    return dv_dz / h**3


def gaussian_bin_prob(lam_lo, lam_hi, mean, sigma):
    """
    Probability that a Gaussian N(mean, sigma^2) falls in [lam_lo, lam_hi).
    """
    if sigma <= 0.0:
        return 1.0 if (lam_lo <= mean < lam_hi) else 0.0

    a = (lam_lo - mean) / (math.sqrt(2.0) * sigma)
    b = (lam_hi - mean) / (math.sqrt(2.0) * sigma)
    return 0.5 * (math.erf(b) - math.erf(a))


def p_lambda_true_bin_y1(
    lam_lo,
    lam_hi,
    M,
    z,
    Mmin,
    M1,
    alpha,
    eta,
    sigma_intr,
    z_pivot=0.45,
):
    """
    Same simplified observable model as the current likelihood.
    """

    lambda_cen = 1.0 if M >= Mmin else 0.0

    if M > Mmin:
        mu_sat = ((M - Mmin) / (M1 - Mmin)) ** alpha
        mu_sat *= ((1.0 + z) / (1.0 + z_pivot)) ** eta
    else:
        mu_sat = 0.0

    mu_sat = max(mu_sat, 0.0)

    if mu_sat < 1e-12:
        return 1.0 if (lam_lo <= lambda_cen < lam_hi) else 0.0

    sigma_g = sigma_intr * mu_sat
    n_max = int(max(50, math.ceil(mu_sat + 8.0 * math.sqrt(mu_sat + 1.0))))

    p_n = math.exp(-mu_sat)
    p_total = 0.0

    for n in range(n_max + 1):
        lam_true_mean = lambda_cen + n
        p_bin_given_n = gaussian_bin_prob(lam_lo, lam_hi, lam_true_mean, sigma_g)
        p_total += p_n * p_bin_given_n

        if n < n_max:
            p_n = p_n * mu_sat / (n + 1.0)

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
    n_z=4,
    dlog10m=0.02,
):
    """
    Same count model as the current likelihood.
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
            dlog10m=dlog10m,
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

        n_lambda_z = np.trapz(dndlnM * pbin, x=np.log(M))
        n_of_z.append(n_lambda_z)

    n_of_z = np.asarray(n_of_z)
    vol_integrand = np.array([dVdz_fullsky_h3(z, cosmo, h) for z in z_grid]) * n_of_z
    N = f_sky * np.trapz(vol_integrand, x=z_grid)

    return max(N, 1e-30)


def main():
    rng = np.random.default_rng(RNG_SEED)

    data = load_data_file(INPUT_BIN_FILE)

    cosmo = FlatLambdaCDM(H0=H0, Om0=OMEGA_M_FID)

    mf = MassFunction(
        z=0.2,
        sigma_8=SIGMA8_FID,
        cosmo_params={"H0": H0, "Om0": OMEGA_M_FID},
        Mmin=np.log10(MASS_MIN),
        Mmax=np.log10(MASS_MAX),
        dlog10m=DLOG10M,
        hmf_model="Tinker08",
    )

    Mmin = 10.0 ** LOGM_MIN_FIXED
    M1 = 10.0 ** LOGM1_FIXED
    alpha = ALPHA_FIXED
    eta = ETA_FIXED
    sigma_intr = SIGMA_INTR_FIXED

    n_model = np.zeros_like(data["zmin"], dtype=float)

    for i in range(len(n_model)):
        n_model[i] = expected_count_one_bin(
            mf=mf,
            cosmo=cosmo,
            area_deg2=AREA_DEG2,
            z1=data["zmin"][i],
            z2=data["zmax"][i],
            lam1=data["lam_min"][i],
            lam2=data["lam_max"][i],
            Mmin=Mmin,
            M1=M1,
            alpha=alpha,
            eta=eta,
            sigma_intr=sigma_intr,
            mass_min=MASS_MIN,
            mass_max=MASS_MAX,
            n_z=N_Z,
            dlog10m=DLOG10M,
        )

    if USE_POISSON_NOISE:
        n_mock = rng.poisson(n_model)
    else:
        n_mock = n_model

    out = np.column_stack([
        data["zmin"],
        data["zmax"],
        data["lam_min"],
        data["lam_max"],
        n_mock,
    ])

    header = (
        "Mock abundance data generated from the same simplified model\n"
        f"fiducial_omega_m = {OMEGA_M_FID}\n"
        f"fiducial_sigma8 = {SIGMA8_FID}\n"
        f"area_deg2 = {AREA_DEG2}\n"
        f"mass_min = {MASS_MIN}\n"
        f"mass_max = {MASS_MAX}\n"
        f"dlog10m = {DLOG10M}\n"
        f"n_z = {N_Z}\n"
        f"use_poisson_noise = {USE_POISSON_NOISE}\n"
        "Columns: z_min z_max lambda_min lambda_max N_mock"
    )

    np.savetxt(OUTPUT_MOCK_FILE, out, fmt="%.8g", header=header)
    print(f"Saved mock data to: {OUTPUT_MOCK_FILE}")
    print("n_mock =", n_mock)


if __name__ == "__main__":
    main()