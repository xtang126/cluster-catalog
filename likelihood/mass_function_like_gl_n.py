# likelihood/mass_function_like_gl_n.py
# halo mass function likelihood using cosmosis.gaussian_likelihood for number fitting at one mass bin and one redshift bin
# haven't completely written yet, but this is the general idea

import numpy as np
import os
from cosmosis.datablock import names, option_section
from cosmosis.gaussian_likelihood import GaussianLikelihood
import scipy.interpolate
from hmf import MassFunction
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

class MassFunctionLikelihood(GaussianLikelihood):
    x_section = names.mass_function
    x_name = "m_h"
    y_section = names.mass_function
    y_name = "dndlnmh"
    like_name = "mass_function_xt"

    # --- Define cosmology  ---
    h = 0.7
    H0 = h * 100  # km/s/Mpc
    Om0 = 0.318
    Ob0 = 0.04
    sigma8 = 0.8
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0, Ob0=Ob0, sigma8=sigma8)
    zmin = 0.3
    zmax = 0.8
    
    def volume_shell(self):
        # Calculate the comoving volume of a shell between zmin and zmax
        volume = (self.cosmo.comoving_volume(self.zmax)- self.cosmo.comoving_volume(self.zmin)).to(u.Mpc**3).value
        volume /= self.h**3  # Convert to h^3 Mpc^3
        f_sky = 1000 / 41253  # if your survey covers 1000 deg²
        volume *= f_sky #* 4 * np.pi  # Adjust for the fraction of the sky
        return volume
    
    def number_density_at_z(self, z, Mmin, Mmax):
        Mmin_h = Mmin / self.h
        Mmax_h = Mmax / self.h
        
        mf = MassFunction(
            z=z,
            sigma_8=self.sigma8,         # fixed fiducial
            cosmo_params={"H0": self.H0, "Om0": self.Om0, "Ob0": self.Ob0},
            Mmin=Mmin_h,
            Mmax=Mmax_h,
            #dlog10m=0.4,
            hmf_model="Tinker08",
            #n=0.96,
            )
        
        M = mf.m
        mask = (M >= Mmin_h) & (M <= Mmax_h)
        dM = np.gradient(M)
        n_z = np.sum(mf.dndm[mask] * dM[mask])  # still h^3 Mpc^-3
        
        return n_z
    
    def build_data(self):
        #z_list_str = self.options.get_string("z_indices", default="0")
        #self.z_indices = [int(z.strip()) for z in z_list_str.split(",")]
        self.rel_sigma = self.options.get_double("relative_sigma", default=0.01)
        #self.zmin = self.options.get_double("z_min", default=0.3)
        #self.zmax = self.options.get_double("z_max", default=0.8)
        self.Mmin = self.options.get_double("mass_min", default=5e14)
        self.Mmax = self.options.get_double("mass_max", default=1e15)

        Mmin_h = self.Mmin / self.h
        Mmax_h = self.Mmax / self.h
        
        M = mf.m
        mask = (M >= Mmin_h) & (M <= Mmax_h)
        dM = np.gradient(M)
        n_z = np.sum(mf.dndm[mask] * dM[mask])  # still h^3 Mpc^-3

        return self.data_x, self.data_y

    def build_covariance(self):
        
        self.data_cov_list = [np.diag(sigma**2) for sigma in self.sigma_list]
    
        # Combine individual covariance matrices into one big block-diagonal matrix
        import scipy.linalg
        self.data_cov = scipy.linalg.block_diag(*self.data_cov_list)
        return self.data_cov

    def extract_theory_points(self, block):
        omega_m = block[names.cosmological_parameters, "omega_m"]
        sigma8 = block[names.cosmological_parameters, "sigma8_input"]
    
        model_all = []
        
        # Use caching to avoid recalculating the same MF
        if not hasattr(self, 'mf_cache'):
            self.mf_cache = {}

        for i, z in enumerate(self.z_indices):
            z_val = 0.1 * z
            key = (round(z_val, 4), round(omega_m, 6), round(sigma8, 6))

            if key in self.mf_cache:
                log_mass_model, dndlnm_model = self.mf_cache[key]
            else:
                mf = MassFunction(
                    z=z_val,
                    sigma_8=sigma8,
                    cosmo_params={"H0": 70, "Om0": omega_m, "Ob0": 0.04},
                    Mmin=13,
                    Mmax=15,
                    dlog10m=0.4,
                    hmf_model="Tinker08",
                    n=0.96,
                )
                # Interpolate dense model to match the original 5 mass points
                log_mass_model = np.log(mf.m)
                dndlnm_model = mf.dndlnm
                self.mf_cache[key] = (log_mass_model, dndlnm_model)

            log_mass_obs = np.log(self.data_x_list[i])
            interp = scipy.interpolate.interp1d(log_mass_model, dndlnm_model, kind="linear", bounds_error=False, fill_value="extrapolate")
            dndlnm_at_obs = interp(log_mass_obs)

            model_all.append(dndlnm_at_obs)

            #plt.plot(mf.m, mf.dndlnm, label = f"Model z={0.1*z:.1f}")

            # Plot 5-point data
            #plt.errorbar(self.data_x_list[i], self.data_y_list[i], yerr=self.sigma_list[i], fmt='o',  markersize=5, label=f"Data z={0.1*z:.1f}")
        '''
        plt.xlabel("Mass [$M_\\odot/h$]")
        plt.ylabel("dn/dlnM")
        plt.xscale("log")
        plt.yscale("log")
        plt.ylim(1e-8, 1e-3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        os.makedirs("mf_xt/debug_outputs", exist_ok=True)
        plt.savefig("mf_xt/debug_outputs/all_redshifts_mass_function.png")
        plt.close()
        '''
        return np.concatenate(model_all)
   
setup, execute, cleanup = MassFunctionLikelihood.build_module()
