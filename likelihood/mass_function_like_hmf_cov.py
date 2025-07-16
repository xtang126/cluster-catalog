# likelihood/mass_function_like_hmf_cov.py
# using covariance matrix for likelihood calculation

import numpy as np
import os
from cosmosis.datablock import names, option_section
from cosmosis.gaussian_likelihood import GaussianLikelihood
import scipy.interpolate
from hmf import MassFunction
import matplotlib.pyplot as plt


class MassFunctionLikelihood(GaussianLikelihood):
    x_section = names.mass_function
    x_name = "m_h"
    y_section = names.mass_function
    y_name = "dndlnmh"
    like_name = "mass_function_xt"

    def build_data(self):
        z_list_str = self.options.get_string("z_indices", default="0")
        self.z_indices = [int(z.strip()) for z in z_list_str.split(",")]
        self.rel_sigma = self.options.get_double("relative_sigma", default=0.01)


        self.data_x_list = []
        self.data_y_list = []
        self.sigma_list = []
        
        for z_index in self.z_indices:
            z = 0.1 * z_index
            mf = MassFunction(
                z=z,
                sigma_8=0.8,         # fixed fiducial
                cosmo_params={"H0": 70, "Om0": 0.3, "Ob0": 0.04},
                Mmin=13,
                Mmax=15,
                dlog10m=0.4,
                hmf_model="Tinker08",
                n=0.96,
            )

            mass = mf.m
            dndlnm = mf.dndlnm
            sigma = self.rel_sigma * dndlnm

            self.data_x_list.append(mass)
            self.data_y_list.append(dndlnm)
            self.sigma_list.append(sigma)

        self.data_x = np.concatenate(self.data_x_list)
        self.data_y = np.concatenate(self.data_y_list)

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
