# likelihood/mass_function_like.py
# using covariance matrix for likelihood calculation

import numpy as np
import os
from cosmosis.datablock import names, option_section
from cosmosis.gaussian_likelihood import GaussianLikelihood
import scipy.interpolate


class MassFunctionLikelihood(GaussianLikelihood):
    x_section = names.mass_function
    x_name = "m_h"
    y_section = names.mass_function
    y_name = "dndlnmh"
    like_name = "mass_function_xt"

    def build_data(self):
        self.z_index = self.options.get_int("z_index", default=0)
        self.rel_sigma = self.options.get_double("relative_sigma", default=0.01)
        self.use_poisson = self.options.get_bool("use_poisson", default=False)
        self.output_dir = self.options.get_string("output_dir", default="mf_xt/debug_outputs")
        os.makedirs(self.output_dir, exist_ok=True)

        m_file = self.options.get_string("mass_file")
        mf_file = self.options.get_string("dndlnmh_file")
        print(f"[mass_function_like] Loading:\n  m_h from {m_file}\n  dndlnmh from {mf_file}", flush=True)

        m_h = np.loadtxt(m_file, comments='#')
        dndlnmh = np.loadtxt(mf_file, comments='#')

        if dndlnmh.ndim != 2 or self.z_index >= dndlnmh.shape[0]:
            raise ValueError("HMF data must be 2D and z_index must be valid.")

        obs = dndlnmh[self.z_index, :]
        if m_h.shape[0] != obs.shape[0]:
            raise ValueError("Mismatch between m_h and dndlnmh row lengths.")

        # Apply mass range cut
        mass_min = 1e13
        mass_max = 1e15
        mask = (m_h >= mass_min) & (m_h <= mass_max)

        m_h = m_h[mask]
        obs = obs[mask]
        sigma = np.maximum(self.rel_sigma * obs, 1e-1)

        self.data_x = m_h
        self.data_y = obs
        self.obs_sigma = sigma
        #self.data_cov = np.diag(sigma**2)

        print(f"[mass_function_like] z_index = {self.z_index}, bins = {len(m_h)}, using {'Poisson' if self.use_poisson else 'Gaussian'} likelihood", flush=True)
        
        return m_h, obs

    def build_covariance(self):
        
        return np.diag(self.obs_sigma**2)

    def extract_theory_points(self, block):
        model_m = block[self.x_section, self.x_name]
        model_mf_all = block[self.y_section, self.y_name]
        model_mf = model_mf_all[self.z_index, :]

        # Interpolate model onto observed mass bins
        interp = scipy.interpolate.interp1d(model_m, model_mf, kind="linear", bounds_error=False, fill_value="extrapolate")
        model_at_obs = interp(self.data_x)
        '''
        #checking mismarch
        print("Observed dndlnmh:", self.data_y[600:611], flush=True)
        print("Model dndlnmh   :", model_at_obs[600:611], flush=True)
        print("Max model:", np.max(model_at_obs))
        print("Max data :", np.max(self.data_y))
        print("Data mass range :", self.data_x[0], self.data_x[-1])
        print("Model mass range:", model_m[0], model_m[-1])
        print("Min model:", np.min(model_at_obs))
        print("Min data :", np.min(self.data_y))
        print("Where max model happens:", np.argmax(model_at_obs))
        print("Where max data happens :", np.argmax(self.data_y))
        
        print("[mass_function_like] In extract_theory_points", flush=True)
        print(f"omega_m: {block[names.cosmological_parameters, 'omega_m']}", flush=True)
        print(f"sigma8: {block[names.cosmological_parameters, 'sigma8_input']}", flush=True)
        
        import matplotlib.pyplot as plt

        plt.plot(self.data_x, self.data_y, label="Observed")
        plt.plot(self.data_x, model_at_obs, label="Model")
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Mass")
        plt.ylabel("dn/dlnM")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "model_vs_obs.png"))


        np.savetxt(os.path.join(self.output_dir, "model_dndlnmh.txt"), model_at_obs)
        np.savetxt(os.path.join(self.output_dir, "observed_dndlnmh.txt"), self.data_y)
        np.savetxt(os.path.join(self.output_dir, "mass_bins.txt"), self.data_x)
       '''
        return model_at_obs
   
setup, execute, cleanup = MassFunctionLikelihood.build_module()
