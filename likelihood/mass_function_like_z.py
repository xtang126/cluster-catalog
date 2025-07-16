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
        z_list_str = self.options.get_string("z_indices", default="0")
        self.z_indices = [int(z.strip()) for z in z_list_str.split(",")]
        self.rel_sigma = self.options.get_double("relative_sigma", default=0.01)
        self.output_dir = self.options.get_string("output_dir", default="mf_xt/debug_outputs")
        os.makedirs(self.output_dir, exist_ok=True)

        m_file = self.options.get_string("mass_file")
        mf_file = self.options.get_string("dndlnmh_file")
        print(f"[mass_function_like] Loading:\n  m_h from {m_file}\n  dndlnmh from {mf_file}", flush=True)

        m_h = np.loadtxt(m_file, comments='#')
        dndlnmh = np.loadtxt(mf_file, comments='#')
        
        # Apply mass range cut
        mass_min = 1e13
        mass_max = 1e15
        mask = (m_h >= mass_min) & (m_h <= mass_max)

        self.data_x_list = []
        self.data_y_list = []
        self.sigma_list = []
        
        for z_index in self.z_indices:
            obs = dndlnmh[z_index, :]
            if m_h.shape[0] != obs.shape[0]:
                raise ValueError("Mismatch between m_h and dndlnmh row lengths.")
            m_h_cut = m_h[mask]
            obs_cut = obs[mask]
            sigma = np.maximum(self.rel_sigma * obs_cut, 1e-1)

            self.data_x_list.append(m_h_cut)
            self.data_y_list.append(obs_cut)
            self.sigma_list.append(sigma)
        
        self.data_x = np.concatenate(self.data_x_list)
        self.data_y = np.concatenate(self.data_y_list)
        
        return self.data_x, self.data_y

    def build_covariance(self):
        
        self.data_cov_list = [np.diag(sigma**2) for sigma in self.sigma_list]
    
        # Combine individual covariance matrices into one big block-diagonal matrix
        import scipy.linalg
        full_cov = scipy.linalg.block_diag(*self.data_cov_list)
    
        # Store in self.data_cov so GaussianLikelihood can use it
        self.data_cov = full_cov
        
        return full_cov

    def extract_theory_points(self, block):
        model_all = []
        
        for i, z_index in enumerate(self.z_indices):
            model_m = block[self.x_section, self.x_name]
            model_mf_all = block[self.y_section, self.y_name]
            if model_mf_all.ndim != 2 or z_index >= model_mf_all.shape[0]:
                raise ValueError("Model dndlnmh must be 2D and z_index must be valid.")
            model_mf = model_mf_all[z_index, :]

            # Interpolate model onto observed mass bins
            interp = scipy.interpolate.interp1d(model_m, model_mf, kind="linear", bounds_error=False, fill_value="extrapolate")
            model_at_obs = interp(self.data_x_list[i])
            model_all.append(model_at_obs)
       
        return np.concatenate(model_all)
   
setup, execute, cleanup = MassFunctionLikelihood.build_module()
