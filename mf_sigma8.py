import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt

# Set range of sigma_8 values
sigma8_values = np.linspace(0.5, 0.9, 9)  # [0.5, 0.55, ..., 0.9]

base_ini_path = "mf_xt/values_mf.ini"
template_path = "mf_xt/values_mf_template.ini"
pipeline_file = "mf.ini"
output_dir_base = "mf_xt/results_sigma8"

# For plotting
mass_curves = []
dndm_curves = []

# Read the template content
with open(template_path, "r") as f:
    template_content = f.read()


for sigma8 in sigma8_values:
    print(f"Running CosmoSIS for sigma8 = {sigma8:.2f}")
    
    # Set unique output directory
    output_dir = f"{output_dir_base}/sigma8_{sigma8:.2f}"
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(output_dir):
        print(f"Output directory {output_dir} already exists. Skipping...")
        continue
    
    # Replace sigma8 in template
    updated_content = template_content.replace("sigma8_input = 0.9", f"sigma8_input = {sigma8:.2f}")

    # Save current values file
    current_values_path = base_ini_path
    with open(current_values_path, "w") as f:
        f.write(updated_content)

    # Update and write temporary mf.ini
    mf_ini_temp = "mf_xt/temp_mf.ini"
    updated_mf_ini = f"""
[runtime]
sampler = test

[test]
save_dir={output_dir}
fatal_errors=T

[pipeline]
modules = consistency camb rescale mf_tinker
values = {current_values_path}

[consistency]
file = utility/consistency/consistency_interface.py

[rescale]
file = utility/sample_sigma8/sigma8_rescale.py

[camb]
file = boltzmann/camb/camb_interface.py
mode=all
lmax=2500
feedback=0
zmin = 0.0
zmax = 1.0
nz = 2

[mf_tinker]
file = mass_function/mf_tinker/tinker_mf_module.so
redshift_zero = 0
feedback=0
"""
    with open(mf_ini_temp, "w") as f:
        f.write(updated_mf_ini)

    # Run CosmoSIS
    subprocess.run(["cosmosis", mf_ini_temp])

# Read results and store for plotting
for sigma8 in sigma8_values:
    output_dir = f"{output_dir_base}/sigma8_{sigma8:.2f}"
    mf_path = output_dir + "/mass_function/m_h.txt"
    dndlnm_path = output_dir + "/mass_function/dndlnmh.txt"
    if os.path.exists(mf_path) and os.path.exists(dndlnm_path):
        mass = np.loadtxt(mf_path, comments='#')
        dndlnmh = np.loadtxt(dndlnm_path, comments='#')
        dndm0 = dndlnmh[0, :]
        dndm1 = dndlnmh[1, :]
        mass_curves.append((sigma8, mass, dndm0, dndm1))
    else:
        print(f"Missing output for sigma8 = {sigma8:.2f}")

# Plot mass function comparison
#plt.figure(figsize=(8, 6))
for (sigma8, mass, dndm0, dndm1) in mass_curves:
     plt.loglog(mass[600:], dndm0[600:], label=rf'$\sigma_8$={sigma8:.2f}')

# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # Use default matplotlib colors
# for i, (sigma8, mass, dndm0, dndm1) in enumerate(mass_curves):
#     color = colors[i % len(colors)]  # Cycle through default colors
#     plt.loglog(mass[900:], dndm0[900:], label=rf'$\sigma_8$={sigma8:.2f}', color=color)
#     plt.loglog(mass[900:], dndm1[900:], linestyle='--', color=color)

    
plt.legend()
plt.xlabel(r'$M_h\, [h^{-1}\, M_\odot]$')
plt.ylabel(r'$\frac{d n}{d \ln M_h}\, [h^3\, \mathrm{Mpc}^{-3}]$')
plt.title('Mass Function with z = 0')
#plt.title('Mass Function with z = 0 (solid curves) and z = 1 (dashed curves)')
#plt.grid(True)
plt.xlim(1e13, 1e16)
plt.ylim(1e-15, 1e-2)
#plt.tight_layout()
plt.savefig("mf_xt/mf_s.png", dpi=300)
#plt.savefig("mf_xt/mf_sigma8_z0.png", bbox_inches='tight', dpi=300)
#plt.show()
