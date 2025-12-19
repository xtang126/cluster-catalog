#!/bin/bash
#SBATCH --job-name=cosmosis_noise
#SBATCH --output=mf_xt/output/log_noise_data_v4.out
#SBATCH --error=mf_xt/output/log_noise_data_v4.err
#SBATCH --time=09:00:00          # 9 hours
#SBATCH --nodes=1
#SBATCH --ntasks=4               # Matches your `srun -n 4`
#SBATCH --cpus-per-task=1
#SBATCH --constraint=cpu
#SBATCH --qos=regular            # or 'debug' if short test (<30 mins)
#SBATCH --account=des

# One thread per rank (avoid oversubscription)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Load your modules/environment if needed
# module load openmpi
source /dvs_ro/cfs/projectdirs/des/zuntz/cosmosis-global/setup-cosmosis3

# Go to your CosmoSIS directory
cd ~/cosmosis-standard-library

# Run the pipeline
start=$(date +%s)

srun -n 4 cosmosis --mpi mf_xt/config/scatter_data_only.ini

end=$(date +%s)
runtime=$((end - start))

echo "CosmoSIS run time: $runtime seconds" >> mf_xt/output/log_noise_data_v4.out