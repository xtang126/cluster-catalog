#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------------------
# run_loop_mass.sh
#
# Script to loop over different mass bins and run CosmoSIS.
# IMPORTANT:
#   - This file is located in: ~/cosmosis-standard-library/mf_xt/pipeline_loop_bins/
#   - You MUST run it from the CosmoSIS Standard Library root:
#         cd ~/cosmosis-standard-library
#         ./mf_xt/pipeline_loop_bins/run_loop_mass.sh
#
#   - Outputs (chains + logs) will be stored in:
#         mf_xt/pipeline_loop_bins/output/mass/
# -------------------------------------------------------------------

# --- config ---
BASE_INI="mf_xt/fit_mass_function_hmf_m_n.ini"     # relative to cosmosis-standard-library
OUTDIR="mf_xt/pipeline_loop_bins/output/mass"
mkdir -p "$OUTDIR"

# Define mass bins (Msun, no h) from HIGH → LOW
M_MINS=(1e13 5e13 1e14)
M_MAXS=(5e13 1e14 5e14)

# Sanity check
if [ ${#M_MINS[@]} -ne ${#M_MAXS[@]} ]; then
  echo "M_MINS and M_MAXS lengths differ"; exit 1
fi

# --- loop ---
for i in "${!M_MINS[@]}"; do
  idx=$((i+1))
  mmin="${M_MINS[$i]}"
  mmax="${M_MAXS[$i]}"
  outfile="${OUTDIR}/mf_m${idx}.txt"
  logfile="${OUTDIR}/log_m${idx}.log"

  echo ">>> Bin ${idx}: mass_min=${mmin}, mass_max=${mmax} -> ${outfile}"
  echo "    Log: ${logfile}"

  # run in background with nohup+time
  nohup /usr/bin/time -v cosmosis "$BASE_INI" \
    --override mass_function_like/mass_min="${mmin}" \
    --override mass_function_like/mass_max="${mmax}" \
    --override output/filename="${outfile}" \
    > "${logfile}" 2>&1
done

echo "All jobs submitted in background."
echo "Check logs in: $OUTDIR"
