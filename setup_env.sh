#!/bin/bash

# Shell script to set environment variables when running code in this repository.
# Usage:
#     source setup_env.sh

# Activate conda env
# shellcheck source=${HOME}/.bashrc disable=SC1091
source "${CONDA_SHELL}"
if [ -z "${CONDA_PREFIX}" ]; then
    conda activate e2d2-env
 elif [[ "${CONDA_PREFIX}" != *"/e2d2-env" ]]; then
  conda deactivate
  conda activate e2d2-env
fi

# W&B / HF Setup
source "${HOME}/setup_discdiff.sh"
export HF_HOME="${PWD}/.hf_cache"
echo "HuggingFace cache set to '${HF_HOME}'."

# Add root directory to PYTHONPATH to enable module imports
export PYTHONPATH="${PWD}:${HF_HOME}/modules"

# Enforce verbose Hydra error logging
export HYDRA_FULL_ERROR=1

export NCCL_P2P_LEVEL=NVL
