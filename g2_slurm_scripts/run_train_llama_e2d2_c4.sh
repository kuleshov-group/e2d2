#!/bin/bash
#SBATCH -J c4_llama_e2d2              # Job name
#SBATCH -o ../watch_folder/%x_%j.out  # Output file (%j expands to jobID)
#SBATCH --get-user-env                # Retrieve the users login environment
#SBATCH --partition=kuleshov               # Request partition
#SBATCH --constraint="[a100|a6000|a5000|3090]"
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --mem=64000                   # Server memory requested (per node)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption
#SBATCH --mail-user=yzs2@cornell.edu  # Email
#SBATCH --mail-type=END               # Request status by email


# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

composer -n ${SLURM_GPUS_ON_NODE} scripts/composer_scripts/train_discrete_denoiser.py \
  run_name=c4-llama-e2d2 \
  pretrained_model_name_or_path=meta-llama/Llama-3.2-1B \
  dataset@train_dataset=c4_streaming_train \
  dataset@eval_dataset=c4_streaming_eval \
  model=ao_mdlm \
  model/backbone@model.config.backbone_config=llama_as_encoder_decoder \
  model.config.length=1024 \
  training.global_batch_size=512 \
  train_dataloader.batch_size=2 \
  eval_dataloader.batch_size=2 \
  ~composer.trainer.parallelism_config
