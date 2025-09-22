#!/bin/bash

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

# Important variables (fix during hyperparam sweep)
BLOCK_SIZE=8
EVAL_BLOCK_SIZE=8
HIDDEN_SIZE=256
INTERMEDIATE_SIZE=768 #$(( 4 * HIDDEN_SIZE ))
N_ENCODER_LAYERS=20
ENCODER_TOP_LAYERS=false
N_DECODER_LAYERS=8
DECODER_TOP_LAYERS=false
REINIT_ENCODER=true
REINIT_DECODER=true
TIE_WEIGHTS=false
ENCODER_CAUSAL_MASK=false

# Hyperparameters
LR=3e-4
WARMUP_DURATION="1000ba"
BATCH_SIZE=128
MAX_DURATION="500000ba"

PRETRAINED_MODEL_NAME_OR_PATH=Qwen/Qwen3-0.6B-Base

TAG="e2d2"
if [ "${ENCODER_TOP_LAYERS}" == "true" ]; then
  ENC_LAYERS="TOPenc${N_ENCODER_LAYERS}"
else
  ENC_LAYERS="enc${N_ENCODER_LAYERS}"
fi
if [ "${DECODER_TOP_LAYERS}" == "true" ]; then
  DEC_LAYERS="TOPdec${N_DECODER_LAYERS}"
else
  DEC_LAYERS="dec${N_DECODER_LAYERS}"
fi
RUN_NAME=cnn_block${BLOCK_SIZE}_lr${LR}_bsz${BATCH_SIZE}_warm${WARMUP_DURATION}_${ENC_LAYERS}_${DEC_LAYERS}_hidden${HIDDEN_SIZE}_inter${INTERMEDIATE_SIZE}_${TAG}
if [ "${TIE_WEIGHTS}" == "true" ]; then
  RUN_NAME="${RUN_NAME}_tie-weights"
fi
if [ "${ENCODER_CAUSAL_MASK}" == "true" ]; then
  RUN_NAME="${RUN_NAME}_encoder-causal-mask"
fi
if [ "${REINIT_ENCODER}" == "true" ]; then
  RUN_NAME="${RUN_NAME}_reinit-encoder"
fi
if [ "${REINIT_DECODER}" == "true" ]; then
  RUN_NAME="${RUN_NAME}_reinit-decoder"
fi
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -E 's/.*(A[0-9]+|H100|A6000).*/\1/' | head -n 1)
if [[ "$GPU_TYPE" == "A100" || "$GPU_TYPE" == "H100" ]]; then
    MICRO_BATCH_SIZE=8
elif [[ "$GPU_TYPE" == "A6000" ]]; then
    MICRO_BATCH_SIZE=4
else
    MICRO_BATCH_SIZE=2
fi
#MICRO_BATCH_SIZE=16 #$(( BATCH_SIZE / NUM_VISIBLE_DEVICES ))
NUM_WORKERS=0

composer -n ${NUM_VISIBLE_DEVICES} scripts/composer_scripts/train_discrete_denoiser.py \
  run_name=${RUN_NAME} \
  pretrained_model_name_or_path=${PRETRAINED_MODEL_NAME_OR_PATH} \
  dataset@train_dataset=cnn_dailymail_train \
  dataset@eval_dataset=cnn_dailymail_eval \
  composer.optimizer.lr=${LR} \
  composer.trainer.eval_interval="5000ba" \
  composer.trainer.max_duration=${MAX_DURATION} \
  composer.trainer.save_num_checkpoints_to_keep=1 \
  composer/lr_scheduler=constant_with_warmup \
  composer.lr_scheduler.t_warmup=${WARMUP_DURATION} \
  model=e2d2 \
  model.config.attn_backend="sdpa" \
  training.compile_backbone=true \
  model.config.length=1024 \
  model/backbone@model.config.backbone_config=llm_as_encoder_decoder \
  model.config.backbone_config.use_encoder_causal_mask=${ENCODER_CAUSAL_MASK} \
  model.config.backbone_config.num_encoder_layers=${N_ENCODER_LAYERS} \
  model.config.backbone_config.num_decoder_layers=${N_DECODER_LAYERS} \
  model.config.backbone_config.tie_encoder_decoder_weights=${TIE_WEIGHTS} \
  model.config.backbone_config.reinit_decoder=${REINIT_DECODER} \
  model.config.backbone_config.reinit_encoder=${REINIT_ENCODER} \
  model.config.backbone_config.keep_top_decoder_layers=${DECODER_TOP_LAYERS} \
  model.config.backbone_config.keep_top_encoder_layers=${ENCODER_TOP_LAYERS} \
  +model.config.backbone_config.hidden_size=${HIDDEN_SIZE} \
  +model.config.backbone_config.intermediate_size=${INTERMEDIATE_SIZE} \
  training.global_batch_size=${BATCH_SIZE} \
  training.grad_accum=$(( BATCH_SIZE / NUM_VISIBLE_DEVICES / MICRO_BATCH_SIZE )) \
  ~composer.trainer.compile_config \
  ~composer.trainer.parallelism_config \
  block_size=${BLOCK_SIZE} \
  eval_block_size=${EVAL_BLOCK_SIZE} \
  training.antithetic_sampling=false \
  hydra.run.dir=${RUN_DIR}/${RUN_NAME} \
  composer.trainer.save_interval="1000ba" \
  composer.loggers.name=${RUN_NAME} \
  train_dataloader.num_workers=${NUM_WORKERS} \
  composer.callbacks.hf_compatible_checkpointing.disable_hf=true
