# ![icon](assets/icon.png) E2D2: Encoder-Decoder Diffusion Language Models for Efficient Training and Inference

[![deploy](https://img.shields.io/badge/Paper_📃-green)](https://github.com/kuleshov-group/e2d2)
[![deploy](https://img.shields.io/badge/Blog_📝%20%20-8A2BE2)](https://m-arriola.com/e2d2)
[![deploy](https://img.shields.io/badge/HuggingFace_🤗%20-E2D2%20-orange)](https://huggingface.co/collections/kuleshov-group/e2d2)


This repository contains code and scripts for reproducing experimental results from our
work.

## 0. Getting started

### Setup environment

Install conda:
```bash
# For conda: https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
bash miniconda.sh -b -p /opt/conda
```

Setup a conda environment and install dependencies using:

```bash
conda env create -f requirements.yaml
```

Activate the environment:

```bash
conda activate e2d2-env
```

We also include a [`setup_env.sh`](./setup_env.sh) script that can be used to set up the
environment on a new machine.
Run the script using:
```bash
source setup_env.sh
```

You can also include this snippet in shell / slurm scripts to set up the environment on
a compute node.

In this script, we set up WandB and HuggingFace tokens by sourcing a script which is
expected to be in the `/home/<YOUR_USER_NAME>/` directory.
Copy the contents below into a shell script `/home/<YOUR_USER_NAME>/setup_discdiff.sh`
and replace the placeholder tokens with your own:
```shell
# W&B / HF Setup
export WANDB__SERVICE_WAIT=600
export _WANDB_STARTUP_DEBUG="true"
export WANDB_ENTITY="<WANDB_ENTITY>"
export WANDB_API_KEY="<WANDB_API_KEY>"
echo "Logging into W&B as '${WANDB_ENTITY}'."

# HF Setup
export HUGGINGFACE_TOKEN="<HF_TOKEN>"
huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
```
- WandB token can be found [here](https://wandb.ai/authorize).
- HuggingFace token can be setup [here](https://huggingface.co/settings/tokens).

### Contributing to the repo
We will try to use GitHub issues to track bugs, features, and todos.
To contribute to the repo, please create a new issue and assign it to yourself.
Then [create a new branch from the issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/creating-a-branch-for-an-issue)
and open a pull request.


We use [pre-commit](https://pre-commit.com/) to run linters and formatters on the code.
To install the pre-commit hooks, run:

```bash
pre-commit install
```
On every `git commit`,
the pre-commit hooks will run automatically and report any issues / automatic fixes that
were applied.

## 1. Code Organization
1. [`bash_scripts`](bash_scripts): These shells scripts can be used to reproduce the
experiments from our work.
See [below](#2-reproducing-experiments).
2. [`configs`](configs): We utilize hydra config files to organize experiments.
   1. [`config.yaml`](configs/config.yaml) This config is the entry point for launching
   training experiments.
   2. [`eval_config.yaml`](configs/eval_config.yaml) This config is the entry point for
   evaluations.
3. [`scripts`](scripts): The main training and evaluation scripts
   1. [`scripts/composer_scripts/train_discrete_denoiser.py`](scripts/composer_scripts/train_discrete_denoiser.py):
   This script is the main training entry point.
   2. [`scripts/evals`](scripts/eval): These scripts run the evaluation for the
   translation, summarization, and math reasoning datasets, as well as any likelihood
   evaluation.
4. [`src`](src):
   1. [`src/denoiser`](src/denoiser): During training, denoisers take in "noisy" inputs
   and predict clean signals.
   At inference, starting from a purely noisy signal, through iterative denoising, these
   classes produce samples that resemble data.
      1. `AR`: We can view autoregressive models within this paradigm.
      Noise is applied by masking tokens one at a time from right-to-left.
      Denoising is done one token at a time, left-to-right.
      2. `Diffusion`: We implement masked diffusion models:
         - `MDLM`: Standard masked diffusion.
         - `BD3LM`: Block diffusion models.
         - `E2D2`: Our encoder-decoder implementation.
   2. [`src/backbone`](src/backbone): These are the underlying neural networks the take
   in noisy inputs and produce logits.
   Each denoiser is parameterized by a backbone.
   The denoiser can optionally, post-process the logit outputs of the backbone to
   produce log-probs over the clean sequence.


## 2. Reproducing Experiments
The shell scripts provided in [`bash_scripts`](bash_scripts) can be used to reproduce
the training and evaluations from our work.
- For training, the files follow a convention where the dataset and denoiser class are
specified.
For example, to train the fine-tuning E2D2 model on the GSM8K dataset, use the following
shell script: [`run_train_e2d2_gsm8k.sh`](bash_scripts/run_train_e2d2_gsm8k.sh).
- Once models have been trained, the provided evaluation scripts can be used to reproduce
tables and figures from our work.
For example, to evaluate models trained on the WMT translation dataset, use the
following shell script: [`run_seq2seq_eval_wmt.sh`](bash_scripts/run_seq2seq_eval_wmt.sh).
In that file, and similar ones for other evaluations, specify the path to the saved
checkpoints, and uncomment the relevant section for a given denoiser class.
We also provide scripts that will produce the generation throughput numbers we report.
These files contain a `_tput` at the end of the script name.

## 3. HuggingFace Integration
We release the following models on HuggingFace:
- E2D2 for text summarization (trained from scratch):
[`kuleshov-group/e2d2-cnndm`](https://huggingface.co/kuleshov-group/e2d2-cnndm)
- E2D2 for machine translation (trained from scratch):
[`kuleshov-group/e2d2-wmt`](https://huggingface.co/kuleshov-group/e2d2-wmt)
- E2D2 for mathematical reasoning (fine-tuned from Qwen3):
[`kuleshov-group/e2d2-gsm8k-finetune-Qwen3-2B`](https://huggingface.co/kuleshov-group/e2d2-gsm8k-finetune-Qwen3-2B)
- E2D2 trained on OpenWebText (trained from scratch):
[`kuleshov-group/e2d2-owt`](https://huggingface.co/kuleshov-group/e2d2-owt)

To use these models, follow the snippet below:
```python
from transformers import AutoModelForMaskedLM

# model_config_overrides = {}  # Use this to optionally override config parameters
model = AutoModelForMaskedLM.from_pretrained(
    "kuleshov-group/e2d2-cnndm",  # Use one of the repos from above
    trust_remote_code=True,
    # **model_config_overrides,
)
```

These models can also be used in the evaluation scripts by setting
`pretrained_model_name_or_path=` to one of the options above.

## Citation
```
TODO: Add bibtex
```
