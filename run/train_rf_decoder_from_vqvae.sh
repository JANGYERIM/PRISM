#!/usr/bin/bash
#SBATCH -J RFDecoder_BASE
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v3
#SBATCH -t 4-0
#SBATCH -o /nas2/data/dpfla3573/code/PRISM/logs/slurm-%A_RFDecoder_base_moscale_fmTrue.out
cd /nas2/data/dpfla3573/code/PRISM
export PYTHONPATH=/nas2/data/dpfla3573/code/PRISM:$PYTHONPATH

# ~/.netrc lives on the (node-local) /home disk, invisible to compute nodes -- wandb.init()
# fails there with "No API key configured". Feed the key in directly via env var instead.
# Key lives in run/.wandb_api_key (chmod 600, gitignored -- never commit this file).
export WANDB_API_KEY=$(cat /nas2/data/dpfla3573/code/DisCoRD/run/.wandb_api_key)

/nas2/data/dpfla3573/anaconda3/envs/prism/bin/python run/train_rf_decoder_from_vqvae.py \
--data_cfg_path ./configs/config_data.yaml \
--model_cfg_path ./configs/config_model.yaml \
--wandb

