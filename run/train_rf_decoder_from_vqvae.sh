#!/usr/bin/bash
#SBATCH -J RFDecoder_PostRefinement
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v7
#SBATCH -t 4-0
#SBATCH -o /nas2/data/dpfla3573/code/PRISM/logs/slurm-%A_RFDecoder_moscale_fmTrue_postRefinement_bs384_noise1.0.out
cd /nas2/data/dpfla3573/code/PRISM
export PYTHONPATH=/nas2/data/dpfla3573/code/PRISM:$PYTHONPATH

# SBATCH -o만으로는 실제 시작 시각을 파일명에 못 넣어서(제출 시점 %-치환만 지원), 여기서 실제 시작 시각을 붙여 다시 리다이렉트
START_TIME=$(date +%Y%m%d_%H%M%S)
exec > "logs/slurm-${SLURM_JOB_ID}_RFDecoder_moscale_fmTrue_postRefinement_x0False_bs384_noise1.0_${START_TIME}.out" 2>&1

# ~/.netrc lives on the (node-local) /home disk, invisible to compute nodes -- wandb.init()
# fails there with "No API key configured". Feed the key in directly via env var instead.
# Key lives in run/.wandb_api_key (chmod 600, gitignored -- never commit this file).
export WANDB_API_KEY=$(cat /nas2/data/dpfla3573/code/DisCoRD/run/.wandb_api_key)

/nas2/data/dpfla3573/anaconda3/envs/prism/bin/python run/train_rf_decoder_from_vqvae.py \
--data_cfg_path ./configs/config_data.yaml \
--model_cfg_path ./configs/config_model.yaml \
--wandb

