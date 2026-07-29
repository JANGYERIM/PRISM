#!/usr/bin/bash
#SBATCH -J RFDecoder_EVAL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_grad
#SBATCH -w ariel-v3
#SBATCH -t 1-0
#SBATCH -o /nas2/data/dpfla3573/code/PRISM/logs/slurm-%A_EVAL_RFDecoder_t2m_DitTrue_x0False_fmTrue_bs384_ep300_noise1.0_0728233454.out
cd /nas2/data/dpfla3573/code/PRISM
export PYTHONPATH=/nas2/data/dpfla3573/code/PRISM:$PYTHONPATH

/nas2/data/dpfla3573/anaconda3/envs/prism/bin/python run/rf_evaluation.py \
--eval_cfg_pth ./configs/moscale_eval_config_t2m.yaml \
--data_cfg_path ./configs/config_data.yaml \
--model_cfg_path ./checkpoints/RFDecoder_t2m_0724221600_RFDecoder_Base/configs/config_model.yaml \
--train_data t2m \
--checkpoints_dir ./checkpoints/models \
--model_ckpt_path "./checkpoints/RFDecoder_t2m_DitTrue_x0False_fmTrue_bs384_ep300_noise1.0_0728233454/checkpoints/RFDecoder_270_0.260025_fid0.11282_mpjpe0.06575.pth" \
--num_sample_steps 16 \
--seed 24 

