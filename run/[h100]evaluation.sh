cd /home/yerim/Code/PRISM
export PYTHONPATH=/home/yerim/Code/PRISM:$PYTHONPATH
export TZ='KST-9'

# ~/.netrc가 없는 서버라 wandb.init()이 "No API key configured"로 실패함.
# 키를 env var로 직접 넘겨줌. 키는 run/.wandb_api_key에 있음 (chmod 600, gitignore 대상 -- 절대 커밋 금지)
export WANDB_API_KEY=$(cat /home/yerim/Code/PRISM/run/.wandb_api_key)


# GPUS=4

mkdir -p logs
LOGFILE=logs/eval_flow_decoder_reproduce_$(date +%y%m%d_%H%M).log


CUDA_VISIBLE_DEVICES=$GPUS \
  python /home/yerim/Code/PRISM/run/rf_evaluation.py \
--eval_cfg_pth ./configs/momask_trans_eval_config_t2m.yaml \
--data_cfg_path ./configs/config_data.yaml \
--model_cfg_path ./configs/config_model.yaml \
--train_data t2m \
--model_ckpt_path "./checkpoints/RFDecoder_t2m_fmTrue_bs384_ep500_PostRefinement_0727224718/checkpoints/RFDecoder_250_27.681012_fid0.01788_mpjpe0.04623.pth" \
--num_sample_steps 16 \
--seed 24 2>&1 | tee $LOGFILE