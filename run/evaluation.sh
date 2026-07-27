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
--model_ckpt_path "./checkpoints/Momask_RVQVAE_t2m_newdecoder_RFDecoder_0724161621_usTrue_di512_di[1, 2]_c_512_c_256_upRepeat_and_stack_and_linear_dr0.0_leFalse_leFalse_raFalse_le16_si10000_usFalse_at32_at4/checkpoints/Unet1DforFlowDecoder_best_196_0.109505.pth" \
--num_sample_steps 16 \
--seed 24 2>&1 | tee $LOGFILE