cd /home/yerim/Code/PRISM
export PYTHONPATH=/home/yerim/Code/PRISM:$PYTHONPATH
export TZ='KST-9'

# ~/.netrc가 없는 서버라 wandb.init()이 "No API key configured"로 실패함.
# 키를 env var로 직접 넘겨줌. 키는 run/.wandb_api_key에 있음 (chmod 600, gitignore 대상 -- 절대 커밋 금지)
export WANDB_API_KEY=$(cat /home/yerim/Code/PRISM/run/.wandb_api_key)


GPUS=3

PY_ARGS=${@:1}

mkdir -p logs
LOGFILE=logs/train_rf_decoder_reproduce_$(date +%y%m%d_%H%M).log

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPUS \
  /home/yerim/.conda/envs/prism/bin/python run/train_rf_decoder_from_vqvae.py \
  --data_cfg_path ./configs/config_data.yaml \
  --model_cfg_path ./configs/config_model.yaml \
  --exp_tag "RFDecoder_Base" \
  ${PY_ARGS} 2>&1 | tee $LOGFILE
