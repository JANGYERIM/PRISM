import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from omegaconf import OmegaConf
from MotionPriors.models.MoScale.hrvqvae import HRVQVAE

vae_cfg = OmegaConf.load('checkpoints/MoScale/configs/config_model.yaml')
opt = vae_cfg.model
model = HRVQVAE(
    args=opt,
    input_width=263,
    down_t=opt.down_t,
    stride_t=opt.stride_t,
    width=opt.width,
    depth=opt.depth,
    dilation_growth_rate=opt.dilation_growth_rate,
    activation=opt.vq_act,
    use_attn=opt.use_attn,
    norm=opt.vq_norm,
)
ckpt = torch.load('checkpoints/MoScale/checkpoints/net_best_fid.tar', map_location='cpu')

model_keys = set(model.state_dict().keys())
ckpt_keys = set(ckpt['vq_model'].keys())
print('model params:', len(model_keys), '/ ckpt params:', len(ckpt_keys))
print('missing in ckpt:', model_keys - ckpt_keys)
print('unexpected in ckpt:', ckpt_keys - model_keys)

msd = model.state_dict()
mismatch = [(k, msd[k].shape, ckpt['vq_model'][k].shape) for k in model_keys & ckpt_keys if msd[k].shape != ckpt['vq_model'][k].shape]
print('shape mismatches:', mismatch)

result = model.load_state_dict(ckpt['vq_model'], strict=True)
print('load_state_dict result:', result)
print('checkpoint epoch (ep):', ckpt.get('ep'))
