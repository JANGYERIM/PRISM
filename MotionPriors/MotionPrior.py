import os
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from einops import rearrange

from .models import rf_decoder
from .models.rf_decoder.rectified_flow import RectifiedFlowDecoder
from .models.MoScale.hrvqvae import HRVQVAE


def lengths_to_mask(lengths, max_len):
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask  # (b, len)


def load_MotionPrior(model_ckpt_path, model_cfg):
    if model_cfg.model.name == "MoScale_HRVQVAE":
        opt = model_cfg.model
        pose_dim = 263 if opt.dataset_name == 't2m' else 251
        model = HRVQVAE(
            args=opt,
            input_width=pose_dim,
            down_t=opt.down_t,
            stride_t=opt.stride_t,
            width=opt.width,
            depth=opt.depth,
            dilation_growth_rate=opt.dilation_growth_rate,
            activation=opt.vq_act,
            use_attn=opt.use_attn,
            norm=opt.vq_norm
        )
        ckpt = torch.load(model_ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['vq_model'])
        return model
    elif model_cfg.model.name == "RFDecoder":
        denoiser = rf_decoder.get_flow_backbone(model_cfg)
        model = RectifiedFlowDecoder(model=denoiser)
    else:
        raise ValueError(f"Model {model_cfg.model.name} not supported.")

    if model_ckpt_path is not None:
        model.load_state_dict(torch.load(model_ckpt_path, map_location="cpu"))
        print(f"Loaded {model_cfg.model.name} model from {model_ckpt_path}")
    else:
        print(f"Created {model_cfg.model.name} model from scratch")
    return model


class MotionPriorWrapper(nn.Module):

    def __init__(self, model_cfg, model_ckpt, device):
        super(MotionPriorWrapper, self).__init__()
        self.device = device
        self.model_cfg = model_cfg
        self.model_ckpt = model_ckpt

        if "vqvae_weight_path" in model_cfg.model.keys():
            vae_ckpt = self.model_cfg.model.vqvae_weight_path
            vae_root = os.path.dirname(os.path.dirname(vae_ckpt))
            vae_cfg_path = os.path.join(vae_root, "configs/config_model.yaml")
            self.vae_cfg = OmegaConf.load(vae_cfg_path)

        try:
            if "quant_factor" in model_cfg.model.keys():
                self.quant_factor = model_cfg.model.quant_factor
            else:
                self.quant_factor = self.vae_cfg.model.quant_factor
        except:
            print("WARNING: quant_factor not found in model_cfg or vae_cfg, is this intended?")

        if self.model_ckpt is not None:
            self.model = self._create_model()
        else:
            self.model = None
            print("No model checkpoint provided.")

        self.num_sample_steps = 16
        self.deterministic = False
        self.vqvae = None

    def set_vqvae(self, frozen=True):
        if self.model.__class__.__name__ in ("Flow", "RectifiedFlowDecoder"):
            self.vqvae = MotionPriorWrapper(self.vae_cfg, self.model_cfg.model.vqvae_weight_path, self.device)
            if frozen:
                for param in self.vqvae.parameters():
                    param.requires_grad = False
            self.vqvae.eval()
        else:
            raise ValueError(f"set_vqvae not supported for {self.model.__class__.__name__}")

    def _create_model(self):
        model = load_MotionPrior(self.model_ckpt, self.model_cfg)
        return model.to(self.device)

    def get_z(self, motion):
        net = self.model
        if net.__class__.__name__ == 'HRVQVAE':
            m_lens = torch.full((motion.shape[0],), motion.shape[1], device=motion.device)
            code_idx, all_codes, accumulated_fhat = net.encode(
                motion, m_lens=m_lens, perturb_rate=[0.0, 0.0], codebook=None, train=False
            )
            return accumulated_fhat
        else:
            raise ValueError(f"Unsupported model type: {net.__class__.__name__}")

    def get_z_and_recon(self, motion, m_length=None):
        # Flow 모델용 - RVQVAE에서 z와 reconstruction을 함께 반환
        if self.model.__class__.__name__ == 'RVQVAE':
            x_out, commit_loss, perplexity, z = self.model(motion, return_z=True)
            return z, x_out
        else:
            raise ValueError(f"Unsupported model type: {self.model.__class__.__name__}")
    
    def sample_from_z(self, z, m_length=None, text_embedding=None, noise=None):
        """z: (B,T,C) - 이미 ocdebook에서 dequantize된 값.
        vqvae encode/decode를 거치지 않고 flow decoder sampling만 수행(eval을 위함)"""
        net = self.model
        assert net.__class__.__name__ == "RectifiedFlowDecoder"
        bs, length, dim = z.shape
        padding_mask = ~lengths_to_mask(m_length, 196).to(z.device) if m_length is not None else None
        pred_pose_eval = net.sample(
            z, batch_size=bs, steps=self.num_sample_steps,
            padding_mask=padding_mask, text_embedding=text_embedding,
            data_shape=(196, self.model_cfg.model.output_dim),
            noise=noise,
        )
        return pred_pose_eval

    def forward(self, motion, m_length=None, cfg_scale=1.0, text_embedding=None):
        """
        motion: (B, seq_len, 263)
        """
        net = self.model

        if net.__class__.__name__ == 'Flow':
            assert self.vqvae is not None
            assert self.vqvae.__class__.__name__ == "MotionPriorWrapper"
            if net.net.z_condition:
                z, noisy_motion = self.vqvae.get_z_and_recon(motion)
                bs, length, dim = noisy_motion.shape
                others = (noisy_motion, z)
                z = z.repeat_interleave(4, dim=-1)
                z = rearrange(z, 'b c n -> b n c')
                padding_mask = ~lengths_to_mask(m_length, 196).to(noisy_motion.device) if m_length is not None else None
                pred_pose_eval = net.sample(
                    noisy_motion, z=z, deterministic=self.deterministic,
                    num_sample_steps=self.num_sample_steps, bsz=bs, length=length,
                    padding_mask=padding_mask, text_embedding=text_embedding,
                )
            else:
                noisy_motion, others = self.vqvae(motion)
                others = (noisy_motion,) + others
                bs, length, dim = noisy_motion.shape
                padding_mask = ~lengths_to_mask(m_length, 196).to(noisy_motion.device) if m_length is not None else None
                pred_pose_eval = net.sample(
                    noisy_motion, deterministic=self.deterministic,
                    num_sample_steps=self.num_sample_steps, bsz=bs, length=length,
                    padding_mask=padding_mask, text_embedding=text_embedding,
                )

        elif net.__class__.__name__ == 'RectifiedFlowDecoder':
            assert self.vqvae is not None
            assert self.vqvae.__class__.__name__ == "MotionPriorWrapper"

            if motion.shape[1] != 196:
                old_shape = motion.shape
                motion = torch.nn.functional.pad(motion, (0, 0, 0, 196 - motion.shape[1]), mode='constant', value=0)
                print(f"Padding motion from {old_shape} to {motion.shape}")

            y = self.vqvae.get_z(motion)  # (B, dim, N/4)
            if hasattr(self.vqvae.model, "decoder"):
                vqvae_out = self.vqvae.model.decoder(y)
            else:
                vqvae_out = self.vqvae.model.vqvae.decoder(y)
            y = y.permute(0, 2, 1)  # (B, N/4, dim)
            
            x0 = vqvae_out + self.model_cfg.model.noise_std * torch.randn_like(vqvae_out)
            pred_pose_eval = self.sample_from_z(y, m_length=m_length, text_embedding=text_embedding, noise=x0)
            others = (vqvae_out,)

        else:
            raise ValueError(f"Unsupported model type: {net.__class__.__name__}")

        return pred_pose_eval, others
