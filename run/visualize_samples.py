"""
testset에서 랜덤 N개를 뽑아 gt / 생성된 모션(moscale+discord) / moscale base만 디코딩한 모션을
한 mp4 안에 나란히 애니메이션으로 저장하는 스크립트. caption과 length는 captions.txt에 기록.

동일한 discrete code(mids)를 한 번만 생성해서 두 디코더(discord, base)에 각각 흘려보내므로
디코더 차이만 순수하게 비교할 수 있음. 생성 시 길이는 evaluation과 동일하게 GT length(m_length)를
그대로 넘겨 강제한다 (free-form length 예측이 아님).

python run/visualize_samples.py \
    --eval_cfg_pth ./configs/moscale_eval_config_t2m.yaml \
    --model_cfg_path ./checkpoints/RFDecoder_t2m_.../configs/config_model.yaml \
    --model_ckpt_path ./checkpoints/RFDecoder_t2m_.../checkpoints/xxx.pth \
    --num_samples 30
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # PRISM repo root
sys.path.append('/nas2/data/dpfla3573/code/MoScale')

from os.path import join as pjoin
import argparse
import numpy as np
import torch

from config.load_config import load_config

from MotionPriors import MotionPrior
from evaluation.get_opt import get_opt
from datasets.t2m_dataset import Text2MotionDatasetEval, collate_fn
from utils.word_vectorizer import WordVectorizer
from utils.utils import seed_everything
from utils.visualize import plot_3d_motion_compare
from configs import config_utils

from run.rf_evaluation import load_vq_model, load_trans_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_cfg_pth', type=str, default='./configs/moscale_eval_config_t2m.yaml')
    parser.add_argument('--data_cfg_path', type=str, default='./configs/config_data.yaml')
    parser.add_argument('--model_cfg_path', type=str, default="./checkpoints/RFDecoder_t2m_DitTrue_x0False_fmTrue_bs384_ep300_noise1.0_0728233454/configs/config_model.yaml")
    parser.add_argument('--model_ckpt_path', type=str, default="./checkpoints/RFDecoder_t2m_DitTrue_x0False_fmTrue_bs384_ep300_noise1.0_0728233454/checkpoints/RFDecoder_270_0.260025_fid0.11282_mpjpe0.06575.pth", help='RFDecoder(discord) checkpoint')
    parser.add_argument('--train_data', type=str, default='t2m', choices=['t2m', 'kit'])
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/models')
    parser.add_argument('--num_sample_steps', type=int, default=16)
    parser.add_argument('--num_samples', type=int, default=30)
    parser.add_argument('--out_dir', type=str, default='./samples_vis_DitNoise1.0_x0False')
    parser.add_argument('--seed', type=int, default=24)
    parser.add_argument('--noise_std', type=float, default=None,
                         help='x0 = decoder(z) + noise_std * randn 에서 쓸 noise_std. '
                              '안 주면 checkpoint config에 저장된 값(보통 0.1) 사용')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    eval_cfg = config_utils.get_yaml_config(args.eval_cfg_pth)
    data_cfg = config_utils.get_yaml_config(args.data_cfg_path)
    model_cfg = config_utils.get_yaml_config(args.model_cfg_path)
    data_cfg = data_cfg.t2m if args.train_data == 't2m' else data_cfg.kit

    moscale_cfg = load_config(pjoin(args.checkpoints_dir, args.train_data, 'moscale', eval_cfg.moscale_name, 'train_moscale.yaml'))
    vq_cfg = load_config(pjoin(args.checkpoints_dir, args.train_data, 'hrvqvae', eval_cfg.vq_name, 'train_hrvqvae.yaml'))
    moscale_cfg.vq = vq_cfg.quantizer

    vq_model = load_vq_model(vq_cfg, moscale_cfg, args.checkpoints_dir, args.train_data, device)
    trans = load_trans_model(moscale_cfg, 'net_best_fid.tar', eval_cfg.moscale_name, args.checkpoints_dir, args.train_data, device)

    meta_dir = os.path.dirname(os.path.dirname(args.model_ckpt_path)) + '/meta'
    mean = np.load(meta_dir + '/mean.npy')
    std = np.load(meta_dir + '/std.npy')

    rf_model = MotionPrior.MotionPriorWrapper(model_cfg, args.model_ckpt_path, device)
    rf_model.eval()
    rf_model.num_sample_steps = args.num_sample_steps
    rf_model.set_vqvae()

    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    test_dataset = Text2MotionDatasetEval(data_cfg, mean, std, w_vectorizer, split='test')
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.num_samples, drop_last=True,
        collate_fn=collate_fn, shuffle=True, pin_memory=True,
    )

    word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = next(iter(test_dataloader))
    m_length = m_length.to(device)
    pose = pose.to(device).float()

    text_condition = model_cfg.model.get('text_condition', False)

    with torch.no_grad():
        mids = trans.generate(
            clip_text, m_length // 4, eval_cfg.cond_scale,
            temperature=eval_cfg.temperature, vq_model=vq_model,
            sample_time=list(eval_cfg.sample_time), top_p_thres=eval_cfg.top_p_thres,
        )

        # moscale base: 기존 VQVAE 디코더
        base_motion = vq_model.forward_decoder(mids, m_length.clone())

        # moscale + discord: 동일 mids를 RFDecoder(rectified flow)로 디코딩
        z = vq_model.quantizer.get_codes_from_indices(mids)
        m_len = m_length if rf_model.model_cfg.train.get('full_motion', False) else None
        vqvae_out = vq_model.decoder(z.permute(0, 2, 1))
        noise_std = args.noise_std if args.noise_std is not None else rf_model.model_cfg.model.noise_std
        x0 = vqvae_out + noise_std * torch.randn_like(vqvae_out)

        if text_condition:
            text_embedding = rf_model.model.net.encode_text(clip_text)
            t5_embedding, t5_padding_mask = rf_model.model.net.encode_text_t5(clip_text)
        else:
            text_embedding, t5_embedding, t5_padding_mask = None, None, None

        discord_motion = rf_model.sample_from_z(
            z, m_length=m_len, noise=x0,
            text_embedding=text_embedding, t5_embedding=t5_embedding, t5_padding_mask=t5_padding_mask,
        )
        print(f'[info] using noise_std={noise_std} (checkpoint default={rf_model.model_cfg.model.noise_std})')

        # x0 대비 flow가 실제로 얼마나 움직였는지 정량 확인 (정규화된 스케일 기준)
        diff_x0 = (discord_motion - x0).abs()
        diff_base = (discord_motion - base_motion).abs()
        x0_scale = x0.abs().mean().item()
        print(
            f"[diagnostic] |discord - x0| mean={diff_x0.mean().item():.6f} "
            f"({diff_x0.mean().item() / (x0_scale + 1e-8):.2%} of |x0| mean={x0_scale:.6f}), "
            f"max={diff_x0.max().item():.6f} | |discord - base| mean={diff_base.mean().item():.6f}"
        )

        # HumanML3D 263-dim pose 벡터 구간별 breakdown (root / ric position / 6D rotation / local velocity / foot contact)
        pose_segments = {
            'root(0:4)': slice(0, 4),
            'ric_pos(4:67)': slice(4, 67),
            'rot6d(67:193)': slice(67, 193),
            'local_vel(193:259)': slice(193, 259),
            'foot_contact(259:263)': slice(259, 263),
        }
        print("[diagnostic] per-segment |discord - x0| breakdown:")
        for name, sl in pose_segments.items():
            seg_diff = diff_x0[..., sl]
            seg_scale = x0[..., sl].abs().mean().item()
            print(
                f"    {name}: mean={seg_diff.mean().item():.6f} "
                f"({seg_diff.mean().item() / (seg_scale + 1e-8):.2%} of |x0| mean={seg_scale:.6f}), "
                f"max={seg_diff.max().item():.6f}"
            )

    caption_lines = ['idx\ttoken_len(gt)\tframe_len(gt)\tdiff_vs_x0(mean)\tdiff_vs_x0(max)\tcaption']
    for i in range(args.num_samples):
        length = int(m_length[i].item())
        token_len = length // 4  # trans.generate에 강제로 넘긴 길이 (evaluation과 동일한 방식)
        caption = clip_text[i]

        gt_np = pose[i, :length].cpu().numpy() * std + mean
        x0_np = x0[i, :length].cpu().numpy() * std + mean
        discord_np = discord_motion[i, :length].cpu().numpy() * std + mean
        base_np = base_motion[i, :length].cpu().numpy() * std + mean

        sample_diff = (discord_motion[i, :length] - x0[i, :length]).abs()
        diff_mean = sample_diff.mean().item()
        diff_max = sample_diff.max().item()

        plot_3d_motion_compare(
            pjoin(args.out_dir, f'{i}.mp4'),
            [gt_np, x0_np, discord_np, base_np],
            sub_titles=['GT', 'x0 (decoder(z)+noise)', 'Generated (moscale+discord)', 'MoScale base'],
            title=f'[{i}] {caption}',
            figsize=(4, 4), fps=20,
        )
        caption_lines.append(f'{i}\t{token_len}\t{length}\t{diff_mean:.6f}\t{diff_max:.6f}\t{caption}')
        print(f'[{i}] token_len(gt)={token_len} diff_vs_x0(mean)={diff_mean:.6f} "{caption}" -> {i}.mp4')

    with open(pjoin(args.out_dir, 'captions.txt'), 'w') as f:
        f.write('\n'.join(caption_lines) + '\n')

    print(f'Saved {args.num_samples} comparison videos (mp4) + captions.txt to {args.out_dir}')
