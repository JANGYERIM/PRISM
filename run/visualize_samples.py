"""
testset에서 랜덤 N개를 뽑아 gt / 생성된 모션(moscale+discord) / moscale base만 디코딩한 모션을
한 gif 안에 나란히 애니메이션으로 저장하는 스크립트. caption과 length는 captions.txt에 기록.

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
    parser.add_argument('--model_cfg_path', type=str, default="./checkpoints/RFDecoder_t2m_fmTrue_bs384_ep500_PostRefinement_0727224718/configs/config_model.yaml")
    parser.add_argument('--model_ckpt_path', type=str, default="./checkpoints/RFDecoder_t2m_fmTrue_bs384_ep500_PostRefinement_0727224718/checkpoints/Unet1DforFlowDecoder_best_1_20.955944.pth", help='RFDecoder(discord) checkpoint')
    parser.add_argument('--train_data', type=str, default='t2m', choices=['t2m', 'kit'])
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/models')
    parser.add_argument('--num_sample_steps', type=int, default=16)
    parser.add_argument('--num_samples', type=int, default=30)
    parser.add_argument('--out_dir', type=str, default='./samples_vis')
    parser.add_argument('--seed', type=int, default=24)
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
        x0 = vqvae_out + rf_model.model_cfg.model.noise_std * torch.randn_like(vqvae_out)
        discord_motion = rf_model.sample_from_z(z, m_length=m_len, noise=x0)

    caption_lines = ['idx\ttoken_len(gt)\tframe_len(gt)\tcaption']
    for i in range(args.num_samples):
        length = int(m_length[i].item())
        token_len = length // 4  # trans.generate에 강제로 넘긴 길이 (evaluation과 동일한 방식)
        caption = clip_text[i]

        gt_np = pose[i, :length].cpu().numpy() * std + mean
        discord_np = discord_motion[i, :length].cpu().numpy() * std + mean
        base_np = base_motion[i, :length].cpu().numpy() * std + mean

        plot_3d_motion_compare(
            pjoin(args.out_dir, f'{i}.gif'),
            [gt_np, discord_np, base_np],
            sub_titles=['GT', 'Generated (moscale+discord)', 'MoScale base'],
            title=f'[{i}] {caption}',
            figsize=(4, 4), fps=20,
        )
        caption_lines.append(f'{i}\t{token_len}\t{length}\t{caption}')
        print(f'[{i}] token_len(gt)={token_len} "{caption}" -> {i}.gif')

    with open(pjoin(args.out_dir, 'captions.txt'), 'w') as f:
        f.write('\n'.join(caption_lines) + '\n')

    print(f'Saved {args.num_samples} comparison gifs + captions.txt to {args.out_dir}')
