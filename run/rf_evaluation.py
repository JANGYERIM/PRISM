import os
import sys
sys.path.append('/nas2/data/dpfla3573/code/MoScale')

from os.path import join as pjoin
import torch
import numpy as np
import argparse
from tqdm import tqdm

from model.transformer.moscale import MoScale
from model.vq.hrvqvae import HRVQVAE
from config.load_config import load_config

from MotionPriors import MotionPrior
from evaluation.get_opt import get_opt
from evaluation.t2m_eval_wrapper import EvaluatorModelWrapper
from evaluation import eval_t2m
from datasets.t2m_dataset import Text2MotionDatasetEval, collate_fn
from utils.word_vectorizer import WordVectorizer
from utils.utils import generate_date_time, seed_everything
from configs import config_utils


def load_vq_model(vq_cfg, moscale_cfg, checkpoints_dir, train_data, device):
    vq_model = HRVQVAE(
        vq_cfg,
        vq_cfg.data.dim_pose,
        vq_cfg.model.down_t,
        vq_cfg.model.stride_t,
        vq_cfg.model.width,
        vq_cfg.model.depth,
        vq_cfg.model.dilation_growth_rate,
        vq_cfg.model.vq_act,
        vq_cfg.model.use_attn,
        vq_cfg.model.vq_norm,
    )
    ckpt = torch.load(
        pjoin(checkpoints_dir, train_data, 'hrvqvae', moscale_cfg.vq_name, 'model', moscale_cfg.vq_ckpt),
        map_location=device,
    )
    model_key = 'vq_model' if 'vq_model' in ckpt else 'model'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading HRVQVAE from {moscale_cfg.vq_name}')
    vq_model.to(device)
    vq_model.eval()
    return vq_model


def load_trans_model(moscale_cfg, which_model, moscale_name, checkpoints_dir, train_data, device):
    moscale = MoScale(
        code_dim=moscale_cfg.vq.code_dim,
        latent_dim=moscale_cfg.model.latent_dim,
        num_heads=moscale_cfg.model.n_heads,
        dropout=moscale_cfg.model.dropout,
        attn_drop_rate=moscale_cfg.model.attn_drop_rate,
        text_dim=moscale_cfg.text_embedder.dim_embed,
        cond_drop_prob=moscale_cfg.training.cond_drop_prob,
        device=device,
        cfg=moscale_cfg,
        full_length=moscale_cfg.data.max_motion_length // 4,
        scales=[8, 4, 2, 1],
    )
    ckpt = torch.load(
        pjoin(checkpoints_dir, train_data, 'moscale', moscale_name, 'model', which_model),
        map_location=device,
    )
    moscale.load_state_dict(ckpt['moscale'])
    print(f'Loading MoScale from {moscale_name}, epoch {ckpt["ep"]}')
    moscale.to(device)
    moscale.eval()
    return moscale


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_cfg_pth', type=str, default='./configs/moscale_eval_config_t2m.yaml')
    parser.add_argument('--data_cfg_path', type=str, default='./configs/config_data.yaml')
    parser.add_argument('--model_cfg_path', type=str, default='./configs/config_model.yaml')
    parser.add_argument('--train_data', type=str, default='t2m', choices=['t2m', 'kit'])
    parser.add_argument('--num_sample_steps', type=int, default=16)
    parser.add_argument('--model_ckpt_path', type=str, default=None)
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/models')
    parser.add_argument('--seed', type=int, default=24)
    parser.add_argument('--no_refinement', action='store_true',
                         help='disable refinement (x0=decoder(z)+noise) and start from pure Gaussian noise instead -- '
                              'use this to evaluate checkpoints trained before refinement was added')
    args = parser.parse_args()

    eval_cfg = config_utils.get_yaml_config(args.eval_cfg_pth)
    data_cfg = config_utils.get_yaml_config(args.data_cfg_path)
    model_cfg = config_utils.get_yaml_config(args.model_cfg_path)

    opt = eval_cfg
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.device = str(device)

    if args.train_data == 't2m':
        data_cfg = data_cfg.t2m
    elif args.train_data == 'kit':
        data_cfg = data_cfg.kit

    # Load MoScale configs from saved yamls in checkpoint dir
    moscale_cfg = load_config(pjoin(args.checkpoints_dir, args.train_data, 'moscale', opt.moscale_name, 'train_moscale.yaml'))
    vq_cfg = load_config(pjoin(args.checkpoints_dir, args.train_data, 'hrvqvae', opt.vq_name, 'train_hrvqvae.yaml'))
    moscale_cfg.vq = vq_cfg.quantizer

    vq_model = load_vq_model(vq_cfg, moscale_cfg, args.checkpoints_dir, args.train_data, device)
    trans = load_trans_model(moscale_cfg, 'net_best_fid.tar', opt.moscale_name, args.checkpoints_dir, args.train_data, device)

    dataset_opt_path = pjoin(args.checkpoints_dir, args.train_data, 'Comp_v6_KLD005', 'opt.txt')
    wrapper_opt = get_opt(dataset_opt_path, device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    EvaluatorModelWrapper.device = device

    if args.train_data == 't2m':
        test_mean = np.load('./datasets/t2m-mean.npy')
        test_std = np.load('./datasets/t2m-std.npy')
    else:
        test_mean = np.load('./datasets/kit_mean.npy')
        test_std = np.load('./datasets/kit_std.npy')

    if args.model_ckpt_path:
        meta_dir = os.path.dirname(os.path.dirname(args.model_ckpt_path)) + '/meta'
        mean = np.load(meta_dir + '/mean.npy')
        std = np.load(meta_dir + '/std.npy')
        rf_model = MotionPrior.MotionPriorWrapper(model_cfg, args.model_ckpt_path, device)
        rf_model.eval()
        rf_model.num_sample_steps = args.num_sample_steps
        rf_model.set_vqvae()
    else:
        rf_model = None
        mean = test_mean
        std = test_std

    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    test_dataset = Text2MotionDatasetEval(data_cfg, mean, std, w_vectorizer, split='test')
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, drop_last=True,
        collate_fn=collate_fn, shuffle=True, pin_memory=True,
    )

    fid, div, top1, top2, top3, matching, mm = [], [], [], [], [], [], []
    repeat_time = 20
    for i in tqdm(range(repeat_time), desc='Repeat Time'):
        with torch.no_grad():
            best_fid, best_div, Rprecision, best_matching, best_mm = \
                eval_t2m.evaluation_moscale_with_rf(
                    test_dataloader,
                    vq_model,
                    trans,
                    i,
                    eval_wrapper=eval_wrapper,
                    cond_scale=opt.cond_scale,
                    temperature=opt.temperature,
                    top_p_thres=opt.top_p_thres,
                    sample_time=list(opt.sample_time),
                    rf_model=rf_model,
                    cal_mm=True,
                    use_refinement=not args.no_refinement,
                )
        fid.append(best_fid)
        div.append(best_div)
        top1.append(Rprecision[0])
        top2.append(Rprecision[1])
        top3.append(Rprecision[2])
        matching.append(best_matching)
        mm.append(best_mm)

    fid = np.array(fid)
    div = np.array(div)
    top1 = np.array(top1)
    top2 = np.array(top2)
    top3 = np.array(top3)
    matching = np.array(matching)
    mm = np.array(mm)

    if args.model_ckpt_path:
        base_dir = os.path.dirname(os.path.dirname(args.model_ckpt_path))
        model_name = os.path.basename(args.model_ckpt_path).split('.')[0]
        eval_dir = pjoin(base_dir, 'eval')
        os.makedirs(eval_dir, exist_ok=True)
        file = pjoin(eval_dir, f'{model_name}_eval_moscale_step{args.num_sample_steps}_seed{args.seed}_date{generate_date_time()}.txt')
    else:
        eval_dir = pjoin(args.checkpoints_dir, args.train_data, 'moscale', opt.moscale_name, 'eval')
        os.makedirs(eval_dir, exist_ok=True)
        file = pjoin(eval_dir, f'eval_seed{args.seed}_date{generate_date_time()}.txt')

    f = open(file, 'w')
    print(f'{file} final result:')
    print(f'{file} final result:', file=f, flush=True)

    msg_final = (
        f"\tFID: {np.mean(fid):.3f}, conf. {np.std(fid) * 1.96 / np.sqrt(repeat_time):.3f}\n"
        f"\tDiversity: {np.mean(div):.3f}, conf. {np.std(div) * 1.96 / np.sqrt(repeat_time):.3f}\n"
        f"\tTOP1: {np.mean(top1):.3f}, conf. {np.std(top1) * 1.96 / np.sqrt(repeat_time):.3f}, "
        f"TOP2. {np.mean(top2):.3f}, conf. {np.std(top2) * 1.96 / np.sqrt(repeat_time):.3f}, "
        f"TOP3. {np.mean(top3):.3f}, conf. {np.std(top3) * 1.96 / np.sqrt(repeat_time):.3f}\n"
        f"\tMatching: {np.mean(matching):.3f}, conf. {np.std(matching) * 1.96 / np.sqrt(repeat_time):.3f}\n"
        f"\tMultimodality:{np.mean(mm):.3f}, conf.{np.std(mm) * 1.96 / np.sqrt(repeat_time):.3f}\n\n"
    )
    print(msg_final)
    print(msg_final, file=f, flush=True)
    f.close()
