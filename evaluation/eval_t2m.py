import os
import sys

import numpy as np
import torch
from MotionPriors import MotionPrior

from evaluation.get_opt import get_opt
from evaluation.metrics import *
from evaluation.motion_process import recover_from_ric
from evaluation.t2m_eval_wrapper import EvaluatorModelWrapper
from scipy.ndimage import gaussian_filter


@torch.no_grad()
def evaluation_vqvae_plus_mpjpe(val_loader, net, repeat_id, eval_wrapper, num_joint, test_mean, test_std, smooth_sigma=0.0):
    net.eval()
    device = next(net.parameters()).device
    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    mpjpe = 0
    num_poses = 0
    for batch in val_loader:
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch
        motion = motion.to(device)
        bs, seq = motion.shape[0], motion.shape[1]
        text_embedding = None
        if net.__class__.__name__ == "MotionPriorWrapper":
            if 'text_condition' in net.model_cfg.model.keys() and net.model_cfg.model.text_condition:
                text_embedding = net.model.net.encode_text(caption)
            if 'full_motion' in net.model_cfg.train.keys() and net.model_cfg.train.full_motion:
                pred_pose_eval, others = net(motion, text_embedding=text_embedding, m_length=m_length)
            else:
                pred_pose_eval, others = net(motion, text_embedding=text_embedding, m_length=None)

        elif net.__class__.__name__ == "RVQVAE":
            pred_pose_eval, commit_loss, perplexity = net(motion)
        else:
            try:
                pred_pose_eval, mu, logvar = net(motion)
            except:
                raise ValueError("Unknown model type")

        motion = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        pred_pose_eval = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())

        if smooth_sigma > 0:
            pred_pose_eval = gaussian_filter(pred_pose_eval, sigma=(0, smooth_sigma, 0))

        motion = (motion - test_mean) / test_std
        pred_pose_eval = (pred_pose_eval - test_mean) / test_std

        motion = torch.from_numpy(motion).to(device)
        pred_pose_eval = torch.from_numpy(pred_pose_eval).to(device)

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length)

        bgt = motion.detach().cpu().numpy() * test_std + test_mean
        bpred = pred_pose_eval.detach().cpu().numpy() * test_std + test_mean

        for i in range(bs):
            gt = recover_from_ric(torch.from_numpy(bgt[i, : m_length[i]]).float(), num_joint)
            pred = recover_from_ric(torch.from_numpy(bpred[i, : m_length[i]]).float(), num_joint)
            mpjpe += torch.sum(calculate_mpjpe(gt, pred))
            num_poses += gt.shape[0]

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample
    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    mpjpe = mpjpe / num_poses

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = (
        "--> \t Eva. Re %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, "
        "R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), "
        "matching_real. %.4f, matching_pred. %.4f, MPJPE. %.4f"
        % (
            repeat_id, fid, diversity_real, diversity,
            R_precision_real[0], R_precision_real[1], R_precision_real[2],
            R_precision[0], R_precision[1], R_precision[2],
            matching_score_real, matching_score_pred, mpjpe,
        )
    )
    print(msg)
    sys.stdout.flush()
    return fid, diversity, R_precision, matching_score_pred, mpjpe


def evaluate_motion_prior(test_dataloader, vae_model, model_cfg, device, vqvae=None, train_data="t2m", repeat_time=1):
    if train_data == "t2m":
        test_mean = np.load("./datasets/t2m-mean.npy")
        test_std = np.load("./datasets/t2m-std.npy")
    elif train_data == "kit":
        test_mean = np.load("./datasets/kit_mean.npy")
        test_std = np.load("./datasets/kit_std.npy")

    dataset_opt_path = f"./checkpoints/models/{train_data}/Comp_v6_KLD005/opt.txt"
    wrapper_opt = get_opt(dataset_opt_path, device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    EvaluatorModelWrapper.device = device
    nb_joints = 21 if train_data == "kit" else 22

    vae_model.eval()
    vae_model.to(device)

    net = MotionPrior.MotionPriorWrapper(model_cfg=model_cfg, model_ckpt=None, device=device)
    net.model = vae_model

    if vqvae is not None:
        net.vqvae = vqvae
    net.eval()
    net.to(device)

    fid, div, top1, top2, top3, matching, mae = [], [], [], [], [], [], []

    for i in range(repeat_time):
        best_fid, best_div, Rprecision, best_matching, l1_dist = evaluation_vqvae_plus_mpjpe(
            test_dataloader, net, i, eval_wrapper=eval_wrapper, num_joint=nb_joints, test_mean=test_mean, test_std=test_std
        )
        fid.append(best_fid)
        div.append(best_div)
        top1.append(Rprecision[0])
        top2.append(Rprecision[1])
        top3.append(Rprecision[2])
        matching.append(best_matching)
        mae.append(l1_dist)

    return {
        "FID": np.mean(fid),
        "Diversity": np.mean(div),
        "Top1": np.mean(top1),
        "Top2": np.mean(top2),
        "Top3": np.mean(top3),
        "Matching": np.mean(matching),
        "MAE": np.mean(mae),
    }


@torch.no_grad()
def evaluation_moscale_with_rf(
    val_loader,
    vq_model,
    trans,
    repeat_id,
    eval_wrapper,
    cond_scale,
    temperature=0.05,
    top_p_thres=0.9,
    sample_time=None,
    rf_model=None,
    cal_mm=True,
    use_refinement=True,  # False면 rf_model이 있어도 순수 가우시안 노이즈에서 시작 (refinement 이전 체크포인트 평가용)
):
    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0

    nb_sample = 0
    num_mm_batch = 3 if cal_mm else 0

    def generate_pred_motions(clip_text, m_length):
        mids = trans.generate(
            clip_text, m_length // 4, cond_scale,
            temperature=temperature, vq_model=vq_model,
            sample_time=sample_time, top_p_thres=top_p_thres,
        )
        if rf_model is None:
            return vq_model.forward_decoder(mids, m_length.clone())

        z = vq_model.quantizer.get_codes_from_indices(mids)
        m_len = m_length if rf_model.model_cfg.train.get('full_motion', False) else None

        if use_refinement:
            vqvae_out = vq_model.decoder(z.permute(0, 2, 1))
            x0 = vqvae_out + rf_model.model_cfg.model.noise_std * torch.randn_like(vqvae_out)
        else:
            x0 = None  # 순수 가우시안 노이즈에서 시작 (refinement 이전 체크포인트와 호환)

        return rf_model.sample_from_z(z, m_length=m_len, noise=x0)

    for i, batch in enumerate(val_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.cuda()
        bs, seq = pose.shape[:2]

        if i < num_mm_batch:
            motion_multimodality_batch = []
            for _ in range(30):
                pred_motions = generate_pred_motions(clip_text, m_length)
                et_pred, em_pred = eval_wrapper.get_co_embeddings(
                    word_embeddings, pos_one_hots, sent_len, pred_motions.clone(), m_length
                )
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            pred_motions = generate_pred_motions(clip_text, m_length)
            et_pred, em_pred = eval_wrapper.get_co_embeddings(
                word_embeddings, pos_one_hots, sent_len, pred_motions.clone(), m_length
            )

        pose = pose.cuda().float()
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match

        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)

    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample
    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = (
        f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, "
        f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, "
        f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, "
        f"matching_score_real. {matching_score_real:.4f}, matching_score_pred. {matching_score_pred:.4f}, "
        f"multimodality. {multimodality:.4f}"
    )
    print(msg)
    sys.stdout.flush()
    return fid, diversity, R_precision, matching_score_pred, multimodality
