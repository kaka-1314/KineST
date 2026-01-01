# Copyright (c) Meta Platforms, Inc. All Rights Reserved
from operator import concat

import torch
from torchstat import stat
import torch.nn as nn
from model.networks_a6000 import log_map_SO3
# import train_mamba2p_trans_struct
import train1
import torch.nn.functional as F
from model.networks import M2SP_transformer_struct_temp_loss1 as model1
from human_body_prior.body_model.body_model import BodyModel
from utils import utils_transform

def update_lr_multistep(
    nb_iter, total_iter, max_lr, min_lr, optimizer, lr_anneal_steps
):
    if lr_anneal_steps < nb_iter:
        current_lr = min_lr
    else:
        current_lr = max_lr

    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr


def angle_velocity_loss(pred_6d, gt_6d, interval=1, loss_func=nn.L1Loss()):
    """
    pred_6d, gt_6d: [B, T, J, 6]
    interval: int, the temporal interval to compute velocity (default=1)
    """
    def sixd_to_matrix(sixd):  # (..., 6) → (..., 3, 3)
        x_raw = sixd[..., :3]
        y_raw = sixd[..., 3:6]
        x = F.normalize(x_raw, dim=-1)
        z = F.normalize(torch.cross(x, y_raw, dim=-1), dim=-1)
        y = torch.cross(z, x, dim=-1)
        return torch.stack([x, y, z], dim=-2)  # (..., 3, 3)

    def log_map_SO3(R):  # (..., 3, 3) → (..., 3)
        cos_theta = (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2
        cos_theta = torch.clamp(cos_theta, -1 + 1e-5, 1 - 1e-5)
        theta = torch.acos(cos_theta)
        skew = (R - R.transpose(-1, -2)) / (2 * torch.sin(theta)[..., None, None] + 1e-5)
        x = skew[..., 2, 1]
        y = skew[..., 0, 2]
        z = skew[..., 1, 0]
        return theta.unsqueeze(-1) * torch.stack([x, y, z], dim=-1)  # (..., 3)

    R_pred = sixd_to_matrix(pred_6d)  # [B, T, J, 3, 3]
    R_gt = sixd_to_matrix(gt_6d)

    # 角速度：log(R_{t+interval} @ R_t^T)
    rel_R_pred = torch.matmul(R_pred[:, interval:].transpose(-1, -2), R_pred[:, :-interval])
    rel_R_gt = torch.matmul(R_gt[:, interval:].transpose(-1, -2), R_gt[:, :-interval])

    angle_vel_pred = log_map_SO3(rel_R_pred)  # [B, T-interval, J, 3]
    angle_vel_gt = log_map_SO3(rel_R_gt)

    loss = loss_func(angle_vel_pred, angle_vel_gt)
    return loss


def smooth_loss(motion_input,
        motion_target,
        model,
        optimizer,
        nb_iter,
        total_iter,
        max_lr,
        min_lr,
        device,
        lr_anneal_steps,
        writer
):
    body_model = BodyModel(
        bm_fname="./support_data/body_models/smplh/male/model.npz",
        num_betas=16,
        num_dmpls=8,
        dmpl_fname="./support_data/body_models/dmpls/male/model.npz",
    )
    body_model.to(device)
    motion_input = motion_input.to(device)
    motion_target = motion_target.to(device)
    root0_target = motion_target[..., :6].to(device)
    root1_target = motion_target[..., 6:12].to(device)
    root2_target = motion_target[..., 12:18].to(device)
    root3_target = motion_target[..., 6 * 3:6 * 4].to(device)
    # head_target = motion_target[..., 6 * 15:6 * 16].to(device)
    # hand0_target = motion_target[..., 6 * 20:6 * 21].to(device)
    # hand1_target = motion_target[..., 6 * 21:6 * 22].to(device)
    target_feature = [root0_target, root1_target, root2_target, root3_target]
    motion_pred, joint_position0 = model(motion_input)
    # motion_pred, joint_feature = model(motion_input)
    root0_pred = motion_pred[..., :6].to(device)
    root1_pred = motion_pred[..., 6:12].to(device)
    root2_pred = motion_pred[..., 12:18].to(device)
    root3_pred = motion_pred[..., 6 * 3:6 * 4].to(device)
    joint_feature = [root0_pred, root1_pred, root2_pred, root3_pred]

    global_orientation1 = motion_target[:, :, :6].to(device)
    joint_rotation1 = motion_target[:, :, 6:132].to(device)

    loss_global = torch.mean(
        torch.norm(
            (motion_pred - motion_target),
            2,
            1,
        )
    )
    
    loss_smooth = angle_velocity_loss(motion_pred[..., :132].view(motion_pred.shape[0], motion_pred.shape[1], 22, 6), motion_target[..., :132].view(motion_pred.shape[0], motion_pred.shape[1], 22, 6))
    
    loss_root0 = torch.mean(
        torch.norm(
            (joint_feature[0] - target_feature[0]),  
            2,  
            1,  
        )
    )
    loss = loss_global + 0.02 * loss_root0 + loss_smooth
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer, current_lr = update_lr_multistep(
        nb_iter, total_iter, max_lr, min_lr, optimizer, lr_anneal_steps
    )
    writer.add_scalar('loss', loss.item(), nb_iter)

    return loss.item(), optimizer, current_lr

