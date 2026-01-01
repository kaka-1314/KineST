# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import json
import os
import random
import time
from tqdm import tqdm
import numpy as np

import torch
import datetime
from tensorboardX import SummaryWriter
from data_loaders.dataloader import get_dataloader, load_data, TrainDataset
from model.network import *
from runner.train_mlp import *
from runner.training_loop import TrainLoop
from torchstat import stat
from utils import dist_util

from utils.model_util import create_model_and_diffusion
from utils.parser_util import train_args
from human_body_prior.body_model.body_model import BodyModel

cuda_device = 7
seq_len = 20
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置可见的GPU设备


### NEW
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fk_module1(global_orientation, joint_rotation, body_model):
    global_orientation = utils_transform.sixd2aa(global_orientation.reshape(-1, 6)).reshape(global_orientation.shape[0],
                                                                                            -1).float()
    joint_rotation = utils_transform.sixd2aa(joint_rotation.reshape(-1, 6)).reshape(joint_rotation.shape[0], -1).float()
    body_pose = body_model(**{'pose_body': joint_rotation, 'root_orient': global_orientation})
    joint_position = body_pose.Jtr[:, :22]
    # joint_position = body_pose.Jtr
    return joint_position


def body_model():
    bm_fname_male = os.path.join("support_data/body_models", "smplh/{}/model.npz".format("male"))
    dmpl_fname_male = os.path.join(
        "support_data/body_models", "dmpls/{}/model.npz".format("male")
    )
    num_betas = 16  # number of body parameters
    num_dmpls = 8  # number of DMPL parameters
    body_model = BodyModel(
        bm_fname=bm_fname_male,
        num_betas=num_betas,
        num_dmpls=num_dmpls,
        dmpl_fname=dmpl_fname_male,
    )
    return body_model


def train_mlp_model(args, dataloader):
    print("creating MLP model...")
    args.arch = args.arch[len("mlp_"):]
    num_gpus = torch.cuda.device_count()
    args.num_workers = args.num_workers * num_gpus

    bm_fname_male = os.path.join("support_data/body_models", "smplh/{}/model.npz".format("male"))
    dmpl_fname_male = os.path.join(
        "support_data/body_models", "dmpls/{}/model.npz".format("male")
    )
    num_betas = 16  # number of body parameters
    num_dmpls = 8  # number of DMPL parameters
    body_model = BodyModel(
        bm_fname=bm_fname_male,
        num_betas=num_betas,
        num_dmpls=num_dmpls,
        dmpl_fname=dmpl_fname_male,
    )

    ### NEW_增加了时间参数&记录日志
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = './logs/pretrain/struct_tempt' + current_time  
    writer = SummaryWriter(log_dir)

    model = BiSAN_bitree_V2(
        latent_dim=256,
        d_state=16,
        expand=2,
        d_conv=4,
        num_layers=4,
        seq=96,
        body_model=body_model
    )
    # print(model)

    # model.load_state_dict(torch.load('output_model/tree/‎smooth_loss/model-iter-161470.pth'))
    model.train()

    if num_gpus > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        dist_util.setup_dist()
        model = torch.nn.DataParallel(model).cuda()
        print(
            "Total params: %.2fM"
                %(sum(p.numel() for p in model.module.parameters()) / 1000000.0)
            )
    else:
        print('device:', args.device)
        dist_util.setup_dist(0)
        model.to(dist_util.dev())
        print(
        "Total params: %.2fM"
        %(sum(p.numel() for p in model.parameters()) / 1000000.0)
        )


        ### print parameters' name and size
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"model: {name}, param: {param.numel()}")
    # initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # nb_iter = 161470 - 1
    nb_iter = 0
    avg_loss = 0.0
    avg_lr = 0.0

    while (nb_iter + 1) < args.num_steps:
        start_time = time.time()
        for (motion_target, motion_input) in tqdm(dataloader):
            # print(motion_input.shape)
            # print(stat(model,(256,196,54)))
        
            loss, optimizer, current_lr = smooth_loss(
                motion_input,
                motion_target,
                model,
                optimizer,
                nb_iter,
                args.num_steps,
                args.lr,
                args.lr / 10.0,
                dist_util.dev(),
                args.lr_anneal_steps,
                writer  
            )
            avg_loss += loss
            avg_lr += current_lr
            print("loss:",loss)
            if (nb_iter + 1) % args.log_interval == 0:
                avg_loss = avg_loss / args.log_interval
                avg_lr = avg_lr / args.log_interval
                # train_time1 = time.time()
                # print("Iter {} Summary: ".format(nb_iter + 1))
                # print(f"\t lr: {avg_lr} \t Training loss: {avg_loss} \t train_time: {start_time}")

                output_loss = avg_loss
                output_lr = avg_lr
                avg_loss = 0
                avg_lr = 0

            if (nb_iter + 1) == args.num_steps:
                break
            nb_iter += 1

        with open(
                os.path.join(args.save_dir, "model-iter-" + str(nb_iter + 1) + ".pth"),
                "wb",
        ) as f:
            torch.save(model.state_dict(), f)

        end_time = time.time()
        train_time = start_time - end_time
        print("Iter {} Summary: ".format(nb_iter + 1))
        print(f"\t lr: {output_lr} \t Training loss: {output_loss}")
        with open(os.path.join(args.save_dir, "log.txt"), "a") as file:
            print("Iter {} Summary:".format(nb_iter + 1), file=file)
            print(f"\t lr: {output_lr} \t Training loss: {output_loss} \t train_time: {train_time}", file=file)



def main():
    args = train_args()

    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.save_dir is None:
        raise FileNotFoundError("save_dir was not specified.")
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError("save_dir [{}] already exists.".format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, "args.json")
    with open(args_path, "w") as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    print("creating data loader...")
    motions, sparses, mean, std = load_data(
        args.dataset,
        args.dataset_path,
        "train",
        input_motion_length=args.input_motion_length,
    )
    dataset = TrainDataset(
        args.dataset,
        mean,
        std,
        motions,
        sparses,
        args.input_motion_length,
        args.train_dataset_repeat_times,
        args.no_normalization,
    )
    dataloader = get_dataloader(
        dataset, "train", batch_size=args.batch_size, num_workers=args.num_workers
    )

    model_type = args.arch.split("_")[0]
    if model_type == "mlp":
        train_mlp_model(args, dataloader)


if __name__ == "__main__":
    main()