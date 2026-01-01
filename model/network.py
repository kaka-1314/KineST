import torch.nn as nn
from mamba_ssm.modules.mamba2_simple import Mamba2Simple
import math
import os
from mamba_ssm import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba2_simple import Mamba2Simple

from utils.parser_util import train_args
from utils import utils_transform
from human_body_prior.body_model.body_model import BodyModel
import torch
from utils.parser_util import train_args
from FAN.Periodicity_Modeling import architecture
import torch.nn.functional as F

###############################
############ Layers ###########
###############################
  


class Base_SAN21_Bi(nn.Module):
    def __init__(self, dim, d_state, expand, d_conv,embed_dim, output_dims, input_dims, num_layers, nhead, seq):
        super().__init__()

        layers = []
        for i in range(num_layers):
            layers.append(
                SAN21_Bi_block(dim, d_state, expand, d_conv, embed_dim, output_dims, input_dims, nhead, seq)
            )

        self.mlps = nn.Sequential(*layers)

    def forward(self, x):
        x = self.mlps(x)
        return x


class SAN21_Bi_block(nn.Module):
    def __init__(self, dim, d_state=16, expand=2, d_conv=4, embed_dim=256, output_dims=256, input_dims=256, nhead=8,
                 seq=96):
        super().__init__()

        self.mamba2_for = Mamba2Simple(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand, headdim=64)
        self.mamba2_back = Mamba2Simple(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand, headdim=64)
        self.norm0 = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.act = nn.SiLU()
        self.linear_embedding1 = nn.Linear(input_dims, embed_dim)
        self.linear_embedding2 = nn.Linear(embed_dim, output_dims)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Conv1d(seq, seq, 1),
            nn.SiLU()
        )
        self.fan = architecture.FAN(dim, dim, hidden_dim=512, num_layers=1)

    def Bi_mamba2(self, x):
        x_forward = self.mamba2_for(x)
        x_backward = self.mamba2_back(x.flip(1)).flip(1)  # Reverse the sequence for backward pass
        x = x_forward + x_backward  # Combine forward and backward outputs
        return x

    def forward(self, inputs):
        x = inputs
        x_ = self.norm0(x)
        x_ = self.Bi_mamba2(x_)
        x1 = x + x_  # torch.Size([256, 196, 512])

        x = self.net(x1)  # 512 --> 256
        x2 = x + x1

        x_ = self.linear_embedding1(x2)  # input:256  output:512
        x_ = x_.permute(1, 0, 2)
        x_ = self.norm1(x_)
        x_ = self.transformer_encoder(x_)  # input=output=512
        x_ = self.fan(x_)
        x_ = self.act(x_)
        x_ = x_.permute(1, 0, 2)
        x_ = self.linear_embedding2(x_)  # input:512  output:256
        x = x_ + x2  # torch.Size([256, 196, 512])

        return x


class Base_BiSAN_bitree_V3(nn.Module):
    def __init__(self, dim, d_state, expand, d_conv,embed_dim, output_dims, input_dims, num_layers, nhead, seq):
        super().__init__()

        layers = []
        for i in range(num_layers):
            layers.append(
                BiSAN_bitree_block_V3(dim, d_state, expand, d_conv, embed_dim, output_dims, input_dims, nhead, seq)
            )

        self.mlps = nn.Sequential(*layers)

    def forward(self, x):
        x = self.mlps(x)
        return x


class BiSAN_bitree_block_V3(nn.Module):
    def __init__(self, dim, d_state=16, expand=2, d_conv=4, embed_dim=256, output_dims=256, input_dims=256, nhead=8,
                 seq=96, joints=22, jdim=64):
        super().__init__()

        self.joints = joints
        self.jdim = jdim
        # self.mamba2_for = Mamba2Simple(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand, headdim=64)
        # self.mamba2_back = Mamba2Simple(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand, headdim=64)
        self.mamba2_for1 = Mamba2Simple(d_model=jdim, d_state=d_state, d_conv=d_conv, expand=expand, headdim=16)
        self.mamba2_back1 = Mamba2Simple(d_model=jdim, d_state=d_state, d_conv=d_conv, expand=expand, headdim=16)
        self.norm0 = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        # self.norm2 = nn.LayerNorm(dim)
        self.act = nn.SiLU()
        self.linear0 = nn.Linear(dim, joints * jdim)
        self.linear1 = nn.Linear(joints * jdim, dim)  # input:22*64 output:256
        self.linear_embedding1 = nn.Linear(input_dims, embed_dim)
        self.linear_embedding2 = nn.Linear(embed_dim, output_dims)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Conv1d(seq, seq, 1),
            nn.SiLU()
        )
        self.fan = architecture.FAN(dim, dim, hidden_dim=512, num_layers=1)
        self.fan2 = architecture.FAN(joints * jdim, dim, hidden_dim=512, num_layers=1)


    def scan_along_joints(self, x, joint_idx, mamba_module):
        # x: [B, S, J, C]
        # joint_idx: list[int], like [21, 19, 17, ...]
        x = x[:, :, joint_idx, :]  # [B, S, len, C]
        B, S, J_, C_ = x.shape
        # x = x.view(B * S, J_, C_)  # 
        x = x.view(B, S * J_, C_)
        out = mamba_module(x)  # Mamba: [B * J_, S, C]
        out = out.view(B, S, J_, C_).contiguous()  # [B, S, J_, C]
        return out

    def Bitree_mamba2(self, x3):  #  [b, s, dim]
        forward_chain = [21, 19, 17, 14, 15, 12, 20, 18, 16, 13, 9, 6, 3, 0, 1, 4, 10, 2, 5, 8, 11]
        backward_chain = [11, 8, 5, 2, 10, 4, 1, 0, 3, 6, 9, 13, 16, 18, 20, 12, 15, 14, 17, 19, 21]
        B, S = x3.shape[0], x3.shape[1]

        x = self.linear0(x3)  # [b,s,256]-->x=[b,s, 22*64]
        x = x.view(x.shape[0], x.shape[1], self.joints, self.jdim)  # [b*s,22,64]
        x_forward = self.scan_along_joints(x, forward_chain, self.mamba2_for1)
        x_backward = self.scan_along_joints(x, backward_chain, self.mamba2_back1)

        x_full_f = torch.zeros_like(x)
        x_full_b = torch.zeros_like(x)

        idx_f = torch.tensor(forward_chain, device=x.device).view(1, 1, -1, 1).expand(B, S, -1, self.jdim)
        idx_b = torch.tensor(backward_chain, device=x.device).view(1, 1, -1, 1).expand(B, S, -1, self.jdim)
        x_full_f.scatter_(dim=2, index=idx_f, src=x_forward)
        x_full_b.scatter_(dim=2, index=idx_b, src=x_backward)
        x_full = x_full_f + x_full_b  # [b, s, 22, 64]

        x_full = x_full.reshape(x3.shape[0], x3.shape[1], self.joints * self.jdim)  # [b,s,22*64]
        x_full = self.fan2(x_full) # input:22*64  output:256
        return x_full


    def forward(self, inputs):
        x = inputs # [b,s,256]

        '''temporal feature extraction'''
        x_ = self.norm0(x)
        x_ = self.Bitree_mamba2(x_)
        x1 = x + x_

        x = self.net(x1)
        x2 = x + x1

        x_ = self.linear_embedding1(x2)  # [b,s,d]
        x_ = x_.permute(1, 0, 2) # [s, b, d]?
        x_ = self.norm1(x_)
        x_ = self.transformer_encoder(x_)  # input=output=512
        x_ = self.fan(x_)
        x_ = self.act(x_)
        x_ = x_.permute(1, 0, 2)
        x_ = self.linear_embedding2(x_)  # input:512  output:256
        x = x_ + x2  # torch.Size([256, 96, 256])



        return x
  
  
    
class BiSAN_bitree_V2(nn.Module):
    def __init__(
            self, latent_dim=256, input_dim=54, output_dim=132, input_dims=256, seq=196, output_dims=256, num_layer=2,
            embed_dim=256, nhead=8, d_state=16, expand=2, d_conv=4, num_layers=3, body_model=None
    ):
        super(BiSAN_bitree_V2,self).__init__()
        self.input_fc = nn.Linear(input_dim, latent_dim)
        self.temporary = Base_SAN21_Bi(
            dim=latent_dim, d_state=d_state, expand=expand, d_conv=d_conv, embed_dim=embed_dim, output_dims=output_dims,
            input_dims=input_dims, nhead=nhead, num_layers=2, seq = seq
        )
        self.temp_spatial = Base_BiSAN_bitree_V3(
            dim=latent_dim, d_state=d_state, expand=expand, d_conv=d_conv, embed_dim=embed_dim, output_dims=output_dims,
            input_dims=input_dims, nhead=nhead, num_layers=2, seq=seq
        )
        self.output_fc = nn.Linear(latent_dim, output_dim)
        self.body_model = body_model

    def fk_module(self, global_orientation, joint_rotation, body_model):
        B, T, _ = global_orientation.shape
        global_orientation = utils_transform.sixd2aa(global_orientation.reshape(-1, 6)).reshape(B * T, -1).float()
        joint_rotation = joint_rotation.reshape(-1, 6)
        joint_rotation = utils_transform.sixd2aa(joint_rotation).reshape(B * T, -1).float()
        body_pose = body_model(**{'pose_body': joint_rotation, 'root_orient': global_orientation}) #########
        joint_position = body_pose.Jtr[:,:22,:]  # [B*T, J, 3]
        # joint_position[:, 0] = 0.0  
        joint_position = joint_position.reshape(B, T, *joint_position.shape[1:])  # [B, T, J, 3]

        return joint_position

    def forward(self, input_tensor):
        x = self.input_fc(input_tensor)  # 54 --> 512
        x = self.temporary(x)  # 256
        x = self.temp_spatial(x)
        motion_feats = self.output_fc(x)  # 256 --> 132
        global_orientation = motion_feats[:, 0:1, :6]
        joint_rotation = motion_feats[:, 0:1, 6:132]
        joint_position = self.fk_module(global_orientation, joint_rotation, self.body_model)  # [256, 52, 3]

        return motion_feats, joint_position

 
  
