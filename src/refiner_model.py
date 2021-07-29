import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import random
import sys
sys.path.append('./backbones/ms-tcn')
from model import SingleStageModel
import configs.refiner_config as cfg

class ResidualBlock(nn.Module):
    def __init__(self, feat_dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(nn.Conv1d(feat_dim, feat_dim, 3, 1, 1),
                                   nn.LeakyReLU(0.1),
                                   nn.Conv1d(feat_dim, feat_dim, 3, 1, 1),
                                   )
        
    def forward(self, inp):
        return self.block(inp)

class SparseSampleEmbedder(nn.Module):
    def __init__(self, in_dim, feat_dim, num_frames, num_samples):
        super(SparseSampleEmbedder, self).__init__()
        self.num_frames = num_frames
        self.num_samples = num_samples
        
        self.init_conv = nn.Conv1d(in_dim, feat_dim, 3, 1, 1)
        
        self.blocks = nn.ModuleList([ResidualBlock(feat_dim) for i in range(int(np.log2(self.num_frames)))])
        
        self.maxpool = nn.MaxPool1d(2, 2)
        self.relu = nn.LeakyReLU(0.1)
        
    def forward(self, inp):
        B, L, D = inp.shape
  
        sampled_inps = list()

        for i in range(self.num_samples):
            if L >= self.num_frames:
                sample_idx = sorted(random.sample([x for x in range(L)], self.num_frames))
            else:
                sample_idx = sorted([random.randint(0, L-1) for _ in range(self.num_frames)])
            tmp_inp = inp[:, sample_idx, :]
            sampled_inps.append(tmp_inp)

        sampled_inps = torch.cat(sampled_inps, dim=0) # self.num_samples, self.num_frames, D
        out = self.init_conv(sampled_inps.permute(0, 2, 1))
        
        for i in range(int(np.log2(self.num_frames))):
            out = self.relu(out + self.blocks[i](out))
            out = self.maxpool(out)
        
        out = torch.mean(out, dim=0).view(B, -1) # 1, D
        return out

class RefinerModel(nn.Module):
    def __init__(self, num_actions, input_dim, feat_dim, num_highlevel_frames, num_highlevel_samples, device):
        super(RefinerModel, self).__init__()
        
        self.key_embedding = nn.Linear(input_dim, feat_dim, bias=False)
        self.value_embedding = nn.Linear(input_dim, feat_dim, bias=False)
        self.query_embedding = nn.Embedding(num_actions, feat_dim)
        self.label_embedding = nn.Embedding(num_actions, feat_dim)
        
        self.video_embedding = SparseSampleEmbedder(feat_dim*2, feat_dim, num_frames=num_highlevel_frames, num_samples=num_highlevel_samples)
        
        self.refiner = nn.GRU(feat_dim*3, feat_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.refiner_out_linear = nn.Linear(feat_dim*2, num_actions)
        
        self.input_dim = input_dim
        self.feat_dim = feat_dim
        self.num_actions = num_actions
        self.device = device
        
    def get_segment_info(self, action_idx, batch_input, batch_target=None):        
        segment_idx = [0]
        prev_seg = action_idx[0]
        for ii, idx in enumerate(action_idx):
            if idx != prev_seg:
                segment_idx.append(ii)
            prev_seg = idx
        segment_idx.append(len(action_idx))
        
        GTlabel_list = list()
        PREDlabel_list = list()
        segment_feat = list()
        for s_i in range(len(segment_idx)-1):
            prev_idx = segment_idx[s_i]
            curr_idx = segment_idx[s_i+1]
            curr_seg = batch_input[:, :, prev_idx:curr_idx]

            if self.training and batch_target is not None:
                GTseg = batch_target[:, prev_idx:curr_idx]
                GTseg_label = torch.argmax(torch.bincount(GTseg[0]))
                GTlabel_list.append(GTseg_label)
            
            PREDseg_label = torch.mean(action_idx[prev_idx:curr_idx].float()).long()
            PREDlabel_list.append(PREDseg_label)
            
            curr_label_embed = self.query_embedding(PREDseg_label).view(1, -1, 1) # 1 D 1
            
            curr_key = self.key_embedding(curr_seg.permute(0, 2, 1))
            curr_value = self.value_embedding(curr_seg.permute(0, 2, 1)) # b l d
            
            curr_score = torch.bmm(curr_key, curr_label_embed) # B L 1
            curr_score = F.softmax(curr_score/np.sqrt(self.feat_dim), dim=1) # B L 1
            
            curr_attn_value = torch.bmm(curr_value.permute(0, 2, 1), curr_score)
            segment_feat.append(curr_attn_value.permute(0, 2, 1))
            
        GTlabel_list = torch.LongTensor(GTlabel_list).view(1, -1).to(self.device)
        PREDlabel_list = torch.LongTensor(PREDlabel_list).view(1, -1).to(self.device)
        segment_feat = torch.cat(segment_feat, dim=1)
        
        return segment_idx, segment_feat, PREDlabel_list, GTlabel_list
    
    def rollout(self, segment_idx, refine_pred):
        refine_rollout = []
        for s_i in range(len(segment_idx)-1):
            prev_idx = segment_idx[s_i]
            curr_idx = segment_idx[s_i+1]
            curr_refine = refine_pred[0, s_i, :].view(1, -1).repeat(curr_idx-prev_idx, 1)
            refine_rollout.append(curr_refine)
        refine_rollout = torch.cat(refine_rollout, dim=0).unsqueeze(0).transpose(2, 1) # B D L

        return refine_rollout    
    
    def forward(self, action_idx, batch_input, batch_target=None):
        B, D, L = batch_input.shape

        segment_idx, \
        segment_feat, \
        PREDlabel_list, \
        GTlabel_list = self.get_segment_info(action_idx, batch_input, batch_target)
        
        num_seg = segment_feat.shape[1]
        
        label_embed = self.label_embedding(PREDlabel_list)
        
        highlevel_inp = torch.cat([segment_feat, label_embed], dim=-1)
        highlevel_feat = self.video_embedding(highlevel_inp) # b d

        refine_input = torch.cat([segment_feat, label_embed,
                                  highlevel_feat.view(B, 1, -1).repeat(1, num_seg, 1)], dim=2) # B L D
        
        refine_pred, _ = self.refiner(refine_input)  # B L D
        refine_pred = self.refiner_out_linear(refine_pred) # B L D
        
        refine_rollout = self.rollout(segment_idx, refine_pred) # B D L
        
        
        return refine_pred, refine_rollout, GTlabel_list