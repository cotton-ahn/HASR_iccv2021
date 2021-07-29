import torch
import torch.nn as nn
import numpy as np
import random
import sys
sys.path.append('./backbones/asrf')
from libs.postprocess import PostProcessor

def refiner_train(cfg, dataset, train_loader, model, backbones, backbone_names, optimizer, epoch, split_dict, device):
    normal_ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    for idx, sample in enumerate(train_loader):
        model.train()
        x = sample['feature']
        t = sample['label']
        
        split_idx = 0
        for i in range(eval('cfg.num_splits["{}"]'.format(dataset))):
            if sample['feature_path'][0].split('/')[-1].split('.')[0] in split_dict[i+1]:
                split_idx = i+1
                break
                
        bb_key = random.choice(backbone_names)
        curr_backbone = backbones[bb_key][split_idx]
        curr_backbone.load_state_dict(torch.load('{}/{}/{}/split_{}/epoch-{}.model'.format(cfg.model_root,
                                                                                           bb_key,
                                                                                           dataset,
                                                                                           str(i+1),
                                                                                           np.random.randint(10, 51))))
        curr_backbone.to(device)
        curr_backbone.eval()

        x, t = x.to(device), t.to(device)
        
        B, L, D = x.shape
        
        if bb_key == 'mstcn':
            mask = torch.ones(x.size(), device=device)
            action_pred = curr_backbone(x, mask)
            action_idx = torch.argmax(action_pred[-1], dim=1).squeeze().detach()
        
        elif bb_key == 'mgru':
            action_pred = curr_backbone(x)
            action_idx = torch.argmax(action_pred, dim=1).squeeze().detach()
            
        elif bb_key == 'sstda':
            mask = torch.ones(x.size(), device=device)
            action_pred, _, _, _, _, _, _, _, _, _, _, _, _, _ = curr_backbone(x, 
                                                                               x, 
                                                                               mask, 
                                                                               mask, 
                                                                               [0, 0], 
                                                                               reverse=False)
            action_idx = torch.argmax(action_pred[:, -1, :, :], dim=1).squeeze().detach()
            
        elif bb_key == 'asrf':
            out_cls, out_bound = curr_backbone(x)
            postprocessor = PostProcessor("refinement_with_boundary", cfg.boundary_th)
            refined_output_cls = postprocessor(out_cls.cpu().data.numpy(), boundaries=out_bound.cpu().data.numpy(), 
                                               masks=torch.ones(1, 1, x.shape[-1]).bool().data.numpy())
            action_idx = torch.Tensor(refined_output_cls).squeeze().detach()

        refine_pred, refine_rollout, GTlabel_list = model(action_idx.to(device), x, t)
        
        loss = 0.0
        loss += normal_ce(refine_pred[0], GTlabel_list.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss / len(train_loader)
        
    return total_loss.item()