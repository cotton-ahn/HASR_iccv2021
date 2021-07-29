import torch
import numpy as np
import configs.asrf_config as asrf_cfg
import sys
sys.path.append('./backbones/asrf')
from libs.postprocess import PostProcessor
    
    
def predict_refiner(model, main_backbone_name, backbones, split_dict, model_dir, result_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
    model.eval()
    with torch.no_grad():
        model.to(device)
        model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
        file_ptr = open(vid_list_file, 'r')
        list_of_vids = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        for vid in list_of_vids:
            features = np.load(features_path + vid.split('.')[0] + '.npy')
            features = features[:, ::sample_rate]
            input_x = torch.tensor(features, dtype=torch.float)
            input_x.unsqueeze_(0)
            input_x = input_x.to(device)
            
            split_idx = 0
            for i in range(len(split_dict.keys())):
                if vid.split('.')[0] in split_dict[i+1]:
                    split_idx = i+1
                    break
            
            curr_backbone = backbones[split_idx]
            curr_backbone.eval()
            
            if main_backbone_name != 'asrf':
                if main_backbone_name == 'mstcn':
                    mask = torch.ones(input_x.size(), device=device)
                    action_pred = curr_backbone(input_x, mask)[-1]
                elif main_backbone_name == 'mgru':
                    action_pred = curr_backbone(input_x)
                elif main_backbone_name == 'sstda':
                    mask = torch.ones(input_x.size(), device=device)
                    action_pred, _, _, _, _, _, _, _, _, _, _, _, _, _ = curr_backbone(input_x, 
                                                                                       input_x, 
                                                                                       mask, 
                                                                                       mask, 
                                                                                       [0, 0], 
                                                                                       reverse=False)
                    action_pred = action_pred[:, -1, :, :]

                action_idx = torch.argmax(action_pred, dim=1).squeeze().detach()
                
            else:
                out_cls, out_bound = curr_backbone(input_x)
                postprocessor = PostProcessor("refinement_with_boundary", asrf_cfg.boundary_th)
                refined_output_cls = postprocessor(out_cls.cpu().data.numpy(), boundaries=out_bound.cpu().data.numpy(), 
                                                   masks=torch.ones(1, 1, input_x.shape[-1]).bool().data.numpy())
                action_idx = torch.Tensor(refined_output_cls).squeeze().detach()

            _, predictions, _ = model(action_idx.to(device), input_x)
            _, predicted = torch.max(predictions.data, 1)
                
            predicted = predicted.squeeze()
            recognition = []
            for i in range(len(predicted)):
                recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
            f_name = vid.split('/')[-1].split('.')[0]
            f_ptr = open(result_dir + "/" + f_name, "w")
            f_ptr.write("### Frame level recognition: ###\n")
            f_ptr.write(' '.join(recognition))
            f_ptr.close()
            

def predict_refiner_best(model, main_backbone_name, backbones, split_dict, best_fp, result_dir, features_path, vid_list_file_tst, actions_dict, device, sample_rate):
    model.eval()
    with torch.no_grad():
        model.to(device)
        model.load_state_dict(torch.load(best_fp))
        file_ptr = open(vid_list_file_tst, 'r')
        list_of_vids = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        for vid in list_of_vids:
            features = np.load(features_path + vid.split('.')[0] + '.npy')
            features = features[:, ::sample_rate]
            input_x = torch.tensor(features, dtype=torch.float)
            input_x.unsqueeze_(0)
            input_x = input_x.to(device)
            
            split_idx = 0
            for i in range(len(split_dict.keys())):
                if vid.split('.')[0] in split_dict[i+1]:
                    split_idx = i+1
                    break
            
            curr_backbone = backbones[split_idx]
            curr_backbone.eval()
            
            if main_backbone_name != 'asrf':
                if main_backbone_name == 'mstcn':
                    mask = torch.ones(input_x.size(), device=device)
                    action_pred = curr_backbone(input_x, mask)[-1]
                elif main_backbone_name == 'mgru':
                    action_pred = curr_backbone(input_x)
                elif main_backbone_name == 'sstda':
                    mask = torch.ones(input_x.size(), device=device)
                    action_pred, _, _, _, _, _, _, _, _, _, _, _, _, _ = curr_backbone(input_x, 
                                                                                       input_x, 
                                                                                       mask, 
                                                                                       mask, 
                                                                                       [0, 0], 
                                                                                       reverse=False)
                    action_pred = action_pred[:, -1, :, :]

                action_idx = torch.argmax(action_pred, dim=1).squeeze().detach()
                
            else:
                out_cls, out_bound = curr_backbone(input_x)
                postprocessor = PostProcessor("refinement_with_boundary", asrf_cfg.boundary_th)
                refined_output_cls = postprocessor(out_cls.cpu().data.numpy(), boundaries=out_bound.cpu().data.numpy(), 
                                                   masks=torch.ones(1, 1, input_x.shape[-1]).bool().data.numpy())
                action_idx = torch.Tensor(refined_output_cls).squeeze().detach()

            _, predictions, _ = model(action_idx.to(device), input_x)
            _, predicted = torch.max(predictions.data, 1)
                
            predicted = predicted.squeeze()
            recognition = []
            for i in range(len(predicted)):
                recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
            f_name = vid.split('/')[-1].split('.')[0]
            f_ptr = open(result_dir + "/" + f_name, "w")
            f_ptr.write("### Frame level recognition: ###\n")
            f_ptr.write(' '.join(recognition))
            f_ptr.close()
            

def predict_backbone(name, model, model_dir, result_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
    model.eval()
    with torch.no_grad():
        model.to(device)
        model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
        file_ptr = open(vid_list_file, 'r')
        list_of_vids = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        for vid in list_of_vids:
            features = np.load(features_path + vid.split('.')[0] + '.npy')
            features = features[:, ::sample_rate]
            input_x = torch.tensor(features, dtype=torch.float)
            input_x.unsqueeze_(0)
            input_x = input_x.to(device)
            if name == 'asrf':
                out_cls, out_bound = model(input_x)
                postprocessor = PostProcessor("refinement_with_boundary", asrf_cfg.boundary_th)
                refined_output_cls = postprocessor(out_cls.cpu().data.numpy(), boundaries=out_bound.cpu().data.numpy(), 
                                                   masks=torch.ones(1, 1, input_x.shape[-1]).bool().data.numpy())
                predicted = refined_output_cls
                
            elif name == 'mstcn':
                predictions = model(input_x, torch.ones(input_x.size(), device=device))
                _, predicted = torch.max(predictions[-1].data, 1)
                
            elif name == 'sstda':
                mask = torch.ones(input_x.size(), device=device)
                predictions, _, _, _, _, _, _, _, _, _, _, _, _, _ = model(input_x, 
                                                                           input_x, 
                                                                           mask, 
                                                                           mask, 
                                                                           [0, 0], 
                                                                           reverse=False)
                _, predicted = torch.max(predictions[:, -1, :, :].data, 1)
                
            elif name == 'mgru':
                predictions = model(input_x)
                _, predicted = torch.max(predictions.data, 1)
                
                
            predicted = predicted.squeeze()
            recognition = []
            for i in range(len(predicted)):
                recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
            f_name = vid.split('/')[-1].split('.')[0]
            f_ptr = open(result_dir + "/" + f_name, "w")
            f_ptr.write("### Frame level recognition: ###\n")
            f_ptr.write(' '.join(recognition))
            f_ptr.close()
            
            
def predict_backbone_best(name, model, best_fp, result_dir, features_path, vid_list_file_tst, actions_dict, device, sample_rate):
    model.eval()
    with torch.no_grad():
        model.to(device)
        model.load_state_dict(torch.load(best_fp))
        file_ptr = open(vid_list_file_tst, 'r')
        list_of_vids = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        for vid in list_of_vids:
            features = np.load(features_path + vid.split('.')[0] + '.npy')
            features = features[:, ::sample_rate]
            input_x = torch.tensor(features, dtype=torch.float)
            input_x.unsqueeze_(0)
            input_x = input_x.to(device)
            if name == 'asrf':
                out_cls, out_bound = model(input_x)
                postprocessor = PostProcessor("refinement_with_boundary", asrf_cfg.boundary_th)
                refined_output_cls = postprocessor(out_cls.cpu().data.numpy(), boundaries=out_bound.cpu().data.numpy(), 
                                                   masks=torch.ones(1, 1, input_x.shape[-1]).bool().data.numpy())
                predicted = refined_output_cls
                
            elif name == 'mstcn':
                predictions = model(input_x, torch.ones(input_x.size(), device=device))
                _, predicted = torch.max(predictions[-1].data, 1)
                
            elif name == 'sstda':
                mask = torch.ones(input_x.size(), device=device)
                predictions, _, _, _, _, _, _, _, _, _, _, _, _, _ = model(input_x, 
                                                                           input_x, 
                                                                           mask, 
                                                                           mask, 
                                                                           [0, 0], 
                                                                           reverse=False)
                _, predicted = torch.max(predictions[:, -1, :, :].data, 1)
                
            elif name == 'mgru':
                predictions = model(input_x)
                _, predicted = torch.max(predictions.data, 1)
                
            predicted = predicted.squeeze()
            recognition = []
            for i in range(len(predicted)):
                recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
            f_name = vid.split('/')[-1].split('.')[0]
            f_ptr = open(result_dir + "/" + f_name, "w")
            f_ptr.write("### Frame level recognition: ###\n")
            f_ptr.write(' '.join(recognition))
            f_ptr.close()
