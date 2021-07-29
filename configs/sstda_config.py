use_target = 'uSv'
num_seg = 2
use_best_model = 'none'
DA_adv = 'rev_grad' 
DA_adv_video = 'rev_grad_ssl'
use_attn = 'domain_attn'
DA_ent='attn'

iter_max_gamma = {'gtea':2000, '50salads':20000, 'breakfast':65000}
iter_max_beta = {'gtea':[2000, 1400], '50salads':[20000, 25000], 'breakfast':[65000, 50000]}
iter_max_nu = {'gtea':37500, '50salads':25000, 'breakfast':16000000}
mu = {'gtea':0.01, '50salads':0.055, 'breakfast':0.4251}
eta = {'gtea':0, '50salads':0, 'breakfast':0}
lr = {'gtea':0.0005, '50salads':0.0008, 'breakfast':0.0003}

num_stages = 4
num_layers = 10
num_f_maps = 64
features_dim = 2048
pair_ssl = 'all' # adjacent
place_adv = ['N', 'Y', 'Y', 'N']
multi_adv = ['N', 'N']
weighted_domain_loss = 'Y'
ps_lb = 'soft'
source_lb_weight = 'pseudo'
method_centroid = 'none'
DA_sem = 'mse'
place_sem = ['N', 'Y', 'Y', 'N']
ratio_ma = 0.7
place_ent = ['N', 'Y', 'Y', 'N']
DA_dis = 'none'
place_dis = ['N', 'Y', 'Y', 'N']
DA_ens = 'none'
place_ens = ['N', 'Y', 'Y', 'N']
SS_video = 'none'
place_ss = ['N', 'Y', 'Y', 'N']

split_target = '0'
ratio_source = 1
ratio_label_source = 1

bS = 1
alpha = 0.15
tau = 4
beta = [-2, -2]
gamma = -2
nu = -2
dim_proj = 128

num_epochs = 50
verbose = True
multi_gpu = False
resume_epoch = 0

use_tensorboard = False
epoch_embedding = 50
stage_embedding = -1
num_frame_video_embedding = 50

dataset_root = './dataset'
model_root = './model'
result_root ='./result'
record_root = './record'
iou_thresholds = [0.1, 0.25, 0.5]
