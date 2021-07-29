import os
import csv

boundary_th = 0.5

batch_size = 1
in_channel = 2048
iou_thresholds = [0.1, 0.25, 0.5]
learning_rate = 0.0001
max_epoch = 50
n_features = 64
n_layers = 10
n_stages = 4
n_stages_asb = 4
n_stages_brb = 4
weight_decay= 0.0001

hidden_dim = 512
num_highlevel_frames = 32
num_highlevel_samples = 64
num_stages = n_stages
num_layers = n_layers
num_f_maps = n_features
features_dim = in_channel
lr = learning_rate
num_epochs = max_epoch

gru_hidden_dim = 512
gru_layers = 3

dataset_root = './dataset'
model_root = './model'
best_root = './best_model'
result_root ='./result'
record_root = './record'
csv_dir='./backbones/asrf/csv'

num_splits = dict()
num_splits['gtea'] = 4
num_splits['50salads']=5
num_splits['breakfast']=4

dataset_names = ['gtea', '50salads', 'breakfast']
backbone_names = ['asrf', 'mstcn', 'sstda', 'mgru']
best = {bn:{dn:[] for dn in dataset_names} for bn in backbone_names}

for bn in backbone_names:
    for k, v in num_splits.items():
        record_dir = os.path.join(record_root, bn, k)
        if os.path.exists(record_dir):
            define_flag = True
            for i in range(v):
                if 'split_{}_best.csv'.format(i+1) not in os.listdir(record_dir):
                    define_flag = False
            if define_flag:
                for i in range(v):
                    record_fp = os.path.join(record_dir, 'split_{}_best.csv'.format(i+1))
                    with open(record_fp, 'r') as f:
                        reader = csv.reader(f, delimiter='\t')
                        for ri, row in enumerate(reader):
                            if ri > 0:
                                best_epoch = row[0]
                        best[bn][k].append(int(best_epoch))

