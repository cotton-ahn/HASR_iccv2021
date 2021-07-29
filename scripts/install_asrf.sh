python ./backbones/asrf/utils/generate_gt_array.py --dataset_dir ./dataset
python ./backbones/asrf/utils/generate_boundary_array.py --dataset_dir ./dataset

python ./backbones/asrf/utils/make_csv_files.py --dataset_dir ./dataset

python ./backbones/asrf/utils/make_configs.py --root_dir ./result/asrf/50salads --dataset 50salads --split 1 2 3 4 5
python ./backbones/asrf/utils/make_configs.py --root_dir ./result/asrf/gtea --dataset gtea --split 1 2 3 4
python ./backbones/asrf/utils/make_configs.py --root_dir ./result/asrf/breakfast --dataset breakfast --split 1 2 3 4
