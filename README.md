# HASR
Hierarchical Action Segmentation Refiner

## Requirements
* Python >= 3.7
* pytorch => 1.0
* torchvision
* numpy
* pyYAML
* Pillow
* pandas
- Conda or VirtualEnv is recommended. To set the environment, run:
```
pip install -r requirements.txt
```


## Install
1. Download the dataset from the [SSTDA](https://github.com/cmhungsteve/SSTDA) repository, [Dataset Link Here](https://www.dropbox.com/s/kc1oyz79rr2znmh/Datasets.zip?dl=0)
2. Unzip the zip file, and re-name the './Datasets/action-segmentation' folder as "./dataset"
3. Clone git repositories for this repo and several backbone models
```
git clone https://github.com/cotton-ahn/hasr
cd ./hasr
mkdir backbones
cd ./backbones
git clone https://github.com/yabufarha/ms-tcn
git clone https://github.com/cmhungsteve/SSTDA
git clone https://github.com/yiskw713/asrf
```
4. Run the script for ASRF
```
cd ./hasr
./scripts/install_asrf.sh
```
5. Modify the script of MSTCN
* In ./backbone/ms-tcn/model.py, delete 104th line, which is "print vid"
* In ./backbone/ms-tcn/batch_gen.py, change 49th line to "length_of_sequences=list(map(len, batch_target))"

## Train
1. use (BACKBONE NAME)_train_evaluate.ipynb to train backbones first.
2. use REFINER_train_evaluate.ipynb to train the proposed refiner HASR.
3. When training refiner, specify dataset, split, backbone names to use in training (pool_backbone_name), backbone name to use in testing (main_backbone_name)
```
dataset = 'gtea'     # choose from gtea, 50salads, breakfast
split = 2            # gtea : 1~4, 50salads : 1~5, breakfast : 1~4
pool_backbone_name = ['mstcn'] # 'asrf', 'mstcn', 'sstda', 'mgru'
main_backbone_name = 'mstcn'
```
4. Use visualize_result.ipynb to see the records

## Pretrained models
* Will be relased after the publication
