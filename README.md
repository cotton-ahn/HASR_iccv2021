# HASR_iccv2021
This is an official GitHub Repository for paper "Refining Action Segmentation with Hierarchical Video Representations", which is accepted as a regular paper (poster) in ICCV 2021.

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
git clone https://github.com/cotton-ahn/HASR_iccv2021
cd ./HASR_iccv2021
mkdir backbones
cd ./backbones
git clone https://github.com/yabufarha/ms-tcn
git clone https://github.com/cmhungsteve/SSTDA
git clone https://github.com/yiskw713/asrf
```
4. Run the script for ASRF
```
cd ..
./scripts/install_asrf.sh
```
5. Modify the script of MSTCN
* In ./backbones/ms-tcn/model.py, delete 104th line, which is "print vid"
* In ./backbones/ms-tcn/batch_gen.py, change 49th line to "length_of_sequences=list(map(len, batch_target))"

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
4. Use show_quantitative_results.ipynb to see the saved records in "./records"

## Pretrained backbone models
We release the pretrained backbone models that we have used for our experiments [Link]()

Download the "model.zip" folder, and unzip it as "model" in this workspace "HASR_iccv2021"

## Folder Structure
After you successfully prepare for training, the whole folder structure would be as follows (record, result):
```
HASR_iccv2021
  └── configs
  └── record
  │   └── asrf
  │   └── mstcn
  │   └── sstda
  │   └── mgru
  └── csv
  │   └── gtea
  │   └── 50salads
  │   └── breakfast  
  └── dataset
  │   └── gtea
  │   └── 50salads
  │   └── breakfast  
  └── scripts
  └── src
  └── model
  │   └── asrf
  │   └── mstcn
  │   └── sstda
  │   └── mgru
  └── backbones
  │   └── asrf
  │   └── ms-tcn
  │   └── SSTDA
  └── ASRF_train_evaluate.ipynb
  └── MSTCN_train_evaluate.ipynb
  └── SSTDA_train_evaluate.ipynb
  └── mGRU_train_evaluate.ipynb
  └── REFINER_train_evaluate.ipynb
  └── show_quantitative_results.ipynb
  └── LICENSE
  └── README.md
  └── requirements.txt
```
## Acknowledgements
We hugely appreciate for previous researchers in this field. Especially [MS-TCN](https://github.com/yabufarha/ms-tcn) made a huge contribution for future researchers like us!
