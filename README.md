# CaraNet: Context Axial Reverse Attention Network for Small Medical Objects Segmentation
<div align=center><img src="https://github.com/AngeLouCN/CaraNet/blob/main/figures/caranet.jpg" width="1000" height="450" alt="Result"/></div>
This repository contains the implementation of a novel attention based network (CaraNet) to segment the polyp (CVC-T, CVC-ClinicDB, CVC-ColonDB, ETIS and Kvasir) and brain tumor (BraTS). The CaraNet show great overall segmentation performance (mean dice) on polyp and brain tumor, but also show great performance on small medical objects (small polyps and brain tumors) segmentation. 

**The technique report is coming soon!**

## Architecture of CaraNet
### Backbone
We use **Res2Net** as our backbone.

### Context module
We choose our CFP module as context module, and choose the dilation rate is **8**. The architecture of **CFP module** as shown in following figure:
<div align=center><img src="https://github.com/AngeLouCN/CFPNet/blob/main/figures/cfp module.png" width="800" height="300" alt="Result"/></div>

### Axial Reverse Attention
As shown in architecture of CaraNet, the Axial Reverse Attention (A-RA) module contains two routes: 1) Reverse attention; 2) Axial-attention.

## Installation & Usage
### Enviroment
- Enviroment: Python 3.6;
- Install some packages:
```
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
```
```
conda install opencv-python pillow numpy matplotlib
```
- Clone this repository
```
git clone https://github.com/AngeLouCN/CaraNet
```
### Training
  + Download the training and texting dataset from this link: [Experiment Dataset](https://drive.google.com/file/d/17Cs2JhKOKwt4usiAYJVJMnXfyZWySn3s/view?usp=sharing)
  + Change the --train_path & --test_path in Train.py
  + Run ```Train.py```
  + Testing dataset is ordered as follow:
```
|-- TestDataset
|   |-- CVC-300
|   |   |-- images
|   |   |-- masks
|   |-- CVC-ClinicDB
|   |   |-- images
|   |   |-- masks
|   |-- CVC-ColonDB
|   |   |-- images
|   |   |-- masks
|   |-- ETIS-LaribPolypDB
|   |   |-- images
|   |   |-- masks
|   |-- Kvasir
|       |-- images
|       |-- masks
```
### Testing
  + Change the data_path in Test.py
### Evaluation 
  + Change the image_root and gt_root in eval_Kvasir.py
  + You can also run the matlab code in eval fold, it contains other four measurement metrics results.
  + **Our segmentation results will coming sonn!**

## Segmentation Results
+ Polyp Segmentation Results
<div align=center><img src="https://github.com/AngeLouCN/CaraNet/blob/main/figures/polyp_seg.jpg" width="800" height="650" alt="Result"/></div>

<div align=center><img src="https://github.com/AngeLouCN/CaraNet/blob/main/figures/result_table.jpg" width="800" height="650" alt="Result"/></div>

+ Small polyp analysis

The x-axis is the proportion size (%) of polyp; y-axis is the average mean dice coefficient.

| Kvasir | CVC-ClinicDB | CVC-ColonDB | ETIS | CVC-300 |
| :---: | :---: | :---: | :---: | :---: |
|<div align=center><img src="https://github.com/AngeLouCN/CaraNet/blob/main/figures/Kvasir.jpg" width="150" height="150" alt="Result"/></div>|<div align=center><img src="https://github.com/AngeLouCN/CaraNet/blob/main/figures/ClinicDB.jpg" width="150" height="150" alt="Result"/></div>|<div align=center><img src="https://github.com/AngeLouCN/CaraNet/blob/main/figures/ColonDB.jpg" width="150" height="150" alt="Result"/></div>|<div align=center><img src="https://github.com/AngeLouCN/CaraNet/blob/main/figures/ETIS.jpg" width="150" height="150" alt="Result"/></div>|<div align=center><img src="https://github.com/AngeLouCN/CaraNet/blob/main/figures/CVC-300.jpg" width="150" height="150" alt="Result"/></div>|

+ Brain Tumor Segmentation Results

<div align=center><img src="https://github.com/AngeLouCN/CaraNet/blob/main/figures/BraTS.jpg" width="600" height="200" alt="Result"/></div>

## Citation
