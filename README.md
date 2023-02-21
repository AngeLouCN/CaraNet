# CaraNet: Context Axial Reverse Attention Network for Small Medical Objects Segmentation
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/caranet-context-axial-reverse-attention/medical-image-segmentation-on-cvc-clinicdb)](https://paperswithcode.com/sota/medical-image-segmentation-on-cvc-clinicdb?p=caranet-context-axial-reverse-attention)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/caranet-context-axial-reverse-attention/medical-image-segmentation-on-etis)](https://paperswithcode.com/sota/medical-image-segmentation-on-etis?p=caranet-context-axial-reverse-attention)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/caranet-context-axial-reverse-attention/medical-image-segmentation-on-kvasir-seg)](https://paperswithcode.com/sota/medical-image-segmentation-on-kvasir-seg?p=caranet-context-axial-reverse-attention)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/caranet-context-axial-reverse-attention/medical-image-segmentation-on-cvc-colondb)](https://paperswithcode.com/sota/medical-image-segmentation-on-cvc-colondb?p=caranet-context-axial-reverse-attention)

<div align=center><img src="https://github.com/AngeLouCN/CaraNet/blob/main/figures/caranet.jpg" width="1000" height="450" alt="Result"/></div>
This repository contains the implementation of a novel attention based network (CaraNet) to segment the polyp (CVC-T, CVC-ClinicDB, CVC-ColonDB, ETIS and Kvasir) and brain tumor (BraTS). The CaraNet show great overall segmentation performance (mean dice) on polyp and brain tumor, but also show great performance on small medical objects (small polyps and brain tumors) segmentation.  
  
**:fire: NEWS :fire:**
**The full paper is available:** [CaraNet](https://arxiv.org/ftp/arxiv/papers/2108/2108.07368.pdf)
**The journal version is available:** [CaraNet](https://arxiv.org/ftp/arxiv/papers/2301/2301.13366.pdf)

## Architecture of CaraNet
### Backbone
We use **Res2Net** as our backbone.

### Context module
We choose our CFP module as context module, and choose the dilation rate is **8**. For the details of CFP module you can find here: [CFPNet](https://arxiv.org/ftp/arxiv/papers/2103/2103.12212.pdf). The architecture of **CFP module** as shown in following figure:
<div align=center><img src="https://github.com/AngeLouCN/CFPNet/blob/main/figures/cfp module.png" width="800" height="300" alt="Result"/></div>

### Axial Reverse Attention
As shown in architecture of CaraNet, the Axial Reverse Attention (A-RA) module contains two routes: 1) Reverse attention; 2) Axial-attention (The code of axial attention is applied from [UACANET](https://github.com/plemeri/UACANet))

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
  + Download the training and testing dataset from this link: [Experiment Dataset](https://drive.google.com/file/d/17Cs2JhKOKwt4usiAYJVJMnXfyZWySn3s/view?usp=sharing)
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
  + You can download the segmentation maps of CaraNet from this link: [CaraNet](https://drive.google.com/drive/folders/1nk9PDDYCTyfgyztq80vzQ2f4-TwNwR7m?usp=sharing)
  + ```dice_average.m``` is to compute the averaged dice values according to sizes of objects, for small area analysis.
  
## Segmentation Results
+ Polyp Segmentation Results
<div align=center><img src="https://github.com/AngeLouCN/CaraNet/blob/main/figures/polyp_seg.jpg" width="800" height="650" alt="Result"/></div>

+ Conditions of test datasets:
<div align=center><img src="https://github.com/AngeLouCN/CaraNet/blob/main/figures/testconditions.PNG" width="600" alt="Result"/></div>

<div align=center><img src="https://github.com/AngeLouCN/CaraNet/blob/main/figures/result_table.jpg" width="800" height="650" alt="Result"/></div>

+ Small polyp analysis

The x-axis is the proportion size (%) of polyp; y-axis is the average mean dice coefficient.
<!--
| Kvasir | CVC-ClinicDB | CVC-ColonDB | ETIS | CVC-300 |
| :---: | :---: | :---: | :---: | :---: |
|<div align=center><img src="https://github.com/AngeLouCN/CaraNet/blob/main/figures/Kvasir.jpg" width="150" height="150" alt="Result"/></div>|<div align=center><img src="https://github.com/AngeLouCN/CaraNet/blob/main/figures/ClinicDB.jpg" width="150" height="150" alt="Result"/></div>|<div align=center><img src="https://github.com/AngeLouCN/CaraNet/blob/main/figures/ColonDB.jpg" width="150" height="150" alt="Result"/></div>|<div align=center><img src="https://github.com/AngeLouCN/CaraNet/blob/main/figures/ETIS.jpg" width="150" height="150" alt="Result"/></div>|<div align=center><img src="https://github.com/AngeLouCN/CaraNet/blob/main/figures/CVC-300.jpg" width="150" height="150" alt="Result"/></div>|
-->
<div align=center><img src="https://github.com/AngeLouCN/CaraNet/blob/main/figures/Kvasir.png" width="600"  alt="Result"/></div>
<div align=center><img src="https://github.com/AngeLouCN/CaraNet/blob/main/figures/ClinicDB.png" width="600"  alt="Result"/></div>
<div align=center><img src="https://github.com/AngeLouCN/CaraNet/blob/main/figures/ColonDB.png" width="600"  alt="Result"/></div>
<div align=center><img src="https://github.com/AngeLouCN/CaraNet/blob/main/figures/ETIS.png" width="600"  alt="Result"/></div>
<div align=center><img src="https://github.com/AngeLouCN/CaraNet/blob/main/figures/CVC-300.png" width="600" alt="Result"/></div>

## Brain Tumor Segmentation

+ Dataset

| BraTS input | Segmentation truth |
| :---: | :---: |
|<div align=center><img src="https://github.com/AngeLouCN/CaraNet/blob/main/figures/brain_input.gif" width="240" alt="Result"/></div>|<div align=center><img src="https://github.com/AngeLouCN/CaraNet/blob/main/figures/brain_seg.gif" width="240" alt="Result"/></div>|

+ Results
<div align=center><img src="https://github.com/AngeLouCN/CaraNet/blob/main/figures/BraTS.jpg" width="600" height="115" alt="Result"/></div>

+ Small tumor analysis

For very small areas (<1%):
<div align=left><img src="https://github.com/AngeLouCN/CaraNet/blob/main/figures/Brain_Tumor.png" width="600" alt="Result"/></div>

The difference between results of CaraNet and PraNet:
<div align=left><img src="https://github.com/AngeLouCN/CaraNet/blob/main/figures/BraTS_dice.png" width="600" alt="Result"/></div>

## Citation
**If you think our work is helpful, please cite both conference and journal version.**
```
@inproceedings{lou2021caranet,
author = {Ange Lou and Shuyue Guan and Hanseok Ko and Murray H. Loew},
title = {{CaraNet: context axial reverse attention network for segmentation of small medical objects}},
volume = {12032},
booktitle = {Medical Imaging 2022: Image Processing},
organization = {International Society for Optics and Photonics},
publisher = {SPIE},
pages = {81 -- 92},
year = {2022},
doi = {10.1117/12.2611802}}

@inproceedings{9506485,
  author={Lou, Ange and Loew, Murray},
  booktitle={2021 IEEE International Conference on Image Processing (ICIP)}, 
  title={CFPNET: Channel-Wise Feature Pyramid For Real-Time Semantic Segmentation}, 
  year={2021},
  volume={},
  number={},
  pages={1894-1898},
  doi={10.1109/ICIP42928.2021.9506485}}
  
@article{lou2023caranet,
  title={CaraNet: context axial reverse attention network for segmentation of small medical objects},
  author={Lou, Ange and Guan, Shuyue and Loew, Murray},
  journal={Journal of Medical Imaging},
  volume={10},
  number={1},
  pages={014005},
  year={2023},
  publisher={SPIE}
}
```
