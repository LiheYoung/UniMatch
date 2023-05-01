# UniMatch for Remote Sensing Change Detection

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisiting-weak-to-strong-consistency-in-semi/semi-supervised-change-detection-on-whu-5)](https://paperswithcode.com/sota/semi-supervised-change-detection-on-whu-5?p=revisiting-weak-to-strong-consistency-in-semi)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisiting-weak-to-strong-consistency-in-semi/semi-supervised-change-detection-on-whu-10)](https://paperswithcode.com/sota/semi-supervised-change-detection-on-whu-10?p=revisiting-weak-to-strong-consistency-in-semi)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisiting-weak-to-strong-consistency-in-semi/semi-supervised-change-detection-on-whu-20)](https://paperswithcode.com/sota/semi-supervised-change-detection-on-whu-20?p=revisiting-weak-to-strong-consistency-in-semi)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisiting-weak-to-strong-consistency-in-semi/semi-supervised-change-detection-on-whu-40)](https://paperswithcode.com/sota/semi-supervised-change-detection-on-whu-40?p=revisiting-weak-to-strong-consistency-in-semi)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisiting-weak-to-strong-consistency-in-semi/semi-supervised-change-detection-on-levir-cd)](https://paperswithcode.com/sota/semi-supervised-change-detection-on-levir-cd?p=revisiting-weak-to-strong-consistency-in-semi)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisiting-weak-to-strong-consistency-in-semi/semi-supervised-change-detection-on-levir-cd-1)](https://paperswithcode.com/sota/semi-supervised-change-detection-on-levir-cd-1?p=revisiting-weak-to-strong-consistency-in-semi)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisiting-weak-to-strong-consistency-in-semi/semi-supervised-change-detection-on-levir-cd-2)](https://paperswithcode.com/sota/semi-supervised-change-detection-on-levir-cd-2?p=revisiting-weak-to-strong-consistency-in-semi)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisiting-weak-to-strong-consistency-in-semi/semi-supervised-change-detection-on-levir-cd-3)](https://paperswithcode.com/sota/semi-supervised-change-detection-on-levir-cd-3?p=revisiting-weak-to-strong-consistency-in-semi)

We provide the official PyTorch implementation of our UniMatch in the scenario of **semi-supervised remote sensing change detection**:

> **[Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation](https://arxiv.org/abs/2208.09910)**</br>
> [Lihe Yang](https://liheyoung.github.io), [Lei Qi](http://palm.seu.edu.cn/qilei), [Litong Feng](https://scholar.google.com/citations?user=PnNAAasAAAAJ&hl=en), [Wayne Zhang](http://www.statfe.com), [Yinghuan Shi](https://cs.nju.edu.cn/shiyh/index.htm)</br>
> *In Conference on Computer Vision and Pattern Recognition (CVPR), 2023*


## Results

**You can refer to our [training logs](https://github.com/LiheYoung/UniMatch/blob/main/more-scenarios/remote-sensing/training-logs) for convenient comparisons during reproducing.**

### WHU-CD

The two numbers in each cell denote the **changed-class IoU** and **overall accuracy**, respectively.

| Method                      | 5%        | 10%       | 20%       | 40%       |
| :-------------------------: | :-------: | :-------: | :-------: | :-------: |
| [S4GAN](https://arxiv.org/abs/1908.05724)                       | 18.3 / 96.69      | 62.6 / 98.15      | 70.8 / 98.60      | 76.4 / 98.96      |
| [SemiCDNet](https://ieeexplore.ieee.org/document/9161009)                   | 51.7 / 97.71      | 62.0 / 98.16      | 66.7 / 98.28      | 75.9 / 98.93      |
| [SemiCD](https://arxiv.org/abs/2204.08454)                      | 65.8 / 98.37      | 68.1 / 98.47      | 74.8 / 98.84      | 77.2 / 98.96      |
| **UniMatch (PSPNet)**       | **77.5** / **99.06**  | **78.9** / **99.10**  | **82.9** / **99.26**  | **84.4** / **99.32**  |
| **UniMatch (DeepLabv3+)**   | **80.2** / **99.15**  | **81.7** / **99.22**  | **81.7** / **99.18**  | **85.1** / **99.35**  |


### LEVIR-CD

The two numbers in each cell denote the **changed-class IoU** and **overall accuracy**, respectively.

| Method                      | 5%        | 10%       | 20%       | 40%       |
| :-------------------------: | :-------: | :-------: | :-------: | :-------: |
| [S4GAN](https://arxiv.org/abs/1908.05724)                       | 64.0 / 97.89      | 67.0 / 98.11      | 73.4 / 98.51      | 75.4 / 98.62      |
| [SemiCDNet](https://ieeexplore.ieee.org/document/9161009)                   | 67.6 / 98.17      | 71.5 / 98.42      | 74.3 / 98.58      | 75.5 / 98.63      |
| [SemiCD](https://arxiv.org/abs/2204.08454)                      | 72.5 / 98.47      | 75.5 / 98.63      | 76.2 / 98.68      | 77.2 / 98.72      |
| **UniMatch (PSPNet)**       | **75.6** / **98.62**  | **79.0** / **98.83**  | **79.0** / **98.84**  | **78.2** / **98.79**  |
| **UniMatch (DeepLabv3+)**   | **80.7** / **98.95**  | **82.0** / **99.02**  | **81.7** / **99.02**  | **82.1** / **99.03**  |



## Getting Started

### Installation

```bash
cd UniMatch
conda create -n unimatch python=3.10.4
conda activate unimatch
pip install -r requirements.txt
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

### Pretrained Backbone

- [ResNet-50](https://drive.google.com/file/d/1mqUrqFvTQ0k5QEotk4oiOFyP6B9dVZXS/view?usp=sharing)

```
├── ./pretrained
    └── resnet50.pth
```

### Dataset

- WHU-CD: [imageA, imageB, and label](https://www.dropbox.com/s/r76a00jcxp5d3hl/WHU-CD-256.zip?dl=0)
- LEVIR-CD: [imageA, imageB, and label](https://www.dropbox.com/s/18fb5jo0npu5evm/LEVIR-CD256.zip?dl=0)

Please modify your dataset path in configuration files.

```
├── [Your WHU-CD/LEVIR-CD Path]
    ├── A
    ├── B
    └── label
```


## Usage

### UniMatch

```bash
# use torch.distributed.launch
# switch to current folder
sh scripts/train.sh <num_gpu> <port>
```

To train on other datasets or splits, please modify
``dataset`` and ``split`` in [train.sh](https://github.com/LiheYoung/UniMatch/blob/main/more-scenarios/remote-sensing/scripts/train.sh).


### Supervised Baseline

Modify the ``method`` from ``'unimatch'`` to ``'supervised'`` in [train.sh](https://github.com/LiheYoung/UniMatch/blob/main/more-scenarios/remote-sensing/scripts/train.sh), and double the ``batch_size`` in configuration file if you use the same number of GPUs as semi-supervised setting (no need to change ``lr``). 


## Citation

If you find this project useful, please consider citing:

```bibtex
@inproceedings{unimatch,
  title={Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation},
  author={Yang, Lihe and Qi, Lei and Feng, Litong and Zhang, Wayne and Shi, Yinghuan},
  booktitle={CVPR},
  year={2023}
}
```


## Acknowledgement

The processed WHU-CD and LEVIR-CD datasets are borrowed from [SemiCD](https://github.com/wgcban/SemiCD).
