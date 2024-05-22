# Unimatch for ACDC with DLv3plus and Resnet

This repository contains an adaptation of UniMatch for the ACDC Dataset all using DeeplabV3plus and resnet as backbone.
It was created based on the UniMatch implementation with an effort to only add minimal changes.

**The method itself was not changed, it was only adapted for usage with DeeplabV3plus and resnet**

## Getting started
See [UniMatch](https://github.com/LiheYoung/UniMatch) for reference.

### Installation

```bash
cd UniMatch
# The use of a virtual environment is recommended
pip install -r requirements.txt
pip install torch torchvision torchaudio
```

### Pretrained Backbone

[ResNet-101](https://drive.google.com/file/d/1Rx0legsMolCWENpfvE2jUScT3ogalMO8/view?usp=sharing) 

```
├── ./pretrained
    └── resnet101.pth
```


### Dataset

ACDC: [image and mask](https://drive.google.com/file/d/1LOust-JfKTDsFnvaidFOcAVhkZrgW1Ac/view?usp=sharing)

Please modify your dataset path in configuration files.

```
├── [Your ACDC Path]
    └── data
```


## Usage

### UniMatch

```bash
# use torch.distributed.launch
# switch to current folder
sh scripts/train.sh <num_gpu> <port>
```

To train on other splits, please modify
and ``split`` in [train.sh](https://github.com/JohHub/UniMatch/blob/allDLv3%2BResnet/more-scenarios/medical-dlv3%2Bresnet/scripts/train.sh).

### Supervised Baseline

Modify the ``method`` from ``'unimatch'`` to ``'supervised'`` in [train.sh](https://github.com/JohHub/UniMatch/blob/allDLv3%2BResnet/more-scenarios/medical-dlv3%2Bresnet/scripts/train.sh), and double the ``batch_size`` in configuration file if you use the same number of GPUs as semi-supervised setting (no need to change ``lr``). 



## Acknowledgement

All credits regarding the method go to the creators of [UniMatch](https://github.com/LiheYoung/UniMatch). \
The processed ACDC dataset is borrowed from [SSL4MIS](https://github.com/HiLab-git/SSL4MIS).