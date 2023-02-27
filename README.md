# UniMatch

This codebase contains a strong re-implementation of FixMatch in the scenario of semi-supervised semantic segmentation, as well as the official PyTorch implementation of our paper:

> **[Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation](https://arxiv.org/abs/2208.09910)**</br>
> [Lihe Yang](https://liheyoung.github.io), [Lei Qi](http://palm.seu.edu.cn/qilei), [Litong Feng](https://scholar.google.com/citations?user=PnNAAasAAAAJ&hl=en), [Wayne Zhang](http://www.statfe.com), [Yinghuan Shi](https://cs.nju.edu.cn/shiyh/index.htm)</br>
> *In Conference on Computer Vision and Pattern Recognition (CVPR), 2023*

We have another semi-supervised semantic segmentation work accepted by CVPR 2022: [ST++](https://github.com/LiheYoung/ST-PlusPlus)

> **Abstract.** 
> In this work, we revisit the weak-to-strong consistency framework, 
> popularized by FixMatch from semi-supervised classification, 
> where the prediction of a weakly perturbed image serves as supervision for its strongly perturbed version. 
> Intriguingly, we observe that such a simple pipeline already achieves competitive results against recent advanced works, 
> when transferred to our segmentation scenario. Its success heavily relies on the manual design of strong data augmentations, 
> however, which may be limited and inadequate to explore a broader perturbation space. 
> Motivated by this, we propose an auxiliary feature perturbation stream as a supplement, leading to an expanded perturbation space. 
> On the other, to sufficiently probe original image-level augmentations, we present a dual-stream perturbation technique, 
> enabling two strong views to be simultaneously guided by a common weak view. 
> Consequently, our overall Unified Dual-Stream Perturbations approach (UniMatch) surpasses all existing methods significantly across all evaluation protocols on the Pascal, Cityscapes, and COCO benchmarks. We also demonstrate the superiority of our method in remote sensing interpretation and medical image analysis.

## Results

### Pascal

Labeled images are sampled from the **original high-quality** training set. Results are obtained by DeepLabv3+ based on ResNet-101 with training size 321.

| Method                      | 1/115 (92)| 1/57 (183)| 1/28 (366)| 1/14 (732)| 1/7 (1464)  |
| :-------------------------: | :-------: | :-------: | :-------: | :-------: | :---------: |
| SupOnly                     | 45.1      | 55.3      | 64.8      | 69.7      | 73.5        |
| U<sup>2</sup>PL             | 68.0      | 69.2      | 73.7      | 76.2      | 79.5        |
| ST++                        | 65.2      | 71.0      | 74.6      | 77.3      | 79.1        |
| PS-MT                       | 65.8      | 69.6      | 76.6      | 78.4      | 80.0        |
| **UniMatch (Ours)**         | **75.2**  | **77.2**  | **78.8**  | **79.9**  | **81.2**    |


### Cityscapes

Results are obtained by DeepLabv3+ based on ResNet-50/101. We reproduce U<sup>2</sup>PL results on ResNet-50.

| ResNet-50                      | 1/30     | 1/8     | 1/4       | ResNet-101 | 1/16       | 1/8          | 1/4        |
| :-------------------------: | :-------: | :-------: | :-------: | :-------: | :---------: | :---------: | :---------: |
| SupOnly                     | 56.8      | 70.8      | 73.7      | SupOnly              | 67.9        | 73.5        | 75.4        |
| U<sup>2</sup>PL             | 59.8      | 73.0      | 76.3      | U<sup>2</sup>PL      | 74.9        | 76.5        | 78.5        |
| **UniMatch (Ours)**         | **64.5**  | **75.6**  | **77.4**  | **UniMatch (Ours)**  | **75.7**    | **77.3**    | **78.7**    |


### COCO

Results are obtained by DeepLabv3+ based on Xception-65.

| Method                      | 1/512 (232) | 1/256 (463) | 1/128 (925) | 1/64 (1849) | 1/32 (3697) |
| :-------------------------: | :-------: | :-------: | :-------: | :-------: | :---------: |
| SupOnly                     | 22.9      | 28.0      | 33.6      | 37.8      | 42.2        |
| PseudoSeg                   | 29.8      | 37.1      | 39.1      | 41.8      | 43.6        |
| PC<sup>2</sup>Seg           | 29.9      | 37.5      | 40.1      | 43.7      | 46.1        |
| **UniMatch (Ours)**         | **32.2**  | **40.4**  | **46.2**  | **48.7**  | **51.2**    |

## Getting Started

### Installation

```bash
cd UniMatch
conda create -n unimatch python=3.6.9
conda activate unimatch
pip install -r requirements.txt
pip install pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
```

### Pretrained Backbone:

[ResNet-50](https://drive.google.com/file/d/1mqUrqFvTQ0k5QEotk4oiOFyP6B9dVZXS/view?usp=sharing) | [ResNet-101](https://drive.google.com/file/d/1Rx0legsMolCWENpfvE2jUScT3ogalMO8/view?usp=sharing) | [Xception-65](https://drive.google.com/open?id=1_j_mE07tiV24xXOJw4XDze0-a0NAhNVi)

```
├── ./pretrained
    ├── resnet50.pth
    ├── resnet101.pth
    └── xception.pth
```

### Dataset:

- Pascal: [JPEGImages](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) | [SegmentationClass](https://drive.google.com/file/d/1ikrDlsai5QSf2GiSUR3f8PZUzyTubcuF/view?usp=sharing)
- Cityscapes: [leftImg8bit](https://www.cityscapes-dataset.com/file-handling/?packageID=3) | [gtFine](https://drive.google.com/file/d/1E_27g9tuHm6baBqcA7jct_jqcGA89QPm/view?usp=sharing)
- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip) | [val2017](http://images.cocodataset.org/zips/val2017.zip) | [masks](https://drive.google.com/file/d/166xLerzEEIbU7Mt1UGut-3-VN41FMUb1/view?usp=sharing)

Please modify the dataset path in configuration files.

*The groundtruth mask ids have already been pre-processed. You may use them directly.*

```
├── [Your Pascal Path]
    ├── JPEGImages
    └── SegmentationClass
    
├── [Your Cityscapes Path]
    ├── leftImg8bit
    └── gtFine
    
├── [Your COCO Path]
    ├── train2017
    ├── val2017
    └── masks
```

## Usage

### UniMatch

```bash
# use torch.distributed.launch
sh tools/train.sh <num_gpu> <port>

# or use slurm
# sh tools/slurm_train.sh <num_gpu> <port> <partition>
```

To run on different labeled data partitions or different datasets, please modify:

``config``, ``labeled_id_path``, ``unlabeled_id_path``, and ``save_path`` in [train.sh](https://github.com/LiheYoung/UniMatch/blob/main/tools/train.sh).

### FixMatch

Modify the ``unimatch.py`` to ``fixmatch.py`` in [train.sh](https://github.com/LiheYoung/UniMatch/blob/main/tools/train.sh).

### Supervised Baseline

Modify the ``unimatch.py`` to ``supervised.py`` in [train.sh](https://github.com/LiheYoung/UniMatch/blob/main/tools/train.sh), and double the ``batch_size`` in configuration file if you use the same number of GPUs as semi-supervised setting (no need to change ``lr``). 

## Citation

If you find these projects useful, please consider citing:

```bibtex
@inproceedings{unimatch,
  title={Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation},
  author={Yang, Lihe and Qi, Lei and Feng, Litong and Zhang, Wayne and Shi, Yinghuan},
  booktitle={CVPR},
  year={2023}
}

@inproceedings{st++,
  title={ST++: Make Self-training Work Better for Semi-supervised Semantic Segmentation},
  author={Yang, Lihe and Zhuo, Wei and Qi, Lei and Shi, Yinghuan and Gao, Yang},
  booktitle={CVPR},
  year={2022}
}
```

## Acknowledgement

We thank [AEL](https://github.com/hzhupku/SemiSeg-AEL), [CPS](https://github.com/charlesCXK/TorchSemiSeg), [CutMix-Seg](https://github.com/Britefury/cutmix-semisup-seg), [DeepLabv3Plus](https://github.com/YudeWang/deeplabv3plus-pytorch), [PseudoSeg](https://github.com/googleinterns/wss), [PS-MT](https://github.com/yyliu01/PS-MT), [SimpleBaseline](https://github.com/jianlong-yuan/SimpleBaseline), [U<sup>2</sup>PL](https://github.com/Haochen-Wang409/U2PL) and other relevant works (see below) for their amazing open-sourced projects!


## Semi-Supervised Semantic Segmentation Projects

- [2017 ICCV] [Semi Supervised Semantic Segmentation Using Generative Adversarial Network](https://openaccess.thecvf.com/content_ICCV_2017/papers/Souly__Semi_Supervised_ICCV_2017_paper.pdf)
- [2019 TPAMI] [Semi-Supervised Semantic Segmentation with High- and Low-level Consistency](https://arxiv.org/abs/1908.05724) [[Code](https://github.com/sud0301/semisup-semseg)]
- [2020 BMVC] [Semi-supervised semantic segmentation needs strong, varied perturbations](https://arxiv.org/abs/1906.01916) [[Code](https://github.com/Britefury/cutmix-semisup-seg)]
- [2020 CVPR] [Semi-Supervised Semantic Segmentation with Cross-Consistency Training](https://arxiv.org/abs/2003.09005) [[Code](https://github.com/yassouali/CCT)]
- [2020 CVPR] [Semi-Supervised Semantic Image Segmentation with Self-correcting Networks](https://arxiv.org/abs/1811.07073)
- [2020 ECCV] [Guided Collaborative Training for Pixel-wise Semi-Supervised Learning](https://arxiv.org/abs/2008.05258) [[Code](https://github.com/ZHKKKe/PixelSSL)]
- [2020 ECCV] [Semi-Supervised Segmentation based on Error-Correcting Supervision](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740137.pdf)
- [2021 ICLR] [PseudoSeg: Designing Pseudo Labels for Semantic Segmentation](https://arxiv.org/abs/2010.09713) [[Code](https://github.com/googleinterns/wss)]
- [2021 WACV] [ClassMix: Segmentation-Based Data Augmentation for Semi-Supervised Learning](https://arxiv.org/abs/2007.07936) [[Code](https://github.com/WilhelmT/ClassMix)]
- [2021 CVPR] [Semi-supervised Semantic Segmentation with Directional Context-aware Consistency](https://arxiv.org/abs/2106.14133) [[Code](https://github.com/dvlab-research/Context-Aware-Consistency)]
- [2021 CVPR] [Semi-Supervised Semantic Segmentation with Cross Pseudo Supervision](https://arxiv.org/abs/2106.01226) [[Code](https://github.com/charlesCXK/TorchSemiSeg)]
- [2021 ICCV] [Re-distributing Biased Pseudo Labels for Semi-supervised Semantic Segmentation: A Baseline Investigation](https://arxiv.org/abs/2107.11279) [[Code](https://github.com/CVMI-Lab/DARS)]
- [2021 ICCV] [Semi-Supervised Semantic Segmentation with Pixel-Level Contrastive Learning from a Class-wise Memory Bank](https://arxiv.org/abs/2104.13415) [[Code](https://github.com/Shathe/SemiSeg-Contrastive)]
- [2021 ICCV] [A Simple Baseline for Semi-supervised Semantic Segmentation with Strong Data Augmentation](https://arxiv.org/abs/2104.07256) [[Code](https://github.com/jianlong-yuan/SimpleBaseline)]
- [2021 ICCV] [C3-SemiSeg: Contrastive Semi-supervised Segmentation via Cross-set Learning and Dynamic Class-balancing](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhou_C3-SemiSeg_Contrastive_Semi-Supervised_Segmentation_via_Cross-Set_Learning_and_Dynamic_Class-Balancing_ICCV_2021_paper.pdf) [[Code](https://github.com/SIAAAAAA/C3-SemiSeg)]
- [2021 ICCV] [Pixel Contrastive-Consistent Semi-Supervised Semantic Segmentation](https://arxiv.org/abs/2108.09025)
- [2021 NeurIPS] [Semi-Supervised Semantic Segmentation via Adaptive Equalization Learning](https://arxiv.org/abs/2110.05474) [[Code](https://github.com/hzhupku/semiseg-ael)]
- [2022 PR] [DMT: Dynamic Mutual Training for Semi-Supervised Learning](https://arxiv.org/abs/2004.08514) [[Code](https://github.com/voldemortX/DST-CBC)]
- [2022 AAAI] [GuidedMix-Net: Semi-supervised Semantic Segmentation by Using Labeled Images as Reference](https://arxiv.org/abs/2112.14015)
- [2022 ICLR] [Bootstrapping Semantic Segmentation with Regional Contrast](https://arxiv.org/abs/2104.04465) [[Code](https://github.com/lorenmt/reco)]
- [2022 CVPR] [ST++: Make Self-training Work Better for Semi-supervised Semantic Segmentation](https://arxiv.org/abs/2106.05095) [[Code](https://github.com/LiheYoung/ST-PlusPlus)]
- [2022 CVPR] [Semi-Supervised Semantic Segmentation Using Unreliable Pseudo-Labels](https://arxiv.org/abs/2203.03884) [[Code](https://github.com/Haochen-Wang409/U2PL)]
- [2022 CVPR] [Perturbed and Strict Mean Teachers for Semi-supervised Semantic Segmentation](https://arxiv.org/abs/2111.12903) [[Code](https://github.com/yyliu01/ps-mt)]
- [2022 CVPR] [Unbiased Subclass Regularization for Semi-Supervised Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Guan_Unbiased_Subclass_Regularization_for_Semi-Supervised_Semantic_Segmentation_CVPR_2022_paper.pdf) [[Code](https://github.com/Dayan-Guan/USRN)]
- [2022 CVPR] [Semi-supervised Semantic Segmentation with Error Localization Network](https://arxiv.org/abs/2204.02078) [[Code](https://github.com/kinux98/SSL_ELN)]
- [2022 CVPR] [UCC: Uncertainty guided Cross-head Co-training for Semi-Supervised Semantic Segmentation](https://arxiv.org/abs/2205.10334)
