import logging
import math
import os
from copy import deepcopy
from dataset.transform import *

import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms

from util.utils import init_log


# split should be between 0 and 100 indicating percentage of labeled data
def get_datasets(root, size, split):
    datasets = ['train_pbr', 'train_primesense']
    l_set = []
    u_set = []
    if split == 1:
        for ds in datasets:
            labeled_txt_path = f"splits/{split}/{ds}_labeled.txt"
            l_dataset = TlessDataset(dataset=ds, root=root, mode='train_l', txt_file=labeled_txt_path, size=size)
            l_set.append(l_dataset)
        return ConcatDataset(l_set)

    for ds in datasets:
        unlabeled_txt_path = f"splits/{split}/{ds}_unlabeled.txt"
        labeled_txt_path = f"splits/{split}/{ds}_labeled.txt"

        u_dataset = TlessDataset(dataset=ds, root=root, mode='train_u', txt_file=unlabeled_txt_path, size=size)
        l_dataset = TlessDataset(dataset=ds, root=root, mode='train_l', txt_file=labeled_txt_path, nsample=len(u_dataset.ids), size=size)

        u_set.append(u_dataset)
        l_set.append(l_dataset)

    labeled_dataset = ConcatDataset(l_set)
    unlabeled_dataset = ConcatDataset(u_set)
    return labeled_dataset, unlabeled_dataset


class TlessDataset(Dataset):
    def __init__(self, dataset, root, mode, txt_file=None, nsample=None, size=None):
        self.dataset = dataset
        self.root = root
        self.mode = mode
        self.size = size

        if self.dataset not in ['train_pbr', 'test_primesense', 'train_primesense', 'train_render_reconst']:
            raise ValueError(f'Invalid split: {self.dataset}')

        logger = init_log('tless', logging.INFO)
        if not logger:
            logger = logging.getLogger('tless')

        logger.info(f"Reading in txt file: {txt_file}")

        if mode == 'train_l' or mode == 'train_u':
            with open(txt_file, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open(f"splits/{dataset}.txt", 'r') as f:
                self.ids = f.read().splitlines()

        logger.info(f"Finished loading txt, Number of images: {len(self.ids)}, First: {self.ids[0]}")

    def __getitem__(self, idx):
        id = self.ids[idx]
        relative_img_path = id.split(' ')[0]
        relative_mask_path = id.split(' ')[1]
        img = Image.open(os.path.join(self.root, relative_img_path)).convert("RGB")
        mask = Image.open(os.path.join(self.root, relative_mask_path)).convert("L")

        # continue with unimatch code

        if self.mode == 'val':
            img, mask = normalize(img, mask)
            return img, mask

        img, mask = resize(img, mask, (0.5, 2.0))
        ignore_value = 254 if self.mode == 'train_u' else 255
        img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)

        if self.mode == 'train_l':
            return normalize(img, mask)

        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        img_s2 = normalize(img_s2)

        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255

        return normalize(img_w), img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2

    def __len__(self):
        return len(self.ids)
