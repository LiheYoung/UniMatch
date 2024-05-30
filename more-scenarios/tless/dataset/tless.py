import math
import os
import glob
import json
from copy import deepcopy
from dataset.transform import *

import torch
from torch.utils.data import Dataset, ConcatDataset, random_split
from torchvision import transforms


# split should be between 0 and 1 indicating percentage of labeled data
def get_datasets(root, size, split):
    datasets = ['train_pbr', 'train_primesense']
    l_set = []
    u_set = []
    for ds in datasets:
        if ds == "train_pbr":
            full_size_train = 50000
        elif ds == "train_primesense":
            full_size_train = 37584
        elif ds == "train_render_reconst":
            full_size_train = 76860
        else:
            raise ValueError(f'Invalid split: {ds}')

        indexes = range(full_size_train)
        l_index, u_indexes = random_split(
            dataset=indexes,
            lengths=[split, 1 - split],
            generator=torch.Generator().manual_seed(42)
        )
        l_dataset = TlessDataset(dataset=ds, root=root, mode='train_l', indexes=l_index, nsample=len(u_indexes),
                                 size=size)
        u_dataset = TlessDataset(dataset=ds, root=root, mode='train_u', indexes=u_indexes, size=size)

        l_set.append(l_dataset)
        u_set.append(u_dataset)

    labeled_dataset = ConcatDataset(l_set)
    unlabeled_dataset = ConcatDataset(u_set)
    return labeled_dataset, unlabeled_dataset


class TlessDataset(Dataset):
    def __init__(self, dataset, root, mode, indexes=None, nsample=None, size=None):
        self.dataset = dataset
        self.root = root
        self.mode = mode
        self.size = size

        if self.dataset not in ['train_pbr', 'test_primesense', 'train_primesense', 'train_render_reconst']:
            raise ValueError(f'Invalid split: {self.dataset}')

        self.ids = list(sorted(
            glob.glob(os.path.join(self.root, dataset, "*", "rgb", "*.jpg" if dataset == 'train_pbr' else "*.png"))))
        self.scene_gt_infos = list(sorted(glob.glob(os.path.join(self.root, dataset, "*", "scene_gt_info.json"))))
        self.scene_gts = list(sorted(glob.glob(os.path.join(self.root, dataset, "*", "scene_gt.json"))))

        if indexes:
            self.ids = [self.ids[i] for i in indexes]

        if mode == 'train_l' and nsample is not None:
            self.ids *= math.ceil(nsample / len(self.ids))
            self.ids = self.ids[:nsample]

        self.void_classes = [0]
        self.valid_classes = range(1, 31)  # classes: 30
        self.class_map = dict(zip(self.valid_classes, range(30)))

    def __getitem__(self, idx):
        img_path = self.ids[idx]
        im_id = img_path.split('/')[-1].split('.')[0]
        scene_id = img_path.split('/')[-3]

        img = Image.open(img_path).convert("RGB")

        # Object ids
        scene_gt_item = self.scene_gts[int(scene_id)] if self.dataset == 'train_pbr' else self.scene_gts[
            int(scene_id) - 1]
        with open(scene_gt_item) as f:
            scene_gt = json.load(f)[str(int(im_id))]
        obj_ids = [gt['obj_id'] for gt in scene_gt]

        # we ignore non-visible objects

        # mask_visib
        masks_visib_path = list(
            sorted(glob.glob(os.path.join(self.root, self.dataset, scene_id, "mask_visib", f"{im_id}_*.png"))))
        masks_visib = torch.zeros((len(masks_visib_path), img.size[1], img.size[0]), dtype=torch.uint8)
        for i, mp in enumerate(masks_visib_path):
            masks_visib[i] = torch.from_numpy(np.array(Image.open(mp).convert("L")))

        # create a single label image
        label = torch.zeros((img.size[1], img.size[0]), dtype=torch.int64)
        for i, id in enumerate(obj_ids):
            # print(id, np.sum(masks_visib[i].numpy()==255))
            label[masks_visib[i] == 255] = id

        # going back to "mask" naming from unimatch
        mask = Image.fromarray(label.numpy().astype(np.uint8))

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
