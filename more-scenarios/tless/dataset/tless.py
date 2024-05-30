import math
import os
import glob
import json
from copy import deepcopy
from dataset.transform import *

import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms


# split should be between 0 and 1 indicating percentage of labeled data
def get_datasets(root, size, split):
    datasets = ['train_pbr', 'train_primesense']
    l_set = []
    u_set = []
    for ds in datasets:
        unlabeled_txt_path = os.path.join(root, 'splits', split, f"{ds}_unlabeled.txt")
        labeled_txt_path = os.path.join(root, 'splits', split, f"{ds}_labeled.txt")

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

        if mode == 'train_l' or mode == 'train_u':
            self.ids = self.read_txt_file(txt_file)
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            self.ids = os.path.join(root, 'splits', dataset, 'val.txt')

        self.masks = self.collect_masks()

        self.scene_gt_infos = list(sorted(glob.glob(os.path.join(self.root, dataset, "*", "scene_gt_info.json"))))
        self.scene_gts = list(sorted(glob.glob(os.path.join(self.root, dataset, "*", "scene_gt.json"))))

        self.void_classes = [0]
        self.valid_classes = range(1, 31)  # classes: 30
        self.class_map = dict(zip(self.valid_classes, range(30)))

        # Cache parsed JSON files
        self.scene_gt_cache = {}
        for scene_gt_path in self.scene_gts:
            with open(scene_gt_path) as f:
                self.scene_gt_cache[scene_gt_path] = json.load(f)

    def read_txt_file(self, txt_file):
        ids = []
        with open(txt_file, 'r') as f:
            for line in f:
                scene_id, image_id = line.strip().split()
                image_ext = "jpg" if self.dataset == 'train_pbr' else "png"
                img_path = os.path.join(self.root, self.dataset, scene_id, "rgb", f"{image_id}.{image_ext}")
                ids.append(img_path)
        return sorted(ids)


    def collect_masks(self):
        masks = {}
        for img_path in self.ids:
            scene_id = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
            im_id = os.path.basename(img_path).split('.')[0]
            masks_path = os.path.join(self.root, self.dataset, scene_id, "mask_visib")
            masks_files = sorted([f for f in os.listdir(masks_path) if f.startswith(im_id) and f.endswith('.png')])
            masks[im_id] = [os.path.join(masks_path, f) for f in masks_files]
        return masks

    def __getitem__(self, idx):
        img_path = self.ids[idx]
        im_id = os.path.basename(img_path).split('.')[0]
        #im_id = img_path.split('/')[-1].split('.')[0]
        scene_id = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
        #scene_id = img_path.split('/')[-3]

        img = Image.open(img_path).convert("RGB")

        # Object ids
        scene_gt_item = self.scene_gts[int(scene_id)] if self.dataset == 'train_pbr' else self.scene_gts[
            int(scene_id) - 1]
        scene_gt = self.scene_gt_cache[scene_gt_item][str(int(im_id))]
        obj_ids = [gt['obj_id'] for gt in scene_gt]

        # we ignore non-visible objects

        # mask_visib
        masks_visib_path = self.masks[im_id]
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
