import glob
import json
import logging
import math
import os

import numpy as np
import torch
from PIL import Image


class TlessmaskDatatset:
    def __init__(self, dataset, root, mode, txt_file=None, nsample=None, size=None):
        self.dataset = dataset
        self.root = root
        self.mode = mode
        self.size = size

        if self.dataset not in ['train_pbr', 'test_primesense', 'train_primesense', 'train_render_reconst']:
            raise ValueError(f'Invalid split: {self.dataset}')

        if mode == 'train_l' or mode == 'train_u':
            with open(txt_file, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open(f"splits/{dataset}.txt", 'r') as f:
                self.ids = f.read().splitlines()

        self.scene_gts = list(sorted(glob.glob(os.path.join(self.root, dataset, "*", "scene_gt.json"))))

        # Cache parsed JSON files
        self.scene_gt_cache = {}
        for scene_gt_path in self.scene_gts:
            with open(scene_gt_path) as f:
                self.scene_gt_cache[scene_gt_path] = json.load(f)

    def store_mask(self):
        logger = logging.getLogger('global')
        logger.info(f"Storing masks for {self.mode} set")
        i = 0
        for id in self.ids:
            img_path = id.split(' ')[0]
            mask_count = int(id.split(' ')[1])

            path_parts = img_path.split('/')

            scene_id = path_parts[1]
            img_id = path_parts[3].split('.')[0]

            image_path = os.path.join(self.root, img_path)
            img = Image.open(image_path).convert("RGB")

            # Object ids
            scene_gt_item = self.scene_gts[int(scene_id)] if self.dataset == 'train_pbr' else self.scene_gts[
                int(scene_id) - 1]
            scene_gt = self.scene_gt_cache[scene_gt_item][str(int(img_id))]
            obj_ids = [gt['obj_id'] for gt in scene_gt]

            # we ignore non-visible objects

            # mask_visib
            mask_paths = [os.path.join(self.root, self.dataset, scene_id, "mask_visib", f"{img_id}_{i:06}.png")
                          for i in
                          range(int(mask_count))]
            masks_visib = torch.zeros((mask_count, img.size[1], img.size[0]), dtype=torch.uint8)
            for i, mp in enumerate(mask_paths):
                masks_visib[i] = torch.from_numpy(np.array(Image.open(mp).convert("L")))

            # create a single label image
            label = torch.zeros((img.size[1], img.size[0]), dtype=torch.int64)
            for i, id in enumerate(obj_ids):
                # print(id, np.sum(masks_visib[i].numpy()==255))
                label[masks_visib[i] == 255] = id

            # going back to "mask" naming from unimatch
            mask = Image.fromarray(label.numpy().astype(np.uint8))

            path = os.path.join(self.root, "out", self.dataset, scene_id, "mask_visib", f"{img_id}.png")
            os.makedirs(os.path.dirname(path), exist_ok=True)

            mask.save(path)
            i += 1
            if i % 1000 == 0:
                logger.info("Stored %d masks" % i)

        logger.info(f"Finished storing masks for {self.mode} set, stored {i} masks")

logs = set()

def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def main():
    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    unlabeled_txt_path = f"splits/{1_2}/train_pbr_unlabeled.txt"
    labeled_txt_path = f"splits/{1_2}/train_pbr_labeled.txt"

    u_dataset = TlessmaskDatatset(dataset="train_pbr", root="/data/tless", mode='train_u', txt_file=unlabeled_txt_path, size=512)
    u_dataset.store_mask()
    l_dataset = TlessmaskDatatset(dataset="train_pbr", root="/data/tless", mode='train_l', txt_file=labeled_txt_path,
                                  nsample=len(u_dataset.ids), size=512)
    l_dataset.store_mask()

    valset = TlessmaskDatatset("test_primesense", "/data/tless", 'val')
    valset.store_mask()


if __name__ == '__main__':
    main()
