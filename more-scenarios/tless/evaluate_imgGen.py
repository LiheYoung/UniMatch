import argparse
import logging
import os
import pprint

import torch
import numpy as np
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.tless import TlessDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from util.classes import CLASSES
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
from util.dist_helper import setup_distributed
import torch
import numpy as np
import torch.distributed as dist
from PIL import Image

parser = argparse.ArgumentParser(
    description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--split', type=str, required=True)
parser.add_argument('--out-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)

color_map = {0: [0, 0, 0],
 1: [255, 50, 0],
 2: [255, 100, 0],
 3: [255, 150, 0],
 4: [255, 200, 0],
 5: [251, 247, 0],
 6: [208, 255, 0],
 7: [158, 255, 0],
 8: [108, 255, 0],
 9: [58, 255, 0],
 10: [7, 255, 0],
 11: [0, 255, 42],
 12: [0, 255, 92],
 13: [0, 255, 142],
 14: [0, 255, 192],
 15: [0, 255, 243],
 16: [0, 216, 255],
 17: [0, 166, 255],
 18: [0, 116, 255],
 19: [0, 66, 255],
 20: [0, 15, 255],
 21: [34, 0, 255],
 22: [84, 0, 255],
 23: [134, 0, 255],
 24: [184, 0, 255],
 25: [235, 0, 255],
 26: [255, 0, 224],
 27: [255, 0, 174],
 28: [255, 0, 124],
 29: [255, 0, 74],
 30: [255, 0, 23]}


def evaluate(model, loader, cfg, out_dir):
    model.eval()
    
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    with torch.no_grad():
        for idx, (img, mask) in enumerate(loader):
            img = img.cuda()
            pred = model(img).argmax(dim=1)
            
            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()
            reduced_target = torch.from_numpy(target).cuda()

            dist.all_reduce(reduced_intersection)
            dist.all_reduce(reduced_union)
            dist.all_reduce(reduced_target)

            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())

            iou_class = reduced_intersection.cpu().numpy() / (reduced_union.cpu().numpy() + 1e-10) * 100.0
            mIoU = 0
            count = 0
            for i in range(1, len(iou_class)):
                iou = iou_class[i]
                if iou > 0.0:
                    mIoU += iou
                    count += 1
            if count != 0:
                mIOU = mIoU / count
            save_path = os.path.join(out_dir, f"{idx}_{mIOU:.2f}")
            os.makedirs(save_path, exist_ok=True)
            result = {'image': img, 'mask': mask, 'pred': pred}
            save_segmentation_results(result, save_path)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIOU = np.mean(iou_class)

    return mIOU, iou_class

def unnormalize(tensor, mean, std):
    """
    Reverse the normalization of the image tensor.
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # Multiply by std and add mean
    return tensor

def save_segmentation_results(result, save_path):
    # Reverse the normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = unnormalize(result['image'].squeeze(), mean, std).permute(1, 2, 0).cpu().numpy()

    # Ensure the image values are in [0, 1] range for visualization
    image = np.clip(image, 0, 1)

    # Convert to PIL Image
    image_pil = Image.fromarray((image * 255).astype(np.uint8))

    mask = result['mask'].squeeze().cpu().numpy()
    pred = result['pred'].squeeze().cpu().numpy()

    # Convert mask and prediction to RGB for better visualization
    mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    pred_rgb = np.zeros((*pred.shape, 3), dtype=np.uint8)
    
    for i in range(0, 31):  # assuming 30 classes (1 to 30)
        mask_rgb[mask == i] = color_map[i]
        pred_rgb[pred == i] = color_map[i]

    # Convert masks to PIL Images
    mask_pil = Image.fromarray(mask_rgb)
    pred_pil = Image.fromarray(pred_rgb)

    image_pil.save(os.path.join(save_path, "img.png"))
    mask_pil.save(os.path.join(save_path, "mask.png"))
    pred_pil.save(os.path.join(save_path, "pred.png"))


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))

        writer = SummaryWriter(args.save_path)

        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model = DeepLabV3Plus(cfg)
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                     {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                      'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)

    valset = TlessDataset(cfg['valset'], cfg['data_root'], 'val')

    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)


    if os.path.exists(os.path.join(args.save_path, 'best.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'best.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']

        logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    else:
        raise ValueError('No best model found')

    mIoU, iou_class = evaluate(model, valloader, cfg, args.out_path)

    for (cls_idx, iou) in enumerate(iou_class):
        logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                    'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
    logger.info('***** Evaluation ***** >>>> MeanIoU: {:.2f}\n'.format(mIoU))


if __name__ == '__main__':
    main()
