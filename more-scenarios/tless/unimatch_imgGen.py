import argparse
import logging
import os
import pprint

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
import yaml

from dataset.tless import TlessDataset, get_datasets
from model.semseg.deeplabv3plus import DeepLabV3Plus
from supervised import evaluate
from util.classes import CLASSES
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed

from itertools import islice
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--split', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--out-path', type=str, required=True)

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

def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))

        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model = DeepLabV3Plus(cfg)
    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                     {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                      'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)

    criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)

    trainset_l, trainset_u = get_datasets(cfg['data_root'], cfg['crop_size'], args.split)
    valset = TlessDataset(cfg['valset'], cfg['data_root'], 'val')

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    total_iters = len(trainloader_u) * cfg['epochs']
    print("Iters per run: ", len(trainloader_u))
    previous_best = 0.0
    epoch = -1

    if os.path.exists(os.path.join(args.save_path, 'best.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'best.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']

        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    else:
        raise ValueError('No checkpoint found at %s' % os.path.join(args.save_path, 'best.pth'))
            
    itersForDebugOutput = [0, 80, 854, 991, 1660, 2074, 2463, 2482, 2643, 2952, 3030, 3558, 4689,
                           5632, 5791, 6240, 7134, 8075, 8112, 8461, 10244]

    for epoch in range(epoch + 1, epoch + 1 + cfg['epochsToRun']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_w_fp = AverageMeter()
        total_mask_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u, trainloader_u)

        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2),
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, ignore_mask_mix, _, _)) in enumerate(loader):

            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2, ignore_mask = img_u_s1.cuda(), img_u_s2.cuda(), ignore_mask.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()

            with torch.no_grad():
                model.eval()

                pred_u_w_mix = model(img_u_w_mix).detach()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
                img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]

            model.train()

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            preds, preds_fp = model(torch.cat((img_x, img_u_w)), True)
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])
            pred_u_w_fp = preds_fp[num_lb:]

            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2))).chunk(2)

            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)

            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]
            
            loss_x = criterion_l(pred_x, mask_x)

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255))
            loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()

            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255))
            loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()

            loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
            loss_u_w_fp = loss_u_w_fp * ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255))
            loss_u_w_fp = loss_u_w_fp.sum() / (ignore_mask != 255).sum().item()

            loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5) / 2.0
            
            if rank == 0:
                save_path = os.path.join(args.out_path, str(epoch), str(i))
                save_inputs(save_path, img_x, mask_x, img_u_w, img_u_s1, img_u_s2, img_u_w_mix, img_u_s1_mix, img_u_s2_mix)
                save_outputs(save_path, pred_x, pred_u_s1, mask_u_w_cutmixed1, pred_u_s2, mask_u_w_cutmixed2, pred_u_w_fp, mask_u_w)
                save_losses(save_path, loss, loss_x, loss_u_s1, loss_u_s2, loss_u_w_fp)

            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
            total_loss_w_fp.update(loss_u_w_fp.item())

            mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / \
                         (ignore_mask != 255).sum()
            total_mask_ratio.update(mask_ratio.item())

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            if (len(trainloader_u) // 8) != 0:
                if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                    logger.info(
                        'Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Loss w_fp: {:.3f}, Mask ratio: '
                        '{:.3f}'.format(i, total_loss.avg, total_loss_x.avg, total_loss_s.avg,
                                        total_loss_w_fp.avg, total_mask_ratio.avg))
                            
        eval_mode = 'original'
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg)

        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                           'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
            logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))


def unnormalize(tensor):
    """
    Reverse the normalization of the image tensor.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # Multiply by std and add mean
    tensor = tensor
    return tensor
    
def save_inputs(save_path, img_x, mask_x, img_u_w, img_u_s1, img_u_s2, img_u_w_mix, img_u_s1_mix, img_u_s2_mix):
    save_path = os.path.join(save_path, "inputs")
    save_image(save_path, img_x, "img_x")
    save_mask(save_path, mask_x, "mask_x")
    save_image(save_path, img_u_w, "img_u_w")
    save_image(save_path, img_u_s1, "img_u_s1")
    save_image(save_path, img_u_s2, "img_u_s2")
    save_image(save_path, img_u_w_mix, "img_u_w_mix")
    save_image(save_path, img_u_s1_mix, "img_u_s1_mix")
    save_image(save_path, img_u_s2_mix, "img_u_s2_mix")
    
def save_outputs(save_path, pred_x, pred_u_s1, mask_u_w_cutmixed1, pred_u_s2, mask_u_w_cutmixed2, pred_u_w_fp, mask_u_w):
    save_path = os.path.join(save_path, "outputs")
    save_pred(save_path, pred_x, "pred_x")
    save_pred(save_path, pred_u_s1, "pred_u_s1")
    save_mask(save_path, mask_u_w_cutmixed1, "mask_u_w_cutmixed1")
    save_pred(save_path, pred_u_s2, "pred_u_s2")
    save_mask(save_path, mask_u_w_cutmixed2, "mask_u_w_cutmixed2")
    save_pred(save_path, pred_u_w_fp, "pred_u_w_fp")
    save_mask(save_path, mask_u_w, "mask_u_w")
    
    
def save_image(save_path, img_batch, name):
    for j in range(img_batch.size(0)):
        img_dir = os.path.join(save_path, str(j))
        os.makedirs(img_dir, exist_ok=True)
                
        image = img_batch[j].cpu()  # Move to CPU
        image = unnormalize(image)
        image = image.permute(1, 2, 0).cpu().numpy()
        # Ensure the image values are in [0, 1] range for visualization
        image = np.clip(image, 0, 1)

        # Convert to PIL Image
        image_pil = Image.fromarray((image * 255).astype(np.uint8))    
        image_pil.save(os.path.join(img_dir, f"{name}.png"))


def save_pred(save_path, mask_batch, name):
    mask_batch = mask_batch.cpu().argmax(dim=1)
    save_mask(save_path, mask_batch, name)

    
def save_mask(save_path, mask_batch, name):
    for j in range(mask_batch.size(0)):
        img_dir = os.path.join(save_path, str(j))
        os.makedirs(img_dir, exist_ok=True)

        mask = mask_batch[j].cpu()  # Move to CPU
        mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)

        for i in range(0, 31):  # assuming 30 classes (1 to 30)
            mask_rgb[mask == i] = color_map[i]

        mask_pil = Image.fromarray(mask_rgb)
        mask_pil.save(os.path.join(img_dir, f"{name}.png"))


def save_losses(save_path, loss, loss_x, loss_u_s1, loss_u_s2, loss_u_w_fp):
    # Open the file in append mode ('a'), creating it if it doesn't exist
    filename = os.path.join(save_path, "losses.txt")
    with open(filename, 'a') as f:
        # Write the loss information to the file
        f.write(f"Total loss: {loss}\n")
        f.write(f"Loss x: {loss_x}\n")
        f.write(f"Loss u_s1: {loss_u_s1}\n")
        f.write(f"Loss u_s2: {loss_u_s2}\n")
        f.write(f"Loss u_w_fp: {loss_u_w_fp}\n")
        f.write("\n")  # Add a newline for separation between entries

if __name__ == '__main__':
    main()
