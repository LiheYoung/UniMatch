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

from PIL import Image
import numpy as np

from dataset.semicd import SemiCDDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.pspnet import PSPNet
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
from util.dist_helper import setup_distributed


parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--out-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)



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
    imgA = unnormalize(result['imgA'].squeeze(), mean, std).permute(1, 2, 0).cpu().numpy()
    imgB = unnormalize(result['imgB'].squeeze(), mean, std).permute(1, 2, 0).cpu().numpy()

    # Convert to PIL Image
    imgA_pil = Image.fromarray((imgA * 255).astype(np.uint8))
    imgB_pil = Image.fromarray((imgB * 255).astype(np.uint8))

    mask = result['mask'].squeeze().cpu().numpy()
    pred = result['pred'].squeeze().cpu().numpy()
    
    mask[mask == 1] = 255
    pred[pred == 1] = 255

    # Convert masks to PIL Images
    mask_pil = Image.fromarray(mask.astype(np.uint8))
    pred_pil = Image.fromarray(pred.astype(np.uint8))
    
    imgA_pil.save(os.path.join(save_path, 'imgA.png'))
    imgB_pil.save(os.path.join(save_path, 'imgB.png'))
    pred_pil.save(os.path.join(save_path, 'pred.png'))
    mask_pil.save(os.path.join(save_path, 'mask.png'))


def evaluate(model, loader, cfg, out_dir):
    model.eval()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    correct_pixel = AverageMeter()
    total_pixel = AverageMeter()

    with torch.no_grad():
        for imgA, imgB, mask, id in loader:
            
            imgA = imgA.cuda()
            imgB = imgB.cuda()

            pred = model(imgA, imgB).argmax(dim=1)
            
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
            
            correct_pixel.update((pred.cpu() == mask).sum().item())
            total_pixel.update(pred.numel())
            
            iou_class = reduced_intersection.cpu().numpy() / (reduced_union.cpu().numpy() + 1e-10) * 100.0
            overall_acc = (pred.cpu() == mask).sum().item() / pred.numel() * 100.0
            unchanged_iou = iou_class[0]
            changed_iou = iou_class[1]
            
            if overall_acc != 100.0 or changed_iou > 99.0:
                if unchanged_iou < 70.0:
                    save_path = os.path.join(out_dir, "lowUnchanged", f"unchanged_{unchanged_iou:.2f}_changed_{changed_iou:.2f}_acc_{overall_acc:.2f}_{id[0]}")
                else:
                    save_path = os.path.join(out_dir, "normal", f"changed_{changed_iou:.2f}_unchanged_{unchanged_iou:.2f}_acc_{overall_acc:.2f}_{id[0]}")
                if changed_iou < 10.0:
                    save_path = os.path.join(out_dir, "veryLow", f"changed_{changed_iou:.2f}_unchanged_{unchanged_iou:.2f}_acc_{overall_acc:.2f}_{id[0]}")
                os.makedirs(save_path, exist_ok=True)
                result = {'imgA': imgA, 'imgB': imgB, 'mask': mask, 'pred': pred}
                save_segmentation_results(result, save_path)
            

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    overall_acc = correct_pixel.sum / total_pixel.sum * 100.0

    return iou_class, overall_acc


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

    model_zoo = {'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet}
    assert cfg['model'] in model_zoo.keys()
    model = model_zoo[cfg['model']](cfg)
    
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))


    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)


    valset = SemiCDDataset(cfg['dataset'], cfg['data_root'], 'val')
    
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    epoch = -1

    if os.path.exists(os.path.join(args.save_path, 'best.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'best.pth'))
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    else:
        raise ValueError('No best model found')
    

    iou_class, overall_acc = evaluate(model, valloader, cfg, args.out_path)

    if rank == 0:
        logger.info('***** Evaluation ***** >>>> Unchanged IoU: {:.2f}'.format(iou_class[0]))
        logger.info('***** Evaluation ***** >>>> Changed IoU: {:.2f}'.format(iou_class[1]))
        logger.info('***** Evaluation ***** >>>> Overall Accuracy: {:.2f}\n'.format(overall_acc))



if __name__ == '__main__':
    main()
