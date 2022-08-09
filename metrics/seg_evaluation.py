from metrics.basic_evaluation import BasicEvaluation
from tabulate import tabulate
import numpy as np
import torch
import torch.nn.functional as F

"""
Todo: 
    1. Add more metrics if necessary;
    2. Implement the functions;

"""


class SegmentationEvaluation(BasicEvaluation):
    """
    For segmentation loss functions:
        https://github.com/JunMa11/SegLoss/tree/master/losses_pytorch
        https://github.com/MIC-DKFZ/nnUNet **prefered
        This one could also be used as a segmentation network.
    We use dice loss here
    """

    def __init__(self, classes=None):
        super(SegmentationEvaluation, self).__init__()
        if classes is None:
            self.num_classes = 1
            self.metrics = [
                'dice',
            ]
        else:
            self.num_classes = len(classes) + 1
            self.metrics = [
                'dice_{}'.format(_) for _ in classes
            ]
            self.metrics.append('dice_total')
            self.metrics.append('dice_foreground')

    def __call__(self, pred_labels, samples):
        """

        :param pred_labels: list of H x W x 1, tensor int
        :param samples: list of {'gt': tensor int H x W x 1}
        :return: {'id': 'dice_x': ,}
        """
        reports = {}
        for m in self.metrics:
            reports[m] = []
        for pred_label, sample in zip(pred_labels, samples):
            gt_label = sample['gt']
            for l, m in enumerate(self.metrics, 1):
                if m in ['dice_total', 'dice_foreground']:
                    gtt = torch.from_numpy(gt_label)
                    gtt_one_hot = F.one_hot(gtt.to(torch.long), self.num_classes)
                    gt_np = gtt_one_hot.numpy()
                    pred_torch = torch.from_numpy(pred_label)
                    pred_torch_one_hot = F.one_hot(pred_torch.to(torch.long), self.num_classes)
                    pred_np = pred_torch_one_hot.numpy()
                    if 'total' in m:
                        dice = self.dice_coef(gt_np, pred_np)
                    elif 'foreground' in m:
                        assert gt_np.ndim == 3, 'GT_NP dim is {}'.format(gt_np.ndim)
                        dice = self.dice_coef(gt_np[:, :, 1:], pred_np[:, :, 1:])
                    reports[m].append(dice)
                else:
                    gt = gt_label == l
                    pred = pred_label == l
                    dice = self.dice_coef(gt, pred)
                    reports[m].append(dice)
        for m in reports:
            reports[m] = np.mean(reports[m])
        return reports

    @staticmethod
    def dice_coef(gt, pred, eps=1e-6):
        """
        Dice coefficient for segmentation
        gt, pred shoud be 2d numpy arrays
        :param gt: 0 for background 1 for label, H x W
        :param pred: 0 for background 1 for label, H x W
        :param eps: 1e-6 by default
        :return: score between [0, 1]
        """
        return (2*(gt * pred).sum() + eps) / (gt.sum() + pred.sum() + eps)

    def print(self, report):
        table = []
        row = ['Seg', ]
        for m in self.metrics:
            v = report[m]
            if isinstance(v, (float, int)):
                row += ['{:.4}'.format(v)]
            else:
                if isinstance(v, list) and isinstance(v[0], list):
                    v = np.concatenate(v)
                mean_v = np.mean(v)
                std_v = np.std(v)
                row += ['{:.4}({:.2})'.format(mean_v, std_v)]
        table.append(row)
        headers = ['Seg', ] + self.metrics
        plog = tabulate(table, headers=headers)
        return plog

    def save(self, reports, folder, prefix):
        pass

    def display_images(self, rec_imgs, samples):
        if isinstance(samples, dict):
            samples = [samples]
            rec_imgs = [rec_imgs]
        assert len(rec_imgs) == len(samples), 'Numbers of rec_imgs and samples should match'

        imgs = []
        for r, s in zip(rec_imgs, samples):
            imgs.append(r)
            imgs.append(s['gt'])
        imgs = [_/len(self.metrics) for _ in imgs]
        return {'Pred+GT': imgs}

