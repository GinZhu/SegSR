from metrics.basic_evaluation import BasicEvaluation
from tabulate import tabulate
import numpy as np

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
            self.num_classes = len(classes)
            self.metrics = [
                'dice_{}'.format(_) for _ in classes
            ]

    def __call__(self, pred_label, sample):
        """

        :param pred_label: N x H x W or H x W, tensor int
        :param sample: {'gt': tensor int N x H x W or H x W}
        :return: {'id': , 'imgs': [pred_label, gt_label], 'dice_x': ,}
        """
        gt_label = sample['gt']
        report = {'imgs': [pred_label, gt_label]}
        for l, m in enumerate(self.metrics, 1):
            gt = gt_label == l
            pred = pred_label == l
            dice = self.dice_coef(gt, pred)
            report[m] = dice
        report['id'] = sample['id']
        return report

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

    def display_images(self, rec_img, sample):
        pass
