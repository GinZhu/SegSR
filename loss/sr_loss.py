from loss.basic_loss import BasicLoss
import segmentation_models_pytorch as smp
from torch import nn


class SegLoss(BasicLoss):

    def __init__(self, paras):
        super(SegLoss, self).__init__(paras)

        for l in self.training_loss_names:
            f = SMPLoss(l)
            self.loss_components += f.loss_names
            self.loss_functions[l] = f

    def __call__(self, pred, gt, sr_scales=None):
        repo = {}
        scalars = self.training_loss_scalars[self.current_training_state]
        loss = 0.
        for n in scalars:
            s = scalars[n]
            loss_function = self.loss_functions[n]
            l, r = loss_function(pred, gt)
            for k in r:
                repo[k] = r[k]
            loss += l * s
        return loss, repo


class SMPLoss(object):
    """
    Default mode is multi-class:
        pred: N x C x H x W
        gt: N x H x W
    """

    def __init__(self, mode, type, classes):
        self.loss_names = [type]
        if type is 'Jaccard':
            self.function = smp.losses.JaccardLoss(mode, classes)
        elif type is 'Dice':
            self.function = smp.losses.DiceLoss(mode, classes)
        elif type is 'softBCE':
            self.function = smp.losses.SoftBCEWithLogitsLoss()
        elif type is 'softCE':
            self.function = smp.losses.SoftCrossEntropyLoss()

    def __call__(self, pred, gt):
        loss = self.function(pred, gt)
        return loss, {self.loss_names[0]: loss.item()}
