from models.basic_tester import BasicTester
import torch
import copy
import numpy as np
from os.path import join

from metrics.seg_evaluation import SegmentationEvaluation, BraTSSegEvaluation
from datasets.OASIS_dataset import OASISSegTestSinglePatientDataset
from datasets.BraTS_dataset import BraTSSegTestSinglePatientDataset
from datasets.ACDC_dataset import ACDCSegTestSinglePatientDataset
from datasets.CovidCT_dataset import CovidCTSegTestSinglePatientDataset

import segmentation_models_pytorch as smp

"""
This one is for multi-scales SR tasks evaluation.

@Jin Zhu (jin.zhu@cl.cam.ac.uk) Nov 14 2020
"""


class SegTester(BasicTester):

    def __init__(self, paras):
        super(SegTester, self).__init__(paras)

        # output dir
        self.output_dir = join(self.output_dir, 'Segmentation')

        self.name = '{}_{}'.format(self.name, paras.trained_model_mode)
        self.target_classes = paras.target_classes
        in_channels = paras.input_channel
        classes = paras.rst_classes

        self.trained_model_mode = paras.trained_model_mode
        self.model_names.append('seg_model')
        if self.trained_model_mode is 'UNet':
            self.seg_model = smp.Unet(in_channels=in_channels, classes=classes).to(self.device)
            self.ptm_paths['seg_model'] = paras.well_trained_seg_model

        # testing data
        valid_datasets = ['OASIS', 'BraTS', 'ACDC']
        data_folder = self.paras.data_folder
        self.which_data = None
        if 'OASIS' in data_folder:
            self.testing_patient_ids = self.paras.testing_patient_ids_oasis
            self.which_data = 'OASIS'
        elif 'BraTS' in data_folder:
            self.testing_patient_ids = self.paras.testing_patient_ids_brats
            self.which_data = 'BraTS'
        elif 'ACDC' in data_folder:
            self.testing_patient_ids = self.paras.testing_patient_ids_acdc
            self.which_data = 'ACDC'
        elif 'COVID' in data_folder:
            self.testing_patient_ids = self.paras.testing_patient_ids_covid
            self.which_data = 'COVID'
        else:
            # add more
            raise ValueError('Invalid data, {}, only support {}'.format(data_folder, valid_datasets))

        # evaluation
        if self.which_data in ['BraTS']:
            self.eva_func = BraTSSegEvaluation()
        else:
            self.eva_func = SegmentationEvaluation(classes=self.target_classes)

    def __inference_one__(self, sample):

        img = sample['in']
        img = self.prepare(img)
        self.seg_model.eval()
        with torch.no_grad():
            pred_segmentation = self.seg_model(img)[0]
        # one-hot to map
        pred_segmentation = torch.argmax(pred_segmentation, dim=0).unsqueeze(0)

        # tensor to numpy H x W x 1
        pred_segmentation = self.tensor_2_numpy(pred_segmentation)

        return pred_segmentation

    def modify_image_shape(self, img, s):
        int_s = np.ceil(s)
        H, W = img.shape[:2]
        return self.resize([img, [int(H//int_s*s), int(W//int_s*s)]])

    def test(self):
        all_eva_reports = []
        # inference & evaluate case-by-case
        for pid in self.testing_patient_ids:
            plog = self.fancy_print('Inference & Evaluation on case {} start @ {}'.format(pid, self.current_time()))
            self.write_log(plog)

            if self.which_data == 'OASIS':
                DS = OASISSegTestSinglePatientDataset(
                    self.paras.data_folder, pid, self.paras.gt_folder, gt_imgs=self.paras.gt_imgs
                )
            elif self.which_data == 'BraTS':
                DS = BraTSSegTestSinglePatientDataset(
                    self.paras.data_folder, pid, self.paras.gt_folder, gt_imgs=self.paras.gt_imgs
                )
            elif self.which_data == 'ACDC':
                DS = ACDCSegTestSinglePatientDataset(
                    self.paras.data_folder, pid, self.paras.gt_folder, gt_imgs=self.paras.gt_imgs
                )
            elif self.which_data == 'COVID':
                DS = CovidCTSegTestSinglePatientDataset(
                    self.paras.data_folder, pid, self.paras.gt_folder, gt_imgs=self.paras.gt_imgs
                )
            else:
                DS = None
            eva_report = self.evaluation(pid, DS)
            all_eva_reports.append(eva_report)

        all_eva_reports = self.eva_func.stack_eva_reports(all_eva_reports)

        # summary the all reports
        flag = self.fancy_print(
            'Summary evaluation performance on {} with {} cases @ {}'.format(
                self.which_data, len(self.testing_patient_ids), self.current_time()
            )
        )
        plog = flag + '\n' + 'Case IDs: {}\n'.format(self.testing_patient_ids)
        plog += self.eva_func.print(all_eva_reports)

        self.write_log(plog)

    def select_images_to_save(self, all_images):
        """
        Here we save all pred segmented labels.
        :param all_images: rec_imgs, in segmentation task, pred_labels
        :return:
        """
        return all_images

    def get_gt_images(self, samples):
        """
        In segmentation task, there is no need to return gt images
        :param samples: a list of ori_samples from the dataset
        :return gt_imgs: a dict with the same format of rec_imgs
        """
        return None


