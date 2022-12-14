from datasets.basic_dataset import MIBasicTrain, MIBasicValid, MedicalImageBasicDataset
from datasets.basic_dataset import SRImagePairRandomCrop, CentreCrop
from metrics.seg_evaluation import SegmentationEvaluation

import numpy as np
import nibabel as nib
import torch
from os.path import join
from os import listdir
from glob import glob
import copy

from multiprocessing import Pool

"""
Dataset for medical image segmentation:
    1. loading data with nib from the .nii / .mnc / ... file
    2. feed patch (image and label) to networks
    3. validation dataset:
    4. post processing
    
Dataset for medical image super-resolution:
    1. loading data with nib from the .mnc file
    2. feed patch to 
    
Training / Validation / Testing (GT + SR results)

Under working ... :
    OASISSRTest -> patient wise dataset for inference
    OASISMetaSRDataset 
    OASISMetaSRTest -> patient wise dataset for inference
    OASISSegTest -> patient wise dataset for inference

Test passed:
    OASISRawDataset
    OASISSegDataset
    OASISSRDataset

Todo: @ Oct 23 2020
    1. re-organise
    2. re-test the datasets
    
@Jin (jin.zhu@cl.cam.ac.uk) Aug 11 2020
"""


class CovidCTReader(MIBasicTrain):
    """
    Loading data from the OASIS dataset for training / validation
    Image data example information:
        OAS1_0041_MR1 (176, 208, 176, 1) 3046.0 0.0 231.66056320277733
    To pre-process:
        0. reshape to 3D
        1. select slices (remove background on three dimensions);
        2. normalise;
        3. merge to a list
    """

    def __init__(self, ):
        super(CovidCTReader, self).__init__()

        self.raw_data_folder = ''

        self.image_path_template = '{}.nii.gz'
        self.label_path_template = 'mask/{}.nii.gz'

        self.dim = 2
        self.centre_crop_size = 512
        self.centre_crop = None

        self.toy_problem = True

        self.multi_pool = Pool(8)

        self.patient_ids = None

        self.masks = {}
        self.norm = ''
        self.norm_paras = {}
        self.img_ids = []

        # labels
        self.hr_images = []
        self.gt_labels = []

    def loading(self):

        if self.toy_problem:
            self.patient_ids = self.patient_ids[:2]
        for pid in self.patient_ids:
            image_data, labels = self.load_data(pid)
            for img in image_data:
                self.hr_images.append(img)
            for l in labels:
                self.gt_labels.append(l)
            # pid as image ids
            self.img_ids += [pid] * len(image_data)

        # ## crop image with margin
        self.centre_crop = CentreCrop(self.centre_crop_size)
        self.hr_images = self.multi_pool.map(self.centre_crop, self.hr_images)
        self.gt_labels = self.multi_pool.map(self.centre_crop, self.gt_labels)

    def load_data(self, pid):

        image_data = nib.load(
            join(self.raw_data_folder, self.image_path_template.format(pid))
        ).get_fdata()
        label_data = nib.load(
            join(self.raw_data_folder, self.label_path_template.format(pid))
        ).get_fdata()

        image_data = np.swapaxes(image_data, 0, self.dim)
        label_data = np.swapaxes(label_data, 0, self.dim)

        label_data, mask = self.select_slice(label_data, threshold=100)
        image_data, mask = self.select_slice(image_data, mask)

        image_data, image_min, image_max = self.normalize(image_data)
        self.norm_paras[pid] = [image_min, image_max]

        if image_data.ndim == 3:
            image_data = image_data[:, :, :, np.newaxis]
        if label_data.ndim == 3:
            label_data = label_data[:, :, :, np.newaxis]

        return image_data, label_data

    @staticmethod
    def select_slice(imgs, mask=None, threshold=100):
        # ## get brain slices only
        if mask is None:
            if imgs.ndim == 4:
                mask = np.sum(imgs, axis=(1, 2, 3)) > threshold
            elif imgs.ndim == 3:
                mask = np.sum(imgs, axis=(1, 2)) > threshold
        selected_imgs = imgs[mask]

        return selected_imgs, mask


class CovidCTSegDataset(CovidCTReader, MIBasicTrain, MIBasicValid):
    """
    Behaviours:
        1. generate training patches in a batch;
        2. patches in each batch have the same sr_factor;
        3. how to design validation process?
        4. how to define SR-Evaluation?
    """

    def __init__(self, paras, patient_ids=None):
        super(CovidCTSegDataset, self).__init__()

        self.raw_data_folder = paras.data_folder
        self.toy_problem = paras.toy_problem
        self.dim = paras.medical_image_dim_covid
        if patient_ids is not None:
            self.patient_ids = patient_ids
        else:
            self.patient_ids = paras.training_patient_ids_covid
        self.centre_crop_size = paras.crop_size_covid
        
        self.norm = paras.normal_inputs

        self.patch_size = 0

        self.loading()

        self.mean = [0.] 
        self.std = [1.] 
        if 'zero_mean' in self.norm and len(self.hr_images):
            self.mean = np.mean(self.hr_images, axis=(0, 1, 2))
        if 'unit_std' in self.norm and len(self.hr_images):
            self.std = np.std(self.hr_images, axis=(0, 1, 2))

        # ## crop function to generate patches when __getitem__
        self.random_crop = SRImagePairRandomCrop(self.patch_size, 1.)

        # ## 1. RV: right ventricular cavity;
        # ## 2. MC: Myocardium
        # ## 3. LV: left ventricular cavity
        self.seg_classes = ['LL', 'RL', 'Lesion']

        # ## eva function
        self.quick_eva_func = SegmentationEvaluation(self.seg_classes)
        self.final_eva_func = SegmentationEvaluation(self.seg_classes)

    def __getitem__(self, item):
        img_input = self.hr_images[item]
        img_output = self.gt_labels[item]

        img_input, img_output = self.random_crop([img_input, img_output])

        img_input = self.numpy_2_tensor(img_input)
        img_output = self.numpy_2_tensor(img_output)

        return {'in': img_input, 'out': img_output}

    def __len__(self):
        return len(self.hr_images)

    def test_len(self):
        return len(self.hr_images)

    def get_test_pair(self, item):
        img_input = self.hr_images[item]
        img_output = self.gt_labels[item]
        img_id = self.img_ids[item]

        img_input = self.numpy_2_tensor(img_input).unsqueeze(0)
        return {'in': img_input, 'gt': img_output, 'id': img_id}


class CovidCTSegTestSinglePatientDataset(MIBasicValid):

    def __init__(self, data_folder, patient_id, gt_folder, gt_imgs=False):
        super(CovidCTSegTestSinglePatientDataset, self).__init__()
        self.pid = patient_id
        if gt_imgs:
            gt_imgs_data = np.load(join(gt_folder, '{}_hrimg.npz'.format(patient_id)))
            self.testing_imgs = gt_imgs_data[gt_imgs_data.files[0]]
        else:
            # load SR results
            data_path = join(data_folder, 'inferences', '{}_inference_results.tar'.format(patient_id))
            rec_imgs = torch.load(data_path)['rec_imgs']
            imgs = []
            for img in rec_imgs:
                for sr in img:
                    imgs.append(img[sr])
            self.testing_imgs = imgs
        self.testing_img_ids = [patient_id, ] * len(self.testing_imgs)
        # load GT labels
        gt_data = np.load(join(gt_folder, '{}_gt.npz'.format(patient_id)))
        self.testing_gts = gt_data[gt_data.files[0]]

    def test_len(self):
        return len(self.testing_imgs)

    def get_test_pair(self, item):
        img_input = self.testing_imgs[item]
        img_output = self.testing_gts[item]
        img_id = self.testing_img_ids[item]

        img_input = self.numpy_2_tensor(img_input).unsqueeze(0)

        return {'in': img_input, 'gt': img_output, 'id': img_id}


