from datasets.basic_dataset import MIBasicTrain, MIBasicValid
from datasets.basic_dataset import SRImagePairRandomCrop, SingleImageRandomCrop
from metrics.seg_evaluation import SegmentationEvaluation

import numpy as np
import nibabel as nib
import torch

from os.path import join
from os import makedirs
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


class OASISRawDataset(MIBasicTrain, MIBasicValid):
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

    def __init__(self, data_folder, training_patient_ids=None, validation_patient_ids=None, dim=2,
                 margin=20, toy_problem=True, multi_threads=8, norm=''):
        super(OASISRawDataset, self).__init__()

        self.raw_data_folder = data_folder
        self.image_folder = 'PROCESSED/MPRAGE/T88_111'
        self.label_folder = 'FSL_SEG'

        self.dim = dim

        self.margin = margin

        self.toy_problem = toy_problem

        self.multi_pool = Pool(multi_threads)

        # ## define training data and validation data
        if training_patient_ids is None and validation_patient_ids:
            raise ValueError('Validation list should be None as Training list')
        elif validation_patient_ids is None and training_patient_ids:
            raise ValueError('Training list should be None as Validation list')
        # ## if not specified, randomly choose training the training and validation datasets
        elif validation_patient_ids is None and training_patient_ids is None:
            all_patient_folders = glob(join(self.raw_data_folder, 'OAS1*'))
            all_patient_ids = [_.split('/')[-1] for _ in all_patient_folders]
            np.random.shuffle(all_patient_ids)
            validation_size = 5 if 5 < len(all_patient_ids) / 5 else int(len(all_patient_ids) / 5)
            training_patient_ids = all_patient_ids[validation_size:]
            validation_patient_ids = all_patient_ids[:validation_size]

        if self.toy_problem:
            self.training_patient_ids = training_patient_ids[:5]
            self.validation_patient_ids = validation_patient_ids[:1]
        else:
            self.training_patient_ids = training_patient_ids
            self.validation_patient_ids = validation_patient_ids

        self.masks = {}
        self.norm_paras = {}

        self.training_imgs = []

        # ## loading training data and validation data
        # ## Training dataset should be merged and shuffled, while validation dataset should be patient-wise
        for pid in self.training_patient_ids:
            image_path = glob(join(self.raw_data_folder, pid, self.image_folder, '*masked_gfc.img'))[0]
            image_data = nib.load(image_path).get_fdata()
            # label_path = glob(join(self.raw_data_folder, pid, self.label_folder, '*masked_gfc_fseg.img'))[0]
            # label_data = nib.load(label_path).get_fdata()
            image_data = np.swapaxes(image_data, 0, self.dim)
            image_data, mask = self.select_slice(image_data)
            self.masks[pid] = mask
            image_data, image_min, image_max = self.normalize(image_data)
            self.norm_paras[pid] = [image_min, image_max]
            for img in image_data:
                self.training_imgs.append(img)

        # ## crop image with margin
        self.crop = SingleImageRandomCrop(0, self.margin)
        self.training_imgs = self.multi_pool.map(self.crop, self.training_imgs)

        # ## make all images as zero-mean-unit-variance
        # ## note: this will be done in the model
        self.norm = norm
        self.mean = [0.]
        self.std = [1.]
        if 'zero_mean' in self.norm and len(self.training_imgs):
            self.mean = np.mean(self.training_imgs, axis=(0, 1, 2))
        if 'unit_std' in self.norm and len(self.training_imgs):
            self.std = np.std(self.training_imgs, axis=(0, 1, 2))

        # ## loading validation dataset
        self.testing_imgs = []
        for pid in self.validation_patient_ids:
            image_path = glob(join(self.raw_data_folder, pid, self.image_folder, '*masked_gfc.img'))[0]
            image_data = nib.load(image_path).get_fdata()
            # label_path = glob(join(self.raw_data_folder, pid, self.label_folder, '*masked_gfc_fseg.img'))[0]
            # label_data = nib.load(label_path).get_fdata()
            image_data = np.swapaxes(image_data, 0, self.dim)
            image_data, mask = self.select_slice(image_data)
            self.masks[pid] = mask
            image_data, image_min, image_max = self.normalize(image_data)
            self.norm_paras[pid] = [image_min, image_max]
            for img in image_data:
                self.testing_imgs.append(img)

        # ## testing id
        self.testing_img_ids = []
        for pid in self.validation_patient_ids:
            mask = self.masks[pid]
            self.testing_img_ids += [pid] * mask.sum()

        self.testing_imgs = self.multi_pool.map(self.crop, self.testing_imgs)

    @staticmethod
    def select_slice(imgs, mask=None):
        # ## get brain slices only
        if mask is None:
            mask = np.sum(imgs, axis=(1, 2, 3)) > 0
        selected_imgs = imgs[mask]

        return selected_imgs, mask

    def __len__(self):
        return len(self.training_imgs)

    def __getitem__(self, item):
        pass

    def test_len(self):
        return len(self.testing_imgs)

    def get_test_pair(self, item):
        pass


class OASISSegDataset(OASISRawDataset):
    """
    Example:
        config_file = 'config_files/colab_meta_sr_example.ini'
        from utils.param_loader import ParametersLoader
        paras = ParametersLoader(config_file)
        paras.medical_image_dim_oasis = 2
        print(paras)
        data_folder = paras.data_folder
        toy_problem = paras.toy_problem
        medical_image_dim = paras.medical_image_dim_oasis
        training_patient_ids = paras.training_patient_ids_oasis
        validation_patient_ids = paras.validation_patient_ids_oasis
        margin = paras.margin_oasis
        multi_threads = paras.multi_threads
        from datasets.OASIS_dataset import OASISSegDataset
        ds = OASISSegDataset(data_folder, training_patient_ids, validation_patient_ids, medical_image_dim,
                         margin, toy_problem, multi_threads, patch_size=96)
        print(len(ds), ds.test_len())
        # ## test train pair
        sample = ds.__getitem__(500)
        iis = sample['in']
        ois = sample['out']
        print(iis.shape, ois.shape)
        lr = ds.tensor_2_numpy(iis)
        hr = ds.tensor_2_numpy(ois)
        print(lr.shape, hr.shape)
        import matplotlib.pyplot as plt
        plt.imshow(lr[:, :, 0])
        plt.show()
        plt.imshow(hr[:, :, 0])
        plt.show()
        print(lr.max(), lr.min(), hr.max(), hr.min())
        print(len(ds.training_imgs), len(ds.training_outputs), ds.training_imgs[0].shape, ds.training_outputs[0].shape)
        # test test pair
        sample = ds.get_test_pair(90)
        iis = sample['in']
        ois = sample['gt']
        print(iis.shape, ois.shape, sample['id'])
        lr = ds.tensor_2_numpy(iis)[0]
        hr = ois
        print(lr.shape, hr.shape)
        import matplotlib.pyplot as plt
        plt.imshow(lr[:, :, 0])
        plt.show()
        plt.imshow(hr[:, :, 0])
        plt.show()
        print(lr.max(), lr.min(), hr.max(), hr.min())

        # get batch for training

    """

    def __init__(self, data_folder, training_patient_ids, validation_patient_ids, medical_image_dim=2,
                 margin=[14, 16], toy_problem=True, multi_threads=8, patch_size=0):

        super(OASISSegDataset, self).__init__(
            data_folder=data_folder, training_patient_ids=training_patient_ids,
            validation_patient_ids=validation_patient_ids, dim=medical_image_dim,
            margin=margin, toy_problem=toy_problem, multi_threads=multi_threads)

        self.seg_classes = ['gray', 'white', 'CSF']
        # loading training labels
        self.training_outputs = []
        for pid in self.training_patient_ids:
            label_path = glob(join(self.raw_data_folder, pid, self.label_folder, '*masked_gfc_fseg.img'))[0]
            label_data = nib.load(label_path).get_fdata()
            label_data = np.swapaxes(label_data, 0, self.dim)
            label_data = label_data[self.masks[pid]]
            for l in label_data:
                self.training_outputs.append(l)
        self.training_outputs = self.multi_pool.map(
            self.crop, self.training_outputs
        )

        self.testing_gts = []
        self.testing_img_ids = []
        for pid in self.validation_patient_ids:
            label_path = glob(join(self.raw_data_folder, pid, self.label_folder, '*masked_gfc_fseg.img'))[0]
            label_data = nib.load(label_path).get_fdata()
            label_data = np.swapaxes(label_data, 0, self.dim)
            label_data = label_data[self.masks[pid]]
            for l in label_data:
                self.testing_gts.append(l)
                self.testing_img_ids.append(pid)
        self.testing_gts = self.multi_pool.map(
            self.crop, self.testing_gts
        )

        # ## crop function to generate patches when __getitem__
        self.random_crop = SRImagePairRandomCrop(patch_size, 1.)
        # ## if necessary, add data augmentation here

        # ## eva function
        self.quick_eva_func = SegmentationEvaluation(self.seg_classes)
        self.final_eva_func = SegmentationEvaluation(self.seg_classes)

    def __getitem__(self, item):
        img_input = self.training_imgs[item]
        img_output = self.training_outputs[item]

        img_input, img_output = self.random_crop([img_input, img_output])

        img_input = self.numpy_2_tensor(img_input)
        img_output = self.numpy_2_tensor(img_output)

        return {'in': img_input, 'out': img_output}

    def get_test_pair(self, item):
        img_input = self.testing_imgs[item]
        img_output = self.testing_gts[item]
        img_id = self.testing_img_ids[item]

        img_input = self.numpy_2_tensor(img_input).unsqueeze(0)

        return {'in': img_input, 'gt': img_output, 'id': img_id}


class OASISSegTestSinglePatientDataset(MIBasicValid):

    def __init__(self, data_folder, patient_id):
        super(OASISSegTestSinglePatientDataset, self).__init__()
        self.pid = patient_id
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
        

    def test_len(self):
        pass

    def get_test_pair(self, item):
        img_input = self.testing_imgs[item]
        img_output = self.testing_gts[item]
        img_id = self.testing_img_ids[item]

        img_input = self.numpy_2_tensor(img_input).unsqueeze(0)

        return {'in': img_input, 'gt': img_output, 'id': img_id}





