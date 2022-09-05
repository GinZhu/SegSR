from datasets.basic_dataset import MIBasicTrain, MIBasicValid
from datasets.basic_dataset import SRImagePairRandomCrop, SingleImageRandomCrop
from metrics.seg_evaluation import SegmentationEvaluation


import numpy as np
import nibabel as nib

from os.path import join

from multiprocessing import Pool

"""

@Jin (jin.zhu@cl.cam.ac.uk) Aug 2021
"""


class BraTSReader(MIBasicTrain):
    """
    Loading data from the OASIS dataset for training / validation
    Image data example information:
        4 Modalities:
            1.
            2.
            3.
            4.
        Seg label:
            0 for background; (and brain)
            1 for NCR & NET
            2 for ED
            4 for ET (convert to 3 when load data)

    To pre-process:
        0. reshape to 3D
        1. select slices (remove background on three dimensions);
        2. normalise;
        3. merge to a list
    """

    def __init__(self,):
        super(BraTSReader, self).__init__()

        # data folder
        self.raw_data_folder = ''

        self.modalities = []

        self.dim = 2

        self.margin = 20

        self.toy_problem = True

        self.multi_pool = Pool(8)

        self.patient_ids = None

        self.masks = {}
        self.norm = ''
        self.norm_paras = {}
        self.img_ids = []

        self.remove_margin = None

        # labels
        self.gt_labels = []

    def loading(self):
        """
        Load data into self.hr_images
            swap axe;
            select_slice
            Crop
        Load image id to image ids
        Calculate mean/std of each patient into self.norm_paras
        :return:
        """
        # ## loading training data and validation data
        # ## Training dataset should be merged and shuffled, while validation dataset should be patient-wise
        if self.toy_problem:
            self.patient_ids = self.patient_ids[:2]
        for pid in self.patient_ids:
            image_data, label_data = self.load_data(pid)
            for img in image_data:
                self.hr_images.append(img)
            # label
            for label in label_data:
                self.gt_labels.append(label)
            # pid as image ids
            self.img_ids += [pid] * len(image_data)

        # ## crop image with margin
        self.remove_margin = SingleImageRandomCrop(0, self.margin)
        self.hr_images = self.multi_pool.map(self.remove_margin, self.hr_images)

        # ## crop label with margin
        self.gt_labels = self.multi_pool.map(self.remove_margin, self.gt_labels)

    def load_data(self, pid):
        """Load image data and label data."""
        p_folder, p_name = self.encode_pid(pid)
        pid_data = []
        pid_data_ranges = []
        for m in self.modalities:
            image_path = join(p_folder, '{}_{}.nii.gz'.format(p_name, m))
            image_data = nib.load(image_path).get_fdata()
            image_data = np.swapaxes(image_data, 0, self.dim)
            if pid not in self.masks:
                image_data, mask = self.select_slice(image_data)
                self.masks[pid] = mask
            else:
                image_data, mask = self.select_slice(image_data, mask=self.masks[pid])
            image_data, image_min, image_max = self.normalize(image_data)
            pid_data.append(image_data)
            pid_data_ranges.append([image_min, image_max])
        pid_data = np.stack(pid_data, axis=-1)
        self.norm_paras[pid] = pid_data_ranges

        # label
        label_path = join(p_folder, '{}_{}.nii.gz'.format(p_name, 'seg'))
        label_data = nib.load(label_path).get_fdata()
        label_data = np.swapaxes(label_data, 0, self.dim)
        label_data, mask = self.select_slice(label_data, mask=self.masks[pid])

        # convert label == 3 to label=4
        label_data[label_data==4] = 3

        return pid_data, label_data

    def encode_pid(self, pid):
        sub_dir = pid.split('_')[0]
        pid = pid.replace('{}_'.format(sub_dir), '')
        p_folder = join(self.raw_data_folder, sub_dir, pid)
        return p_folder, pid

    @staticmethod
    def select_slice(imgs, mask=None):
        # ## get brain slices only
        if mask is None:
            if imgs.ndim == 4:
                mask = np.sum(imgs, axis=(1, 2, 3)) > 0
            elif imgs.ndim == 3:
                mask = np.sum(imgs, axis=(1, 2)) > 0
        selected_imgs = imgs[mask]

        return selected_imgs, mask


class BraTSSegDataset(BraTSReader, MIBasicTrain, MIBasicValid):
    """
    Behaviours:
        1. generate training patches in a batch;
        2. patches in each batch have the same sr_factor;
        3. how to design validation process?
        4. how to define SR-Evaluation?
    """

    def __init__(self, paras, patient_ids=None):
        super(BraTSSegDataset, self).__init__()

        # ## data loading <- RGBReader
        self.toy_problem = paras.toy_problem
        self.dim = paras.medical_image_dim_brats
        if patient_ids is not None:
            self.patient_ids = patient_ids
        else:
            self.patient_ids = paras.training_patient_ids_brats
        self.margin = paras.margin_brats
        self.multi_pool = Pool(paras.multi_threads)
        self.raw_data_folder = paras.data_folder
        self.norm = paras.normal_inputs
        self.modalities = paras.modalities_brats

        self.patch_size = 0

        self.loading()

        # ## mean / std
        self.mean = [0.] * len(self.modalities)
        self.std = [1.] * len(self.modalities)
        if 'zero_mean' in self.norm and len(self.hr_images):
            self.mean = np.mean(self.hr_images, axis=(0, 1, 2))
        if 'unit_std' in self.norm and len(self.hr_images):
            self.std = np.std(self.hr_images, axis=(0, 1, 2))

        # ## crop function to generate patches when __getitem__
        self.random_crop = SRImagePairRandomCrop(self.patch_size, 1.)

        self.seg_classes = ['NCR&NET', 'ED', 'ET']

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





