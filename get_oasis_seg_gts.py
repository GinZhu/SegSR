from datasets.OASIS_dataset import OASISSegDataset
from utils.param_loader import ParametersLoader
import argparse
from os.path import isdir, join
from os import makedirs
import numpy as np


parser = argparse.ArgumentParser(description='Parameters')
parser.add_argument('--config-file', type=str, required=True, metavar='CONFIG',
                    help='Path to config file.')
parser.add_argument('--output-dir', type=str, required=True, metavar='OUTPUT',
                    help='where to save the gts')

args = parser.parse_args()
# do distributed training here
config_file = args.config_file

# save the gts
output_dir = args.output_dir
if not isdir(output_dir):
    makedirs(output_dir)

paras = ParametersLoader(config_file)

data_folder = paras.data_folder
toy_problem = paras.toy_problem

medical_image_dim = paras.medical_image_dim_oasis
training_patient_ids = paras.training_patient_ids_oasis
validation_patient_ids = paras.validation_patient_ids_oasis
testing_patient_ids = paras.testing_patient_ids_oasis
margin = paras.margin_oasis
multi_threads = paras.multi_threads


for pid in testing_patient_ids:
    ds = OASISSegDataset(data_folder, [pid], [], medical_image_dim,
                         margin, toy_problem, multi_threads, patch_size=0)

    labels = ds.training_outputs
    np.savez(join(output_dir, '{}_gt.npz'.format(pid)), labels)
    print('GT Segmentation label of {} is saved to {}'.format(
        pid, join(output_dir, '{}_gt.npz'.format(pid))
    ))
    # images
    images = ds.training_imgs
    np.savez(join(output_dir, '{}_hrimg.npz'.format(pid)), images)

