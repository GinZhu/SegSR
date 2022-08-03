from utils.param_loader import ParametersLoader
from models.sota_seg_trainer import SegTrainer
from datasets.OASIS_dataset import OASISSegDataset
import argparse

"""
Example:
    python -W ignore train.py --config-file config_files/colab_sota_seg_example.ini
"""


parser = argparse.ArgumentParser(description='Training Parameters')
parser.add_argument('--config-file', type=str, required=True, metavar='CONFIG',
                    help='Path to config file.')
parser.add_argument('--gpu-id', type=int, metavar='GPU',
                    help='Which gpu to use.')

args = parser.parse_args()
# do distributed training here
config_file = args.config_file
gpu_id = args.gpu_id

paras = ParametersLoader(config_file)

if gpu_id is not None:
    paras.gpu_id = gpu_id
    paras.eva_gpu_id = gpu_id


data_folder = paras.data_folder
toy_problem = paras.toy_problem
if 'OASIS' in data_folder:
    medical_image_dim = paras.medical_image_dim_oasis
    training_patient_ids = paras.training_patient_ids_oasis
    validation_patient_ids = paras.validation_patient_ids_oasis
    margin = paras.margin_oasis
    multi_threads = paras.multi_threads

    ds = OASISSegDataset(data_folder, training_patient_ids, validation_patient_ids, medical_image_dim,
                      margin, toy_problem, multi_threads, patch_size=0)
else:
    raise ValueError('Only support data: [OASIS, BraTS, ACDC, COVID]')

print('DS info:', len(ds), 'training samples, and', ds.test_len(), 'testing cases.')

# ## training
trainer = SegTrainer(paras, ds, ds)
trainer.setup()
trainer.train()

# # ## testing / inference
# for pid in paras.testing_patient_ids:
#     ds_test = OASISSRTest(paras.data_folder, pid, paras.dim, paras.sr_factor)
#     trainer.inference(ds_test, False)

