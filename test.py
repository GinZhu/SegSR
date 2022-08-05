from utils.param_loader import ParametersLoader
from models.sota_seg_tester import SegTester
import argparse

"""
Example:
    python -W ignore test.py --config-file 
"""


parser = argparse.ArgumentParser(description='Parameters')
parser.add_argument('--config-file', type=str, required=True, metavar='CONFIG',
                    help='Path to config file.')
parser.add_argument('--gpu-id', type=int, metavar='GPU',
                    help='Which gpu to use.')
parser.add_argument('--folder', type=str, metavar='DATA-FOLDER',
                    help='data folder')

args = parser.parse_args()
# do distributed training here
config_file = args.config_file
gpu_id = args.gpu_id

paras = ParametersLoader(config_file)

# change data_folder
data_folder = args.folder
if data_folder is not None:
    paras.data_folder = '/'.join(paras.data_folder.split('/')[:-1] + [data_folder])
    paras.model_name = 'OASIS_Segmentation_{}'.format(data_folder.replace('FT_SR_OASIS_', ''))

if gpu_id is not None:
    paras.gpu_id = gpu_id
    paras.eva_gpu_id = gpu_id

tester = SegTester(paras)

tester.setup()
tester.test()

