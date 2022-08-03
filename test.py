from utils.param_loader import ParametersLoader
from models.meta_sr_tester import MetaSRTester
import argparse

"""
Example:
    python -W ignore test.py --config-file config_files/colab_meta_seg_example.ini
"""


parser = argparse.ArgumentParser(description='Parameters')
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


tester.setup()
tester.test()
