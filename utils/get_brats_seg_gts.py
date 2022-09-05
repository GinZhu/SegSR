# from datasets.OASIS_dataset import OASISSegDataset
from datasets.BraTS_dataset import BraTSSegDataset
from utils.param_loader import ParametersLoader
import argparse
from os.path import isdir, join
from os import makedirs
import numpy as np
import nibabel as nib
from glob import glob


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

# # HGG and LGG patients
# from utils.param_loader import ParametersLoader
# import numpy as np
# import nibabel as nib
# from glob import glob
# from os.path import isdir, join
#
#
# config_file = 'config_files/dev_unet_seg_brats.ini'
# paras = ParametersLoader(config_file)

# HGG_paths = glob(join(paras.data_folder, 'HGG', '*', '*_seg.nii.gz'))
# print('HGG', len(HGG_paths))
# total_slice = 0
# threshold = 100
# for p in HGG_paths:
#     label_data = nib.load(p).get_fdata()
#     label_data = np.swapaxes(label_data, 0, 2)
#     mask = np.sum(label_data, axis=(1, 2)) > threshold
#     # print(p.split('/')[-1], np.sum(mask))
#     total_slice += np.sum(mask)
# print('HGG, tumor threshold', threshold, 'total slice', total_slice)
#
# LGG_paths = glob(join(paras.data_folder, 'LGG', '*', '*_seg.nii.gz'))
# print('LGG', len(LGG_paths))
# total_slice = 0
# threshold = 100
# for p in LGG_paths:
#     label_data = nib.load(p).get_fdata()
#     label_data = np.swapaxes(label_data, 0, 2)
#     mask = np.sum(label_data, axis=(1, 2)) > threshold
#     # print(p.split('/')[-1], np.sum(mask))
#     total_slice += np.sum(mask)
# print('LGG, tumor threshold', threshold, 'total slice', total_slice)
#
# LGG_pids = ['LGG_'+p.split('/')[-2] for p in LGG_paths]
#
# HGG_pids = ['HGG_'+p.split('/')[-2] for p in HGG_paths]

# random divide to Training (165 HGG + 60 LGG) and testing (45 HGG + 15 LGG)
# import numpy as np
# HGG_pids = ['HGG_Brats17_CBICA_AOH_1', 'HGG_Brats17_CBICA_AAP_1', 'HGG_Brats17_TCIA_171_1', 'HGG_Brats17_TCIA_309_1', 'HGG_Brats17_2013_14_1', 'HGG_Brats17_CBICA_ASA_1', 'HGG_Brats17_TCIA_186_1', 'HGG_Brats17_TCIA_430_1', 'HGG_Brats17_CBICA_AQQ_1', 'HGG_Brats17_TCIA_499_1', 'HGG_Brats17_CBICA_ASE_1', 'HGG_Brats17_TCIA_328_1', 'HGG_Brats17_CBICA_ABE_1', 'HGG_Brats17_CBICA_AOD_1', 'HGG_Brats17_TCIA_331_1', 'HGG_Brats17_CBICA_BHK_1', 'HGG_Brats17_TCIA_165_1', 'HGG_Brats17_CBICA_AQP_1', 'HGG_Brats17_CBICA_AUN_1', 'HGG_Brats17_2013_11_1', 'HGG_Brats17_TCIA_203_1', 'HGG_Brats17_TCIA_455_1', 'HGG_Brats17_CBICA_AQA_1', 'HGG_Brats17_TCIA_375_1', 'HGG_Brats17_CBICA_AUR_1', 'HGG_Brats17_TCIA_491_1', 'HGG_Brats17_CBICA_ATV_1', 'HGG_Brats17_CBICA_BHB_1', 'HGG_Brats17_TCIA_149_1', 'HGG_Brats17_CBICA_AOZ_1', 'HGG_Brats17_TCIA_300_1', 'HGG_Brats17_TCIA_479_1', 'HGG_Brats17_CBICA_AXL_1', 'HGG_Brats17_TCIA_321_1', 'HGG_Brats17_TCIA_394_1', 'HGG_Brats17_CBICA_BFP_1', 'HGG_Brats17_TCIA_222_1', 'HGG_Brats17_TCIA_118_1', 'HGG_Brats17_CBICA_ATB_1', 'HGG_Brats17_TCIA_226_1', 'HGG_Brats17_CBICA_AQV_1', 'HGG_Brats17_TCIA_105_1', 'HGG_Brats17_TCIA_147_1', 'HGG_Brats17_2013_7_1', 'HGG_Brats17_TCIA_133_1', 'HGG_Brats17_CBICA_AQJ_1', 'HGG_Brats17_CBICA_APZ_1', 'HGG_Brats17_TCIA_280_1', 'HGG_Brats17_TCIA_444_1', 'HGG_Brats17_TCIA_378_1', 'HGG_Brats17_CBICA_ALN_1', 'HGG_Brats17_2013_20_1', 'HGG_Brats17_CBICA_AUQ_1', 'HGG_Brats17_CBICA_ARZ_1', 'HGG_Brats17_2013_19_1', 'HGG_Brats17_CBICA_ABY_1', 'HGG_Brats17_TCIA_436_1', 'HGG_Brats17_2013_3_1', 'HGG_Brats17_TCIA_603_1', 'HGG_Brats17_TCIA_211_1', 'HGG_Brats17_TCIA_208_1', 'HGG_Brats17_TCIA_198_1', 'HGG_Brats17_TCIA_419_1', 'HGG_Brats17_CBICA_ASY_1', 'HGG_Brats17_CBICA_AZD_1', 'HGG_Brats17_TCIA_162_1', 'HGG_Brats17_TCIA_138_1', 'HGG_Brats17_CBICA_AVJ_1', 'HGG_Brats17_CBICA_ABO_1', 'HGG_Brats17_CBICA_BFB_1', 'HGG_Brats17_CBICA_ASV_1', 'HGG_Brats17_2013_25_1', 'HGG_Brats17_CBICA_ASH_1', 'HGG_Brats17_CBICA_ASK_1', 'HGG_Brats17_TCIA_296_1', 'HGG_Brats17_TCIA_242_1', 'HGG_Brats17_TCIA_117_1', 'HGG_Brats17_CBICA_AQN_1', 'HGG_Brats17_TCIA_361_1', 'HGG_Brats17_CBICA_AMH_1', 'HGG_Brats17_TCIA_201_1', 'HGG_Brats17_CBICA_AZH_1', 'HGG_Brats17_CBICA_ALU_1', 'HGG_Brats17_CBICA_AYW_1', 'HGG_Brats17_TCIA_409_1', 'HGG_Brats17_CBICA_ASU_1', 'HGG_Brats17_TCIA_231_1', 'HGG_Brats17_TCIA_180_1', 'HGG_Brats17_CBICA_ATP_1', 'HGG_Brats17_TCIA_448_1', 'HGG_Brats17_TCIA_473_1', 'HGG_Brats17_CBICA_AXO_1', 'HGG_Brats17_TCIA_332_1', 'HGG_Brats17_CBICA_AAG_1', 'HGG_Brats17_TCIA_167_1', 'HGG_Brats17_TCIA_498_1', 'HGG_Brats17_CBICA_AOO_1', 'HGG_Brats17_2013_18_1', 'HGG_Brats17_CBICA_BHM_1', 'HGG_Brats17_TCIA_606_1', 'HGG_Brats17_TCIA_390_1', 'HGG_Brats17_CBICA_AAB_1', 'HGG_Brats17_TCIA_121_1', 'HGG_Brats17_TCIA_135_1', 'HGG_Brats17_TCIA_319_1', 'HGG_Brats17_TCIA_221_1', 'HGG_Brats17_CBICA_AOP_1', 'HGG_Brats17_2013_22_1', 'HGG_Brats17_CBICA_ASO_1', 'HGG_Brats17_TCIA_247_1', 'HGG_Brats17_TCIA_412_1', 'HGG_Brats17_CBICA_ANP_1', 'HGG_Brats17_CBICA_ASN_1', 'HGG_Brats17_CBICA_APR_1', 'HGG_Brats17_TCIA_111_1', 'HGG_Brats17_CBICA_AXQ_1', 'HGG_Brats17_TCIA_235_1', 'HGG_Brats17_TCIA_605_1', 'HGG_Brats17_TCIA_608_1', 'HGG_Brats17_CBICA_AQR_1', 'HGG_Brats17_TCIA_478_1', 'HGG_Brats17_TCIA_218_1', 'HGG_Brats17_CBICA_AYA_1', 'HGG_Brats17_CBICA_ARW_1', 'HGG_Brats17_TCIA_425_1', 'HGG_Brats17_CBICA_ABB_1', 'HGG_Brats17_TCIA_257_1', 'HGG_Brats17_TCIA_370_1', 'HGG_Brats17_TCIA_406_1', 'HGG_Brats17_2013_13_1', 'HGG_Brats17_TCIA_283_1', 'HGG_Brats17_2013_17_1', 'HGG_Brats17_TCIA_322_1', 'HGG_Brats17_CBICA_AQO_1', 'HGG_Brats17_2013_4_1', 'HGG_Brats17_TCIA_274_1', 'HGG_Brats17_TCIA_401_1', 'HGG_Brats17_CBICA_AQY_1', 'HGG_Brats17_CBICA_AXN_1', 'HGG_Brats17_CBICA_ANG_1', 'HGG_Brats17_TCIA_437_1', 'HGG_Brats17_CBICA_ARF_1', 'HGG_Brats17_CBICA_AWI_1', 'HGG_Brats17_CBICA_APY_1', 'HGG_Brats17_TCIA_184_1', 'HGG_Brats17_TCIA_234_1', 'HGG_Brats17_TCIA_469_1', 'HGG_Brats17_2013_12_1', 'HGG_Brats17_CBICA_ABM_1', 'HGG_Brats17_CBICA_ALX_1', 'HGG_Brats17_TCIA_277_1', 'HGG_Brats17_TCIA_314_1', 'HGG_Brats17_TCIA_368_1', 'HGG_Brats17_TCIA_265_1', 'HGG_Brats17_TCIA_343_1', 'HGG_Brats17_2013_26_1', 'HGG_Brats17_TCIA_396_1', 'HGG_Brats17_TCIA_460_1', 'HGG_Brats17_CBICA_AQT_1', 'HGG_Brats17_TCIA_151_1', 'HGG_Brats17_TCIA_607_1', 'HGG_Brats17_TCIA_113_1', 'HGG_Brats17_2013_21_1', 'HGG_Brats17_CBICA_AME_1', 'HGG_Brats17_CBICA_AXJ_1', 'HGG_Brats17_TCIA_372_1', 'HGG_Brats17_CBICA_AYU_1', 'HGG_Brats17_2013_23_1', 'HGG_Brats17_TCIA_199_1', 'HGG_Brats17_CBICA_ATX_1', 'HGG_Brats17_TCIA_335_1', 'HGG_Brats17_TCIA_278_1', 'HGG_Brats17_TCIA_192_1', 'HGG_Brats17_CBICA_AWH_1', 'HGG_Brats17_CBICA_AYI_1', 'HGG_Brats17_CBICA_AQG_1', 'HGG_Brats17_TCIA_150_1', 'HGG_Brats17_CBICA_AQZ_1', 'HGG_Brats17_CBICA_AXW_1', 'HGG_Brats17_TCIA_190_1', 'HGG_Brats17_2013_5_1', 'HGG_Brats17_TCIA_205_1', 'HGG_Brats17_TCIA_374_1', 'HGG_Brats17_CBICA_AVV_1', 'HGG_Brats17_TCIA_471_1', 'HGG_Brats17_CBICA_ASG_1', 'HGG_Brats17_CBICA_ATF_1', 'HGG_Brats17_CBICA_AXM_1', 'HGG_Brats17_CBICA_ANI_1', 'HGG_Brats17_2013_27_1', 'HGG_Brats17_CBICA_ASW_1', 'HGG_Brats17_TCIA_290_1', 'HGG_Brats17_CBICA_ATD_1', 'HGG_Brats17_TCIA_131_1', 'HGG_Brats17_TCIA_474_1', 'HGG_Brats17_2013_2_1', 'HGG_Brats17_CBICA_AVG_1', 'HGG_Brats17_CBICA_AQD_1', 'HGG_Brats17_CBICA_AAL_1', 'HGG_Brats17_CBICA_AWG_1', 'HGG_Brats17_2013_10_1', 'HGG_Brats17_TCIA_377_1', 'HGG_Brats17_TCIA_179_1', 'HGG_Brats17_TCIA_338_1', 'HGG_Brats17_CBICA_ANZ_1', 'HGG_Brats17_TCIA_429_1', 'HGG_Brats17_TCIA_168_1', 'HGG_Brats17_CBICA_AQU_1', 'HGG_Brats17_TCIA_411_1', 'HGG_Brats17_CBICA_ABN_1']
# LGG_pids = ['LGG_Brats17_TCIA_266_1', 'LGG_Brats17_TCIA_480_1', 'LGG_Brats17_TCIA_466_1', 'LGG_Brats17_TCIA_451_1', 'LGG_Brats17_TCIA_310_1', 'LGG_Brats17_TCIA_312_1', 'LGG_Brats17_TCIA_462_1', 'LGG_Brats17_TCIA_261_1', 'LGG_Brats17_TCIA_130_1', 'LGG_Brats17_TCIA_393_1', 'LGG_Brats17_TCIA_428_1', 'LGG_Brats17_TCIA_298_1', 'LGG_Brats17_TCIA_408_1', 'LGG_Brats17_TCIA_346_1', 'LGG_Brats17_2013_1_1', 'LGG_Brats17_TCIA_490_1', 'LGG_Brats17_TCIA_276_1', 'LGG_Brats17_2013_28_1', 'LGG_Brats17_TCIA_141_1', 'LGG_Brats17_TCIA_325_1', 'LGG_Brats17_TCIA_633_1', 'LGG_Brats17_TCIA_642_1', 'LGG_Brats17_TCIA_449_1', 'LGG_Brats17_TCIA_442_1', 'LGG_Brats17_TCIA_493_1', 'LGG_Brats17_TCIA_413_1', 'LGG_Brats17_TCIA_628_1', 'LGG_Brats17_TCIA_637_1', 'LGG_Brats17_TCIA_623_1', 'LGG_Brats17_TCIA_254_1', 'LGG_Brats17_2013_8_1', 'LGG_Brats17_TCIA_615_1', 'LGG_Brats17_2013_29_1', 'LGG_Brats17_TCIA_282_1', 'LGG_Brats17_TCIA_175_1', 'LGG_Brats17_TCIA_624_1', 'LGG_Brats17_TCIA_640_1', 'LGG_Brats17_2013_15_1', 'LGG_Brats17_TCIA_103_1', 'LGG_Brats17_TCIA_402_1', 'LGG_Brats17_TCIA_625_1', 'LGG_Brats17_TCIA_351_1', 'LGG_Brats17_TCIA_639_1', 'LGG_Brats17_TCIA_650_1', 'LGG_Brats17_TCIA_630_1', 'LGG_Brats17_TCIA_109_1', 'LGG_Brats17_TCIA_470_1', 'LGG_Brats17_TCIA_632_1', 'LGG_Brats17_TCIA_653_1', 'LGG_Brats17_TCIA_620_1', 'LGG_Brats17_TCIA_152_1', 'LGG_Brats17_TCIA_621_1', 'LGG_Brats17_TCIA_255_1', 'LGG_Brats17_2013_0_1', 'LGG_Brats17_2013_6_1', 'LGG_Brats17_TCIA_629_1', 'LGG_Brats17_2013_9_1', 'LGG_Brats17_TCIA_241_1', 'LGG_Brats17_2013_16_1', 'LGG_Brats17_TCIA_645_1', 'LGG_Brats17_TCIA_410_1', 'LGG_Brats17_TCIA_299_1', 'LGG_Brats17_TCIA_654_1', 'LGG_Brats17_TCIA_634_1', 'LGG_Brats17_TCIA_307_1', 'LGG_Brats17_TCIA_387_1', 'LGG_Brats17_TCIA_177_1', 'LGG_Brats17_2013_24_1', 'LGG_Brats17_TCIA_420_1', 'LGG_Brats17_TCIA_249_1', 'LGG_Brats17_TCIA_101_1', 'LGG_Brats17_TCIA_618_1', 'LGG_Brats17_TCIA_202_1', 'LGG_Brats17_TCIA_644_1', 'LGG_Brats17_TCIA_330_1']
#
# np.random.shuffle(LGG_pids)
# np.random.shuffle(HGG_pids)
#
# training_pids = LGG_pids[10:32] + HGG_pids[30:108]
# testing_pids = LGG_pids[:8] + HGG_pids[:22]
# validation_pids = LGG_pids[:2] + HGG_pids[:6]

testing_patient_ids = paras.testing_patient_ids_brats

for pid in testing_patient_ids:
    ds = BraTSSegDataset(paras, patient_ids=pid)
    labels = ds.gt_labels

    np.savez(join(output_dir, '{}_gt.npz'.format(pid)), labels)
    print('GT Segmentation label ({}) of {} is saved to {}'.format(
        len(labels), pid, join(output_dir, '{}_gt.npz'.format(pid))
    ))
    # images
    images = ds.hr_images
    np.savez(join(output_dir, '{}_hrimg.npz'.format(pid)), images)
    print('{} HR images ({}) of {} is saved to {}'.format(
        len(images), images[0].shape, pid, join(output_dir, '{}_hrimg.npz'.format(pid))
    ))