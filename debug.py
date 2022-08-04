import segmentation_models_pytorch as smp
from datasets.OASIS_dataset import OASISSegTestSinglePatientDataset, OASISSegDataset
from metrics.seg_evaluation import SegmentationEvaluation
from utils.param_loader import ParametersLoader

import torch


def tensor_2_numpy(t):
    """
    :param t: a tensor, either on GPU or CPU
    :return: a numpy array
        if t is a stack of images (t.dim() == 3 or 4), it will transpose the channels
        if not, will return t as the same shape.
    """
    if t.dim() == 3:
        return t.detach().cpu().numpy().transpose(1, 2, 0)
    elif t.dim() == 4:
        return t.detach().cpu().numpy().transpose(2, 3, 1)
    else:
        return t.detach().cpu().numpy()


def validation(paras, eva_func):
    # ## validation data folder
    data_folder = paras.data_folder
    toy_problem = paras.toy_problem
    medical_image_dim = paras.medical_image_dim_oasis
    training_patient_ids = paras.training_patient_ids_oasis
    validation_patient_ids = paras.validation_patient_ids_oasis
    margin = paras.margin_oasis
    multi_threads = paras.multi_threads

    ds = OASISSegDataset(data_folder, training_patient_ids, validation_patient_ids, medical_image_dim,
                         margin, toy_problem, multi_threads, patch_size=0)

    # ## evaluation
    sample_ids = list(range(ds.test_len()))
    preds = []
    samples = []
    for i in sample_ids:
        sample = ds.get_test_pair(i)
        img = sample['in'].to(device)

        model.eval()
        with torch.no_grad():
            pred_segmentation = model(img)[0]
        # pred is one hot like C x H x W, convert it to 1 x H x W
        pred_segmentation = torch.argmax(pred_segmentation, dim=0).unsqueeze(0)

        # tensor to numpy H x W x 1
        pred_segmentation = tensor_2_numpy(pred_segmentation)
        preds.append(pred_segmentation)
        samples.append(sample)
    repo = eva_func(preds, samples)
    print(eva_func.print(repo))


config_file = 'config_files/test/dev_meta_seg_oasis_gt.ini'

paras = ParametersLoader(config_file)

# model
device = torch.device('cuda:{}'.format(paras.gpu_id))
target_classes = paras.target_classes
in_channels = paras.input_channel
classes = paras.rst_classes
model = smp.Unet(in_channels=in_channels, classes=classes).to(device)
ptm_path = paras.well_trained_seg_model
ptm = torch.load(ptm_path, map_location=device)

# evaluation
eva_func = SegmentationEvaluation(classes=target_classes)

print('Raw model results with validaiton dataset: ')
validation(paras, eva_func)

print('Trained model results with validation dataset: ')
model.load_state_dict(ptm)
validation(paras, eva_func)




