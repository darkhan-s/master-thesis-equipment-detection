import os
import torch
import argparse
from detectron2.config import get_cfg
from adapteacher import add_ateacher_config
import detectron2.utils.comm as comm
import pprint
from collections import OrderedDict
# hacky way to register
from adapteacher.engine.trainer import ATeacherTrainer
from adapteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from adapteacher.modeling.meta_arch.rcnn import DAobjTwoStagePseudoLabGeneralizedRCNN
# from adapteacher.modeling.meta_arch.vgg import build_vgg_backbone  # noqa
from adapteacher.modeling.proposal_generator.rpn import PseudoLabRPN
from adapteacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
import adapteacher.data.datasets.builtin
from detectron2.engine import DefaultTrainer
from torch.nn.parallel import DistributedDataParallel


def removekey(d, listofkeys):
    r = dict(d)
    for key in listofkeys:
        r.pop(key)
        print('key: {} is removed'.format(key))
    return r
 
def parse():
    
    parser = argparse.ArgumentParser(description="Trim Detection weights and save in PyTorch format.")
    # default is my custom model with 2 extra components, better scheduler and without augmentations
    parser.add_argument(
        "--pretrained_path",
        default="/scratch/project_2005695/master-thesis-equipment-detection/misc/adaptive_teacher/output-mymodel-classes-1-20-constLoss-v6.1/model_best.pth",
        help="path to detectron pretrained weight(.pth)",
        type=str,
    )
    parser.add_argument(
        "--save_path",
        default="/scratch/project_2005695/master-thesis-equipment-detection/misc/adaptive_teacher/output-mymodel-classes-1-20-constLoss-v6.1",
        help="path to save the converted model",
        type=str,
    )
    parser.add_argument(
        "--config_file",
        default="/scratch/project_2005695/master-thesis-equipment-detection/misc/adaptive_teacher/configs/faster_rcnn_R101_cross_tless_incremental.yaml",
        help="path to config file",
        type=str,
    )
    # parser.add_argument(
    #     "--out_classes_total",
    #     default=20,
    #     help="total output features (classes) to detect",
    #     type=int,
    # )
    parser.add_argument(
        "--out_classes_append_num",
        default=0,
        help="new number of output features (classes) to append",
        type=int,
    )
    return parser.parse_args()

args = parse()
NUM_CLASS_TO_APPEND = args.out_classes_append_num
DETECTRON_PATH = args.pretrained_path
print('Pretrained model path: {}'.format(DETECTRON_PATH))

def save_model_without_states(full_model, save=False):
    
    del full_model["optimizer"]
    del full_model["scheduler"]
    del full_model["iteration"]
    if save:
        filename_wo_ext, ext = os.path.splitext(DETECTRON_PATH)
        output_file = filename_wo_ext + "_wo_solver_states" + ext
        torch.save(full_model, output_file)
        print("Done. The model without solver states is saved to {}".format(output_file))
    return full_model

def save_new_model(dict, args):
## Save the updated model
    try:
        os.makedirs(args.save_path)
    except FileExistsError:
        # directory already exists
        pass
    torch.save(dict, os.path.join(args.save_path, 'model_extended_best.pth'))
    print('Saved to {}.'.format(args.save_path))

def build_model():

    # build model the same way as in the training loop
    cfg = get_cfg()
    add_ateacher_config(cfg)
    cfg.merge_from_file(args.config_file)
    # cfg.freeze()

    cfg.MODEL.WEIGHTS = DETECTRON_PATH # Set path model .pth
    #cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.out_classes_total
    cfg.MODEL.DEVICE ='cpu'
    # cfg.MODEL.ROI_HEADS.NEW_CLASSES = ["Model 1"]

    model = ATeacherTrainer.build_model(cfg)
    model_teacher = ATeacherTrainer.build_model(cfg)

    # original saved file
    checkpoint = torch.load(DETECTRON_PATH, map_location=torch.device('cpu'))

    cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

    # For training, wrap with DDP. But don't need this for inference.
    if comm.get_world_size() > 1:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )


    ensem_ts_model = EnsembleTSModel(model_teacher, model)

    return checkpoint,ensem_ts_model

def edit_params(checkpoint):

    newdict = dict()
    keysToModify = ['modelStudent.roi_heads.box_predictor.cls_score.weight', 
                    'modelStudent.roi_heads.box_predictor.cls_score.bias',
                    'modelTeacher.roi_heads.box_predictor.cls_score.weight', 
                    'modelTeacher.roi_heads.box_predictor.cls_score.bias',
                    'modelStudent.roi_heads.box_predictor.bbox_pred.weight', 
                    'modelStudent.roi_heads.box_predictor.bbox_pred.bias',  
                    'modelTeacher.roi_heads.box_predictor.bbox_pred.weight', 
                    'modelTeacher.roi_heads.box_predictor.bbox_pred.bias',]

    # removing last layer of the network
    #for str in "modelStudent.", "modelTeacher.":
        #newdict = removekey(checkpoint["model"],[f'{str}roi_heads.box_predictor.cls_score.weight', f'{str}roi_heads.box_predictor.cls_score.bias', f'{str}roi_heads.box_predictor.bbox_pred.weight', f'{str}roi_heads.box_predictor.bbox_pred.bias'])
        # for k,v in checkpoint["model"].items():
        #     if k == f'{str}roi_heads.box_predictor.cls_score.weight':
        #         print(f'{str}roi_heads.box_predictor.cls_score.weight: {v.shape}')

    if NUM_CLASS_TO_APPEND == 0:
        return checkpoint
    ones_weight = torch.rand(NUM_CLASS_TO_APPEND,1024)
    ones_bias = torch.rand(NUM_CLASS_TO_APPEND)
    ones_bias_box = torch.rand(NUM_CLASS_TO_APPEND*4)
    ones_bias_box_weight = torch.rand(NUM_CLASS_TO_APPEND*4, 1024)

    ## extending the last layer
    for key in keysToModify:
        for k,v in checkpoint["model"].items():
            if k == key and key.split('.')[-2] == 'cls_score':
                if key.split('.')[-1] == 'weight':
                    checkpoint["model"][k] = torch.cat((v,ones_weight), 0)
                elif key.split('.')[-1] == 'bias': 
                    checkpoint["model"][k] = torch.cat((v,ones_bias), 0)
            if k == key and key.split('.')[-2] == 'bbox_pred':
                if key.split('.')[-1] == 'weight':
                    checkpoint["model"][k] = torch.cat((v,ones_bias_box_weight), 0)
                elif key.split('.')[-1] == 'bias': 
                    checkpoint["model"][k] = torch.cat((v,ones_bias_box), 0)

    return checkpoint

# load params
if __name__ == "__main__":

    checkpoint, ensem_ts_model = build_model()

    #ensem_ts_model.load_state_dict(checkpoint["model"], strict=True)

    save_model_without_states(checkpoint, True)
    #save_new_model(checkpoint, args)

    #testCheckpoint = torch.load(os.path.join(args.save_path, 'model_extended_best.pth'), map_location=torch.device("cpu"))



   