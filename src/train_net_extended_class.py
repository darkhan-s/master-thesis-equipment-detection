#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

from adapteacher import add_ateacher_config
from adapteacher.engine.trainer import ATeacherTrainer

# hacky way to register
from adapteacher.modeling.meta_arch.rcnn import DAobjTwoStagePseudoLabGeneralizedRCNN
# from adapteacher.modeling.meta_arch.vgg import build_vgg_backbone  # noqa
from detectron2.modeling.backbone import (
    ResNet,
    Backbone,
    build_resnet_backbone,
    BACKBONE_REGISTRY
)
from adapteacher.modeling.proposal_generator.rpn import PseudoLabRPN
from adapteacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
import adapteacher.data.datasets.builtin
from adapteacher.data.datasets.builtin import register_all_tless, register_pump_datasets

from adapteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
import tensorboard

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ateacher_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    if cfg.SEMISUPNET.Trainer == "ateacher":
        Trainer = ATeacherTrainer
    else:
        raise ValueError("Trainer Name is not found.")
    ## @darkhan-s
    class_names = cfg.MODEL.ROI_HEADS.OLD_CLASSES + list(set(cfg.MODEL.ROI_HEADS.NEW_CLASSES) - set(cfg.MODEL.ROI_HEADS.OLD_CLASSES))
    
    # switch between two datasets (manual for now)
    register_all_tless("/scratch/project_2005695/PyTorch-CycleGAN/datasets/", class_names, debug_limit = cfg.DATALOADER.DEBUG_LIMIT_INPUT)
    register_pump_datasets("/scratch/project_2005695/master-thesis-equipment-detection/bin/pumps/", class_names = ["Pump HM-75S"], debug_limit = cfg.DATALOADER.DEBUG_LIMIT_INPUT)
    
    if args.eval_only:
        if cfg.SEMISUPNET.Trainer == "ateacher":
            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            ensem_ts_model = EnsembleTSModel(model_teacher, model)

            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            res = Trainer.test(cfg, ensem_ts_model.modelTeacher)

        else:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)

    ## @darkhan-s Block to retrain only the last layer of the network 
    classesAdded = list(set(cfg.MODEL.ROI_HEADS.NEW_CLASSES) - set(cfg.MODEL.ROI_HEADS.OLD_CLASSES))
    
    if args.resume and len(classesAdded)>0:
        print(f"New classes are requested to train: {classesAdded}. Freezing the original model weights and appending/training extra classifier layer..")

        keysToModify = [
            
                        # enabling gradients for added classes
                        'modelStudent.roi_heads.box_predictor.cls_score_extra_classes.weight', 
                        'modelStudent.roi_heads.box_predictor.cls_score_extra_classes.bias',
                        'modelStudent.roi_heads.box_predictor.bbox_pred_extra_classes.weight', 
                        'modelStudent.roi_heads.box_predictor.bbox_pred_extra_classes.bias',  
                        'modelTeacher.roi_heads.box_predictor.cls_score_extra_classes.weight', 
                        'modelTeacher.roi_heads.box_predictor.cls_score_extra_classes.bias',
                        'modelTeacher.roi_heads.box_predictor.bbox_pred_extra_classes.weight', 
                        'modelTeacher.roi_heads.box_predictor.bbox_pred_extra_classes.bias',

                        # enabling gradients (we set them lower in optimizer to preserve original weights)

                        'modelStudent.roi_heads.box_predictor.cls_score.weight', 
                        'modelStudent.roi_heads.box_predictor.cls_score.bias',
                        'modelStudent.roi_heads.box_predictor.bbox_pred.weight', 
                        'modelStudent.roi_heads.box_predictor.bbox_pred.bias',  
                        'modelTeacher.roi_heads.box_predictor.cls_score.weight', 
                        'modelTeacher.roi_heads.box_predictor.cls_score.bias',
                        'modelTeacher.roi_heads.box_predictor.bbox_pred.weight', 
                        'modelTeacher.roi_heads.box_predictor.bbox_pred.bias',

                        ]

        
        for name, value in trainer.ensem_ts_model.named_parameters():
            value.requires_grad = False        
            for layer in keysToModify:
                if layer in name:
                    value.requires_grad = True
                    print(f'{name} requires_grad is set {value.requires_grad}')
        
        cfg.defrost()
        cfg.MODEL.BACKBONE.FREEZE_AT = 5
        cfg.freeze()
        

    trainer.resume_or_load(resume=args.resume)
    
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
