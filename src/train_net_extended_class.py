#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_setup, launch

from adapteacher import add_ateacher_config
from adapteacher.engine.trainer import ATeacherTrainer

# hacky way to register
from adapteacher.modeling.meta_arch.rcnn import DAobjTwoStagePseudoLabGeneralizedRCNN
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

# @darkhan-s to monitor training in real time
import tensorboard

import sys
import os
# @darkhan-s add some custom args 
import argparse

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
    
    # switch between datasets. Continual learning not yet tested for pumps as only one object is provided
    if args.mode == 0:
        register_all_tless(args.dataset_path, class_names, debug_limit = cfg.DATALOADER.DEBUG_LIMIT_INPUT)
    elif args.mode == 1:
        register_pump_datasets(args.dataset_path, class_names, debug_limit = cfg.DATALOADER.DEBUG_LIMIT_INPUT)
    #register_all_tless("/scratch/project_2005695/PyTorch-CycleGAN/datasets/", class_names, debug_limit = cfg.DATALOADER.DEBUG_LIMIT_INPUT)
    #register_pump_datasets("/scratch/project_2005695/master-thesis-equipment-detection/bin/pumps/", class_names = ["Pump HM-75S"], debug_limit = cfg.DATALOADER.DEBUG_LIMIT_INPUT)
    
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

                        # enabling gradients (we give them lower value in optimizer to preserve original weights)

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

def default_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
Examples:

Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml

Change some config options:
    $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001

Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument('-path', '--dataset_path', help='dataset in VOC format with the path for both domains')
    
    # this parameter should be utilized
    parser.add_argument('--finetune', action='store_true', help='whether to learn continuously or not')

    parser.add_argument(
        '--mode', '-m',
        help='Set mode for training, 0 is for training TLess, 1 is for training pumps.',
        default=1,
        type=int,
        choices=[0,1],
    )

    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


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
