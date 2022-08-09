#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

from adapteacher import add_ateacher_config
from adapteacher.engine.trainer import ATeacherTrainer

# hacky way to register
from adapteacher.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN, DAobjTwoStagePseudoLabGeneralizedRCNN
# from adapteacher.modeling.meta_arch.vgg import build_vgg_backbone  # noqa
from adapteacher.modeling.proposal_generator.rpn import PseudoLabRPN
from adapteacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
import adapteacher.data.datasets.builtin

from adapteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import cv2
import torch
import numpy as np

from fvcore.transforms.transform import (
    NoOpTransform,
)
from detectron2.data.transforms.augmentation_impl import ResizeShortestEdge

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ateacher_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3 
    cfg.MODEL.DEVICE = 'cpu' 
    cfg.MODEL.WEIGHTS = 'output/model_0029999.pth'
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)
    if cfg.SEMISUPNET.Trainer == "ateacher":
        Trainer = ATeacherTrainer
    else:
        raise ValueError("Trainer Name is not found.")

    if cfg.SEMISUPNET.Trainer == "ateacher":
        model = Trainer.build_model(cfg)
        
        model_teacher = Trainer.build_model(cfg)
        ensem_ts_model = EnsembleTSModel(model_teacher, model)

        DetectionCheckpointer(
            ensem_ts_model, save_dir=cfg.OUTPUT_DIR
        ).load('output/model_0029999.pth')
        finalModel = ensem_ts_model.modelTeacher
        

    else:
        finalModel = Trainer.build_model(cfg)
        DetectionCheckpointer(finalModel, save_dir=cfg.OUTPUT_DIR).load(
            'output/model_0004999.pth'
        )
    finalModel.eval()
    #predictor = DefaultPredictor(cfg)
    
    im = cv2.imread("/scratch/project_2005695/PyTorch-CycleGAN/datasets/TLessRendered/JPEGImages/000003_000001.jpg")

    with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
        im = im[:, :, ::-1]
        height, width = im.shape[:2]
        #im = ResizeShortestEdge.get_transform(im).apply_image(im)
        image = torch.as_tensor(im.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}
        predictions = finalModel([inputs])[0]
        

        print(predictions)

        v = Visualizer(im)
        out = v.draw_instance_predictions(predictions["instances"].to("cpu"))
        cv2.imwrite("/scratch/project_2005695/master-thesis-equipment-detection/misc/adaptive_teacher/output/TLessReal_predictions/predictions.jpg", out.get_image()[..., ::-1][..., ::-1])
    
    im = cv2.imread("/scratch/project_2005695/PyTorch-CycleGAN/datasets/TLessReal/JPEGImages/000004_000001.png")

    with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
        height, width = im.shape[:2]
        image = torch.as_tensor(im.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}
        predictions = finalModel([inputs])[0]
        

        print(predictions)

        v = Visualizer(im)
        out = v.draw_instance_predictions(predictions["instances"].to("cpu"))
        cv2.imwrite("/scratch/project_2005695/master-thesis-equipment-detection/misc/adaptive_teacher/output/TLessReal_predictions/predictions2.jpg", out.get_image()[..., ::-1][..., ::-1])

    return None



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
