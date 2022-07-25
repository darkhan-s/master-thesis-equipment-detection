#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import final
import cv2
import torch
import numpy as np
from PIL import Image
import sys
import traceback

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

from adapteacher import add_ateacher_config
from adapteacher.engine.trainer import ATeacherTrainer

# hacky way to register
from adapteacher.modeling.meta_arch.rcnn import DAobjTwoStagePseudoLabGeneralizedRCNN
from adapteacher.modeling.proposal_generator.rpn import PseudoLabRPN

from detectron2.modeling.backbone import (
    ResNet,
    Backbone,
    build_resnet_backbone,
    BACKBONE_REGISTRY
)

from adapteacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
import adapteacher.data.datasets.builtin
from adapteacher.data.datasets.builtin import register_all_tless
from adapteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel

TLESS_CLASS_NAMES = ["Model 1", "Model 2", "Model 3", "Model 4", "Model 5",
        "Model 6", "Model 7", "Model 8", "Model 9", "Model 10", "Model 11",
        "Model 12", "Model 13", "Model 14", "Model 15", "Model 16", "Model 17",
        "Model 18", "Model 19", "Model 20", "Model 21", "Model 22", "Model 23",
        "Model 24", "Model 25", "Model 26", "Model 27", "Model 28", "Model 29", "Model 30"
        ]

class Metadata:
    def get(self, _):
        return TLESS_CLASS_NAMES #your class labels


class Detector:

    def __init__(self):

        # set model and test set
        self.modelpath = 'faster_rcnn_R101_cross_tless_full.yaml'
        # self.weights_path = 'output-mymodel-classes-1-30-FINAL-MyModel_withCustomAugmentation-origScheduler/model_best.pth'
        self.weights_path = 'output-original/model_0044999.pth'

        # obtain detectron2's default config
        self.cfg = self.setup()
        self.load_model()


    def setup(self):
        """
        Create configs and perform basic setups.
        """
        cfg = get_cfg()
        add_ateacher_config(cfg)
        cfg.merge_from_file('configs/' + self.modelpath)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
        cfg.MODEL.DEVICE = 'cpu' 
        cfg.MODEL.WEIGHTS = self.weights_path
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 30
        cfg.freeze()
        
        print('Model cfg is all set', file=sys.stdout)
        return cfg

    def load_model(self):
        
        class_names = self.cfg.MODEL.ROI_HEADS.OLD_CLASSES + list(set(self.cfg.MODEL.ROI_HEADS.NEW_CLASSES) - set(self.cfg.MODEL.ROI_HEADS.OLD_CLASSES))
        TLESS_CLASS_NAMES = class_names
        register_all_tless("/scratch/project_2005695/PyTorch-CycleGAN/datasets/", class_names, debug_limit = self.cfg.DATALOADER.DEBUG_LIMIT_INPUT)

        if self.cfg.SEMISUPNET.Trainer == "ateacher":
            Trainer = ATeacherTrainer
            model = Trainer.build_model(self.cfg)
            
            model_teacher = Trainer.build_model(self.cfg)
            ensem_ts_model = EnsembleTSModel(model_teacher, model)

            DetectionCheckpointer(
                ensem_ts_model, save_dir=self.cfg.OUTPUT_DIR
            ).load(self.weights_path)
            self.model = ensem_ts_model.modelTeacher
            
            self.model.eval()
            
            print('Model .pth is all set', file=sys.stdout)
        else:
            raise ValueError("Trainer Name is not found.")
      
        #im = cv2.imread("/scratch/project_2005695/PyTorch-CycleGAN/datasets/TLessRendered/JPEGImages/000003_000001.jpg")

        #results = predict(im, finalModel)
        #print(results)
        #return results

    def predict(self, im):
        
        print('Attempting to predict', file=sys.stdout)
        try:
            with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
                im = im[:, :, ::-1]
                height, width = im.shape[:2]
                #im = ResizeShortestEdge.get_transform(im).apply_image(im)
                image = torch.as_tensor(im.astype("float32").transpose(2, 0, 1))

                inputs = {"image": image, "height": height, "width": width}
                predictions = self.model([inputs])[0]
                

                print(predictions, file=sys.stdout)

                v = Visualizer(im, metadata=Metadata)
                v = v.draw_instance_predictions(predictions["instances"].to("cpu"))
                #cv2.imwrite("/scratch/project_2005695/master-thesis-equipment-detection/misc/adaptive_teacher/output/TLessReal_predictions/predictions.jpg", out.get_image()[..., ::-1][..., ::-1])
                
                # get image 
                print('Drawing predictions..', file=sys.stdout)
                img = Image.fromarray(np.uint8(v.get_image()[:, :, ::-1]))
                return img  
        except Exception:
                print(traceback.format_exc(), file=sys.stdout)
                return None
