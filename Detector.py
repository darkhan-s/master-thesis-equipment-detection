#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

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
from adapteacher.data.datasets.builtin import register_all_tless, register_pump_datasets
from adapteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
import detectron2.data.transforms as T




class Detector:

    def __init__(self, args):
        self.args = args
        self.modelpath = args.config_file
        self.weights_path = args.weights_file

        self.Metadata = {}
        
        # obtain detectron2's default config
        self.cfg = self.setup()
        self.aug = T.ResizeShortestEdge([self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], self.cfg.INPUT.MAX_SIZE_TEST)

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
        cfg.freeze()
        
        print('Model config file has been loaded succcessfully', file=sys.stdout)
        return cfg

    def load_model(self):
        
        self.class_names = self.cfg.MODEL.ROI_HEADS.OLD_CLASSES + list(set(self.cfg.MODEL.ROI_HEADS.NEW_CLASSES) - set(self.cfg.MODEL.ROI_HEADS.OLD_CLASSES))
    
        # switch between datasets. Continual learning not yet tested for pumps as only one object is provided
        if self.args.mode == 0:
            register_all_tless(self.args.dataset_path, self.class_names, debug_limit = self.cfg.DATALOADER.DEBUG_LIMIT_INPUT)
        elif self.args.mode == 1:
            register_pump_datasets(self.args.dataset_path, self.class_names, debug_limit = self.cfg.DATALOADER.DEBUG_LIMIT_INPUT)
        
        self.Metadata["thing_classes"] = self.class_names
        
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
            
            print('Model .pth weights are ready', file=sys.stdout)
        else:
            raise ValueError("Trainer Name is not found.")
    

    def predict(self, im):
        
        print('Attempting to predict', file=sys.stdout)
        try:
            with torch.no_grad():  
                im = im[:, :, ::-1]
                height, width = im.shape[:2]
                image = self.aug.get_transform(im).apply_image(im)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

                inputs = {"image": image, "height": height, "width": width}
                print(inputs)
                predictions = self.model([inputs])[0]


                print(predictions, file=sys.stdout)

                v = Visualizer(im, metadata = self.Metadata)
                v = v.draw_instance_predictions(predictions["instances"].to("cpu"))                
                # get image 
                print('Drawing predictions..', file=sys.stdout)
                img = Image.fromarray(np.uint8(v.get_image()[:, :, ::-1]))
                return img  
        except Exception:
                print(traceback.format_exc(), file=sys.stdout)
                return None
