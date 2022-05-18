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
from adapteacher.engine.trainer import ATeacherTrainer, BaselineTrainer

# hacky way to register
from adapteacher.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN, DAobjTwoStagePseudoLabGeneralizedRCNN
from adapteacher.modeling.meta_arch.vgg import build_vgg_backbone  # noqa
from adapteacher.modeling.proposal_generator.rpn import PseudoLabRPN
from adapteacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
import adapteacher.data.datasets.builtin
from adapteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel

class Detector:

    def __init__(self):

        # set model and test set
        self.modelpath = 'faster_rcnn_R101_cross_tless.yaml'

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
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3 
        cfg.MODEL.DEVICE = 'cpu' 
        cfg.MODEL.WEIGHTS = 'output-original/model_0029999.pth'
        cfg.freeze()
        
        print('Model cfg is all set', file=sys.stdout)
        return cfg

    def load_model(self):
        if self.cfg.SEMISUPNET.Trainer == "ateacher":
            Trainer = ATeacherTrainer
        elif self.cfg.SEMISUPNET.Trainer == "baseline":
            Trainer = BaselineTrainer
        else:
            raise ValueError("Trainer Name is not found.")

        if self.cfg.SEMISUPNET.Trainer == "ateacher":
            model = Trainer.build_model(self.cfg)
            
            model_teacher = Trainer.build_model(self.cfg)
            ensem_ts_model = EnsembleTSModel(model_teacher, model)

            DetectionCheckpointer(
                ensem_ts_model, save_dir=self.cfg.OUTPUT_DIR
            ).load('output-original/model_0029999.pth')
            self.model = ensem_ts_model.modelTeacher
            

        else:
            self.model = Trainer.build_model(self.cfg)
            DetectionCheckpointer(self.model, save_dir=self.cfg.OUTPUT_DIR).load(
                'output/model_0004999.pth'
            )
        self.model.eval()
        
        print('Model .pth is all set', file=sys.stdout)
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

                v = Visualizer(im)
                v = v.draw_instance_predictions(predictions["instances"].to("cpu"))
                #cv2.imwrite("/scratch/project_2005695/master-thesis-equipment-detection/misc/adaptive_teacher/output/TLessReal_predictions/predictions.jpg", out.get_image()[..., ::-1][..., ::-1])
                
                # get image 
                print('Drawing predictions..', file=sys.stdout)
                img = Image.fromarray(np.uint8(v.get_image()[:, :, ::-1]))
                return img  
        except Exception:
                print(traceback.format_exc(), file=sys.stdout)
                return None
