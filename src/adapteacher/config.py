# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_ateacher_config(cfg):
    """
    Add config for semisupnet.
    """
    _C = cfg
    _C.MODEL.RESNETS.OUT_FEATURES = ["res4"]
    _C.MODEL.RPN.UNSUP_LOSS_WEIGHT = 1.0
    _C.MODEL.RPN.LOSS = "CrossEntropy"
    _C.MODEL.ROI_HEADS.LOSS = "CrossEntropy"
    
    # @darkhan-s here we only initialize the list
    # The classes used are listed in the .yaml file:
    _C.MODEL.ROI_HEADS.OLD_CLASSES = []
    _C.MODEL.ROI_HEADS.NEW_CLASSES = []
   
    _C.SOLVER.IMG_PER_BATCH_LABEL = 2
    _C.SOLVER.IMG_PER_BATCH_UNLABEL = 2
    _C.SOLVER.FACTOR_LIST = (1,)

    # @darkhan-s save best weights 
    _C.SOLVER.BEST_CHECKPOINTER = CN({"ENABLED": True})
    _C.SOLVER.BEST_CHECKPOINTER.METRIC = "bbox_teacher/AP50"
    _C.SOLVER.BEST_CHECKPOINTER.MODE = "max"
    _C.SOLVER.BEST_CHECKPOINTER.PATIENCE = 10 # wait for 10 iterations of bbox/AP50 not increasing before termination 

    # @darkhan-s for fine tuning with more classes
    _C.SOLVER.BASE_CLASSIFIER_LR_FACTOR = CN()
    _C.SOLVER.BASE_CLASSIFIER_LR_FACTOR = 0.001 # multiply base box predictor LR by N to keep weights closer to original


    # @darkhan-s set custom datasets
    _C.DATASETS.TRAIN_LABEL = ("tless_rendered_trainval",)
    _C.DATASETS.TRAIN_UNLABEL = ("tless_real_trainval",)
    _C.TEST.EVALUATOR = "pascal_voc"

    # @darkhan-stesting with cropping weak augm enabled
    _C.INPUT.CROP.ENABLED = False
    _C.INPUT.ROTATION_ENABLED = False
    _C.INPUT.AFFINE_ENABLED = False
    _C.INPUT.CUSTOM_AUGMENTATIONS = False

    _C.SEMISUPNET = CN()

    # Output dimension of the MLP projector after `res5` block
    _C.SEMISUPNET.MLP_DIM = 128

    # Semi-supervised training
    _C.SEMISUPNET.Trainer = "ateacher"
    _C.SEMISUPNET.BBOX_THRESHOLD = 0.7
    _C.SEMISUPNET.PSEUDO_BBOX_SAMPLE = "thresholding"
    _C.SEMISUPNET.TEACHER_UPDATE_ITER = 1
    _C.SEMISUPNET.BURN_UP_STEP = 12000
    _C.SEMISUPNET.EMA_KEEP_RATE = 0.0
    _C.SEMISUPNET.UNSUP_LOSS_WEIGHT = 4.0
    _C.SEMISUPNET.SUP_LOSS_WEIGHT = 0.5
    _C.SEMISUPNET.LOSS_WEIGHT_TYPE = "standard"
    _C.SEMISUPNET.DIS_TYPE = "res4"


    # @darkhan-s optimal is 0.07 for all
    _C.SEMISUPNET.DIS_LOSS_WEIGHT = 0.07 # was 0.1
    _C.SEMISUPNET.DIS_INST_LOSS_WEIGHT = 0.07
    _C.SEMISUPNET.CONSISTENCY_LOSS_WEIGHT = 0.07

    # @darkhan-s dataloader
    _C.DATALOADER.DEBUG_LIMIT_INPUT = 0 # restrict the amount of images to register for faster debug, set 0 to disable

    _C.EMAMODEL = CN()
    _C.EMAMODEL.SUP_CONSIST = True
