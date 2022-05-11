# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from fvcore.common.timer import Timer
# from fvcore.common.file_io import PathManager
from iopath.common.file_io import PathManager
import logging

from .tless_voc import register_tless_voc

logger = logging.getLogger(__name__)

## _root = "/scratch/project_2005695/PyTorch-CycleGAN/datasets/"
# ==== Predefined splits for TLess (PASCAL VOC format) ===========
def register_all_tless(root):
    logger.info("Registering happened here")
    # root = "manifold://mobile_vision_dataset/tree/yujheli/dataset"
    SPLITS = [
        ("tless_rendered_trainval", "TLessRendered", "trainval", ".jpg"),
        ("tless_rendered_test", "TLessRendered", "test", ".jpg"),
        ("tless_real_trainval", "TLessReal", "trainval", ".png"),
        ("tless_real_test", "TLessReal", "test", ".png")
    ]
    for name, dirname, split, ext in SPLITS:
        year = 2012
        register_tless_voc(name, os.path.join(root, dirname), split, year, ext)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"


##register_all_tless(_root)

