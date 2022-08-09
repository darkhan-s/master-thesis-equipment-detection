# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager

#@darkhan-s the model loads custom VOC style datasets as defined here  

__all__ = ["load_tless_voc_instances", "register_tless_voc", "register_pumps_voc"]


CLASS_NAMES = ('Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5',
        'Model 6', 'Model 7', 'Model 8', 'Model 9', 'Model 10', 'Model 11',
        'Model 12', 'Model 13', 'Model 14', 'Model 15', 'Model 16', 'Model 17',
        'Model 18', 'Model 19', 'Model 20', 'Model 21', 'Model 22', 'Model 23',
        'Model 24', 'Model 25', 'Model 26', 'Model 27', 'Model 28', 'Model 29', 'Model 30'
        )

PUMP_CLASSES = ('Pump HM-75S')

def load_tless_voc_instances(dirname: str, split: str, class_names: Union[List[str], Tuple[str, ...]], ext: str, debug_limit = 0, old_classes = None, new_classes = None):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=str)

    # Needs to read many small annotation files. Makes sense at local
    annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))
    dicts = []

    # temporary limit to speed up debugging
    size = 0

    print("Loading only images that contain classes {}.".format(class_names))
    for fileid in fileids:
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ext)

        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            # difficult = int(obj.find("difficult").text)
            # if difficult == 1:
            # continue
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            
            # @darkhan-s We want to load only some classes and evaluate if the model can transfer well to new classes           
            if cls in class_names:
                instances.append(
                    {"category_id": class_names.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS})
            
        # if len(instances)>0:    
        r["annotations"] = instances
        dicts.append(r)
        size += 1

        if size > debug_limit and debug_limit > 0:
            break
    return dicts


def register_tless_voc(name, dirname, split, year, ext, debug_limit, class_names=CLASS_NAMES):
    DatasetCatalog.register(name, lambda: load_tless_voc_instances(dirname, split, class_names, ext, debug_limit=debug_limit))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split
    )

def register_pumps_voc(name, dirname, split, year, ext, debug_limit, class_names=PUMP_CLASSES):
    DatasetCatalog.register(name, lambda: load_tless_voc_instances(dirname, split, class_names, ext, debug_limit=debug_limit))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split
    )


