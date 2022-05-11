# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager

__all__ = ["load_voc_instances", "register_pascal_voc"]


# fmt: off
# CLASS_NAMES = (
#     "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
#     "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
#     "pottedplant", "sheep", "sofa", "train", "tvmonitor"
# )
CLASS_NAMES = ('Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5',
        'Model 6', 'Model 7', 'Model 8', 'Model 9', 'Model 10', 'Model 11',
        'Model 12', 'Model 13', 'Model 14', 'Model 15', 'Model 16', 'Model 17',
        'Model 18', 'Model 19', 'Model 20', 'Model 21', 'Model 22', 'Model 23',
        'Model 24', 'Model 25', 'Model 26', 'Model 27', 'Model 28', 'Model 29', 'Model 30'
        )
# fmt: on


def load_tless_voc_instances(dirname: str, split: str, class_names: Union[List[str], Tuple[str, ...]], ext: str):
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
            instances.append(
                {"category_id": class_names.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )
        r["annotations"] = instances
        dicts.append(r)
    return dicts


def register_tless_voc(name, dirname, split, year, ext, class_names=CLASS_NAMES):
    DatasetCatalog.register(name, lambda: load_tless_voc_instances(dirname, split, class_names, ext))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split
    )

if __name__ == "__main__":
    import random
    import cv2
    from detectron2.utils.visualizer import Visualizer
    import argparse
    print("Started..")
    # Parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="trainval")
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()
    
    print("Registering the dataset..")
    dataset_name = f"tless_{args.split}"
    register_tless_voc(name="tless_trainval", dirname="/scratch/project_2005695/PyTorch-CycleGAN/datasets/TLessReal", year=2012, split ="trainval", ext=".png")
    dataset_dicts = DatasetCatalog.get(dataset_name)
    for d in random.sample(dataset_dicts, args.samples):
        print(d["file_name"])
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1],
                                metadata=MetadataCatalog.get(dataset_name),
                                scale=args.scale)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imwrite("test.jpg", vis.get_image()[:, :, ::-1])
        # Exit? Press ESC
        break

