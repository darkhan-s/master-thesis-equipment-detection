# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import contextlib
from detectron2.data import DatasetCatalog, MetadataCatalog
from fvcore.common.timer import Timer
from iopath.common.file_io import PathManager

from detectron2.data.datasets.pascal_voc import register_pascal_voc
from .tless_voc import register_tless_voc, register_pumps_voc

from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
import io
import logging

logger = logging.getLogger(__name__)

JSON_ANNOTATIONS_DIR = ""
_SPLITS_COCO_FORMAT = {}
_SPLITS_COCO_FORMAT["coco"] = {
    "coco_2017_unlabel": (
        "coco/unlabeled2017",
        "coco/annotations/image_info_unlabeled2017.json",
    ),
    "coco_2017_for_voc20": (
        "coco",
        "coco/annotations/google/instances_unlabeledtrainval20class.json",
    ),
}


def register_coco_unlabel(root):
    for _, splits_per_dataset in _SPLITS_COCO_FORMAT.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            meta = {}
            register_coco_unlabel_instances(
                key, meta, os.path.join(root, json_file), os.path.join(root, image_root)
            )


def register_coco_unlabel_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root

    # 1. register a function which returns dicts
    DatasetCatalog.register(
        name, lambda: load_coco_unlabel_json(json_file, image_root, name)
    )

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


def load_coco_unlabel_json(
    json_file, image_root, dataset_name=None, extra_annotation_keys=None
):
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info(
            "Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds())
        )

    id_map = None
    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())

    imgs = coco_api.loadImgs(img_ids)

    logger.info("Loaded {} images in COCO format from {}".format(len(imgs), json_file))

    dataset_dicts = []

    for img_dict in imgs:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        dataset_dicts.append(record)

    return dataset_dicts


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_coco_unlabel(_root)



# @darkhan-s ==== Predefined splits for TLess (PASCAL VOC format) ===========
def register_all_tless(root, class_names, debug_limit=0):
    SPLITS = [
        ("tless_rendered_trainval", "TLessRendered", "trainval", ".jpg"),
        ("tless_rendered_test", "TLessRendered", "test", ".jpg"),
        ("tless_real_trainval", "TLessReal", "trainval", ".png"),
        ("tless_real_test", "TLessReal", "test", ".png")
    ]
    for name, dirname, split, ext in SPLITS:
        year = 2012
        register_tless_voc(name, os.path.join(root, dirname), split, year, ext, debug_limit = debug_limit, class_names = class_names)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc_tless"

# @darkhan-s ==== Predefined splits for pumps (PASCAL VOC format) ===========
def register_pump_datasets(root, class_names, debug_limit=0):
    SPLITS = [
        ("pumps_rendered_trainval", "PumpsRendered", "trainval", ".jpg"),
        ("pumps_rendered_test", "PumpsRendered", "test", ".jpg"),
        ("pumps_real_trainval", "PumpsReal", "trainval", ".JPEG"),
        ("pumps_real_test", "PumpsReal", "test", ".JPEG")
    ]
    for name, dirname, split, ext in SPLITS:
        year = 2012
        register_pumps_voc(name, os.path.join(root, dirname), split, year, ext, debug_limit = debug_limit, class_names = class_names)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc_tless"


