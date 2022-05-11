from detectron2.config import LazyConfig
from omegaconf import OmegaConf
from detectron2.utils.registry import _convert_target_to_string
from detectron2.config import LazyConfig, get_cfg
import os
from collections import abc


def get_config_file(config_path):
    """
    Returns path to a builtin config file.

    Args:
        config_path (str): config file name relative to detectron2's "configs/"
            directory, e.g., "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"

    Returns:
        str: the real path to the config file.
    """
    # cfg_file = pkg_resources.resource_filename(
    #     "detectron2.model_zoo", os.path.join("configs", config_path)
    # )
    cfg_file=config_path
    if not os.path.exists(cfg_file):
        raise RuntimeError("{} not available in Model Zoo!".format(config_path))
    return cfg_file


def get_config(config_path, trained: bool = False):
    """
    Returns a config object for a model in model zoo.

    Args:
        config_path (str): config file name relative to detectron2's "configs/"
            directory, e.g., "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
        trained (bool): If True, will set ``MODEL.WEIGHTS`` to trained model zoo weights.
            If False, the checkpoint specified in the config file's ``MODEL.WEIGHTS`` is used
            instead; this will typically (though not always) initialize a subset of weights using
            an ImageNet pre-trained model, while randomly initializing the other weights.

    Returns:
        CfgNode or omegaconf.DictConfig: a config object
    """
    cfg_file = get_config_file(config_path)
    if cfg_file.endswith(".yaml"):
        cfg = get_cfg()
        cfg.merge_from_file(cfg_file)
        return cfg
    elif cfg_file.endswith(".py"):
        cfg = LazyConfig.load(cfg_file)
        return cfg


def to_py(cfg, prefix: str = "cfg."):
    """
    Try to convert a config object into Python-like psuedo code.

    Note that perfect conversion is not always possible. So the returned
    results are mainly meant to be human-readable, and not meant to be executed.

    Args:
        cfg: an omegaconf config object
        prefix: root name for the resulting code (default: "cfg.")


    Returns:
        str of formatted Python code
    """
    import black

    cfg = OmegaConf.to_container(cfg, resolve=True)

    def _to_str(obj, prefix=None, inside_call=False):
        if prefix is None:
            prefix = []
        if isinstance(obj, abc.Mapping) and "_target_" in obj:
            # Dict representing a function call
            target = _convert_target_to_string(obj.pop("_target_"))
            args = []
            for k, v in sorted(obj.items()):
                args.append(f"{k}={_to_str(v, inside_call=True)}")
            args = ", ".join(args)
            call = f"{target}({args})"
            return "".join(prefix) + call
        elif isinstance(obj, abc.Mapping) and not inside_call:
            # Dict that is not inside a call is a list of top-level config objects that we
            # render as one object per line with dot separated prefixes
            key_list = []
            for k, v in sorted(obj.items()):
                if isinstance(v, abc.Mapping) and "_target_" not in v:
                    key_list.append(_to_str(v, prefix=prefix + [k + "."]))
                else:
                    key = "".join(prefix) + k
                    key_list.append(f"{key}={_to_str(v)}")
            return "\n".join(key_list)
        elif isinstance(obj, abc.Mapping):
            # Dict that is inside a call is rendered as a regular dict
            return (
                "{"
                + ",".join(
                    f"{repr(k)}: {_to_str(v, inside_call=inside_call)}"
                    for k, v in sorted(obj.items())
                )
                + "}"
            )
        elif isinstance(obj, list):
            return "[" + ",".join(_to_str(x, inside_call=inside_call) for x in obj) + "]"
        else:
            return repr(obj)

    py_str = _to_str(cfg, prefix=[prefix])
    try:
        return black.format_str(py_str, mode=black.Mode())
    except black.InvalidInput:
        return py_str

pseudo_config = to_py(get_config("src/cross_teacher_fcos/configs/fcos_R_50_FPN_1x.py"))

f = open("src/cross_teacher_fcos/configs/pseudo_config_fcos.txt", "w")
f.write(pseudo_config)
f.close()

